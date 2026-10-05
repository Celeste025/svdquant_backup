# E008 — 单prompt rCM-Wan BF16/nativefast 配对四步视频

2026-10-02，运行前计划；**当前仅实现，等待E007 nativefast整模SHA/profile通过，不启动GPU。** 本实验是定性成对视频证据，不是泛化测试或视频质量指标。

## 固定内容与前置检查

- 唯一prompt：旧校准集0001，现有文本为“A person is roller skating”。直接复用E007缓存 `[1,512,4096]` BF16 embedding，不加载text encoder。
- 两臂：同一rCM-Wan1.3B原始BF16权重；nativefast使用同一旧rank32 checkpoint、原smooth/LR、300个NVFP4主支、已验证 `wan_nvfp4_fastpack.pack_activation_fast`。同torch BF16 FLASH SDPA、norm/RoPE。
- 必须有完整E007 correctness、真实/合成fastpacker逐byte parity与native profile成功结果；native profile的warmup、三次重复和独立profiler输出SHA须等于E007 reference-native。fastpacker/native库与模型/checkpoint源码哈希不一致则停止。
- 固定77帧、480×832、16FPS、guidance0、sigma80，四步angles `[atan(80),1.5,1.4,1,0]`。直接复用现有`WanPipeline.prepare_latents`、rCM四步循环和`decode_spatial_tiled`，不发明新采样算法。

## Seed及严格配对

1. **只试seed1**历史候选：CUDA generator产生原FP32 initial noise，乘FP64 `t0=80/81`；若转BF16与step0缓存完全一致，保留该FP64初始latent与已推进的RNG状态。不能从BF16缓存反推丢失的FP64位。
2. 若seed1初始化不一致，保存候选输入及误差/原因；仅切换为预指定**新seed42**，明确标为新paired run，不能称历史复现，不搜索更多seed。
3. 从选定generator状态按原脚本生成四个FP32更新noise（包括最后`t_next=0`的无效noise，保持原RNG调用顺序），连同同一个FP64初始latent保存。两臂逐元素共享这些输入，各自递推自身velocity；不把teacher cache状态注入native。
4. 若seed1初始化通过，BF16四步的输入、timestep、velocity逐步与四份历史缓存完全核对；任何一步失败先保存当时latent/输出与错误，再停止，**不切seed、不替换teacherstate、不声称复现成功**。
5. 每一步记录实际60次BF16 SDPA；nativefast额外恰好300次原生GEMM、300个fastpack flag经context退出验证后才接受输出。任何非有限值或flag失败停止。保存各步velocity、更新前后FP64 latent、输入/输出SHA。

## 解码、产物与预算

- 先完整完成两臂四步denoiser，最终latent落盘，释放整个DiT及hook权重/packing临时量；之后才加载同一个BF16 `AutoencoderKLWan` 依次解码。
- VAE取本地Wan2.1-T2V-1.3B-Diffusers/vae，固定同一权重与 `latent*latents_std+latents_mean` 恢复约定；`torch.inference_mode()`，复用decoder，core128≥104、halo0，单完整空间块避免拼缝。核对输出77×480×832，统一16FPS导出两个MP4。
- 数据盘保存RNG共享输入、seed1候选、每步张量、最终latent、两条MP4；小JSON总结和日志在repo。拒绝覆写既有产物。视频只作单个校准prompt的定性比较，不运行VBench/MJVIDEO，不以latent NMSE宣称感知质量。
- GPU1，named tmux，日志 `results/logs/E008_wan_native_paired_video.log`；**外层timeout3600秒**，内部预算检查3600秒；诊断显存超过60GiB停止保留部分结果，不能减帧/裁token改变协议。

Runner：`scripts/research/run_wan_native_paired_video.py`。默认前置profile `results/research/E007_profile_native.json`，可用`--native-profile`指向经验证的同协议文件。当前不执行。
