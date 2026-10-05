# H3 本地参考精度合同核查（2026-10-04）

目的：为 E058 的 scheduler 舍入分解和未来强基线确定边界。只读本地源码、已安装包的 RECORD、pruned checkpoint header；没有模型前向、GPU 初始化、代码/基线修改或自动重跑。本页不把 FP32 accumulator 当创新，也不宣布过去内部对照失效。

## 可执行判断

**当前冻结的 Comfy pruned + DiffSynth BF16 实验路径，与本地已安装 Diffusers 0.40.0 的 H3 默认精度合同存在真实差异。** 前者保存 BF16 latent、接收 BF16 velocity，并执行 BF16 Euler 乘加；后者保存 FP32 video/audio latent、以 FP32 velocity 调用 FP32 scheduler，并保留一组 FP32 投影/时间模块。更改 accumulator 是已有参考实现采用的普通工程基线。

这不等于“Diffusers 就是当前同一 20.1B pruned 模型”。两者时间条件结构不同；checkpoint 身份、全量参数映射和数值等价尚未核实。当前 plain/SVD/BF16 同一冻结实现下的配对证据仍成立；其适用范围是该实现。E057 endpoint interaction 不能在 E058 分解前全归为 DiT 非线性，未来也不能把克服旧实现舍入问题本身包装成量化创新。

## 1. 当前实验实际合同

- [公共加载入口](/home/wjq/workspace/svdquant-exp/scripts/minimax_h3_svdquant_common.py:89)：disk_config 的 preparing/computation dtype 均为 BF16（93–94）；132–135 显式创建 BF16 pipeline；136–137 强制 `MiniMaxH3DiTComfyPruned`。模型路径在29–34，Comfy 模型名在47–50。
- [E014/E057 复用加载](/home/wjq/workspace/svdquant-exp/scripts/research/probe_h3_plain_baseline.py:216)：加载、materialize 后调用 resident conversion。[resident conversion](/home/wjq/workspace/svdquant-exp/scripts/research/bench_h3_native_nvfp4.py:54) 只解包已验证 wrapper，88–109 强制其计算 BF16；非 wrapper dtype 保留，69–83 对原 `rope.inv_freq` 按保存 dtype 物化。因此不能笼统称“整个模型所有张量都是 BF16”。
- [DiffSynth H3 pipeline](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:26) 默认 BF16；243、245 以 `pipe.torch_dtype` 生成 video/audio noise；187–194 将 scheduler 结果直接存为下一步 latent。[scheduler](/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py:377) 第386行直接 `sample + model_output * (sigma_ - sigma)`，没有 `.float()` accumulator。
- [冻结状态推进函数](/home/wjq/workspace/svdquant-exp/scripts/research/prepare_h3_crossmodal_states.py:71) 明确检查 sample/velocity/result 都是 BF16（85–92），timestep/sigma 为原 0D FP32（60–67）。0D FP32 scale 不意味着整条 BF16 tensor 运算自动升 FP32。
- [pruned 专有实现](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:69) 注册 FP32 `adaln_t_table[1025,8]`，替换原时间 MLP，并把各 block/最终层 AdaLN 换为曲线投影；39–45 用 FP32 table/插值，最后 cast 到指定 dtype。[当前 embedding](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:301) 以 text embedding dtype 投影 video/audio（302–306），时间条件输出也按该 dtype（316）。

## 2. 本地 Diffusers 参考：状态和模块均有不同默认

已安装根目录：`/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/`。[METADATA](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers-0.40.0.dist-info/METADATA:2) 为 `diffusers 0.40.0`，标注 Hugging Face 项目。这里核实的是本地安装分发中的代码，不是联网核实 MiniMax 原始发布仓库。

| 项目 | 精确本地证据 | 观察 |
|---|---|---|
| FP32 persistent state | [before_denoise.py](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/modular_pipelines/minimax_h3/before_denoise.py:853) | video random FP32（863），已有 video 也转 FP32（865）；audio random/传入均 FP32（872/876），881保存两个状态。 |
| FP32 scheduler operands | [denoise.py](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/modular_pipelines/minimax_h3/denoise.py:225) | 226/232 对两路 velocity `.float()`，sample 是上述 FP32 latent，225/231原位写回。 |
| Euler blend | [scheduling_minimax_h3.py](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/schedulers/scheduling_minimax_h3.py:260) | 266–269由 transformer timestep 重建 sigma，先构造 denoised；272–277用 FP32 ratio/blend，返回 sample dtype。单独给该 scheduler BF16 sample 会先在 BF16形成 denoised、最后cast BF16，不能把这个局部上采样等同于 pipeline 的全程 FP32 state。 |
| FP32 模块 | [transformer_minimax_h3.py](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/models/transformers/transformer_minimax_h3.py:440) | `_keep_in_fp32_modules` 包含 `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out`, `rope`（444–450）；不是全部 AdaLN 保FP32。注释说明主体/文本等为BF16。 |
| 显式按模块 dtype 计算 | [transformer forward](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/models/transformers/transformer_minimax_h3.py:624) | 628–630输入按对应 projection 参数 dtype 转换；633–636 packed hidden 以 text dtype存储；642 time embedding 按自身参数 dtype；658–660输出头按输出参数 dtype。 |

四个关键 `.py` 文件已分别计算 SHA256，并逐个匹配安装分发 `diffusers-0.40.0.dist-info/RECORD` 中的 urlsafe-base64 SHA256，均 `True`：

| 文件（相对于上述 site-packages） | SHA256 |
|---|---|
| `diffusers/models/transformers/transformer_minimax_h3.py` | `1926b1bc15a5bebda05e3dc8cde1b3955d56641ba7f8f9d6e90c8f78c66ae30c` |
| `diffusers/modular_pipelines/minimax_h3/before_denoise.py` | `530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff` |
| `diffusers/modular_pipelines/minimax_h3/denoise.py` | `bf0224f3ac8f3bba8366599143f60d78bd14e9faf77cdb207a844549bb8c1dc4` |
| `diffusers/schedulers/scheduling_minimax_h3.py` | `307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd` |

验证边界：这是安装包 RECORD 的文件完整性校验，不是 wheel archive 整体 hash 或远端签名验证；本地未见该分发 `direct_url.json`，没有据此确定源 commit。不能把 RECORD 一致扩写为“已与 MiniMax 官方 checkpoint 逐参验证”。

FastVideo 本地副本也采用 FP32 latent/velocity：`/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_latent_preparation.py:325–363` 与 `minimax_h3_denoising.py:338–349`。其 [测试 reference helper](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/tests/local_tests/minimax_h3/_reference.py:11) 指向 `DiffusersMiniMaxH3` checkout，pin commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`；[README](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/tests/local_tests/minimax_h3/README.md:8) 指 Diffusers PR14355 和 `MiniMaxAI/MiniMax-H3`。默认本地 reference checkout 未找到，未运行其 parity 测试；这不构成额外官方等价证明。

## 3. Comfy pruned 身份、F32磁盘参数与映射缺口

本轮重读实际文件 header：`/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`，40,225,724,176 bytes，532 tensor，20,111,438,744存储元素（含buffer），header 无 `__metadata__`。未重新流式hash40GB；[已有身份核查](/home/wjq/workspace/svdquant-exp/research_state/02_problems/h3_output_geometry_entry_20261004.md) 继承 E010 的 SHA `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4`。

Header 当前证实以下本来保存为F32：video/audio patch projection W/b、video/audio output W/b、`rope.inv_freq[16]`、`adaln_t_table[1025,8]`。其中 projection/head 被当前 BF16 wrapper 加载为BF16计算；table/RoPE已有FP32保留。对应尺寸：video input `[5376,96]`、audio input `[5376,32]`、video output `[96,5376]`、audio output `[32,5376]`。

这使 `video_patch_proj ↔ proj_in`、`audio_patch_proj ↔ audio_proj_in`、`final_layer.video_out ↔ proj_out`、`final_layer.audio_out ↔ audio_proj_out` 成为**可核对候选对应**，还不是已验证转换。Diffusers默认 hidden5376、50blocks、2refiners、patch1×2×2（[484–499](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/models/transformers/transformer_minimax_h3.py:484)）与当前几何兼容，但其 time_embed_dim2688/完整time MLP与pruned table8不同。QKV排列也不能只改名：Comfy override在[17–23](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:17) 用 `[total,3,heads,head_dim]`。当前加载注册也明确区分[原始 MiniMaxH3DiT 与 ComfyPruned](/home/wjq/workspace/DiffSynth-Studio/diffsynth/configs/model_configs.py:1446)，后者在1473–1476单独注册。

没有找到/验证本地能够把这个pruned safetensors直接、等价转换到上述Diffusers完整H3类的转换器。未证明：所有权重值对应、pruning转换误差、AdaLN曲线与原时间MLP等价、QKV/位置/输出行序完全一致、相同conditioning及sigma输入下forward一致。不能据名称、相同层数或4组head尺寸就宣称同一checkpoint，也不能自动将完整模型FP32 time MLP策略替换当前pruned table。

## 4. 对 E058 和未来协议的约束

1. E058沿用冻结的输入、velocity、sigma网格，CPU分解已有endpoint四角中的低精度scheduler余项；保持确切旧endpoint记录可重放。它能回答当前E057 interaction有多少来自最后一步算术，不能模拟改为FP32 master trajectory后的新模型输入/预测，更不能同时回答FP32 head的效果。
2. 若以后建立更强precision baseline，分别冻结并说明 persistent latent dtype、Euler公式/舍入、模型输入/输出head dtype、pruned time table/AdaLN、checkpoint身份；这些改变是普通基线对齐，不作为贡献。是否值得新增模型试验由主流程单独决定，本页不授权或触发重跑。
3. 不可直接把两个scheduler互换后仍称同一20step配方。DiffSynth [339–344](/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py:339) 用N+1网格去终点、N次forward；Diffusers [133–168](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/diffusers/schedulers/scheduling_minimax_h3.py:133) N个含终点网格、N−1次forward；模型时间/velocity符号约定也不同。要比较精度须先保同一实际sigma网格和模型时钟，不能同时换调度/步数。
4. 过去内部BF16/plain/SVD三臂仍共享实际实现，是该合同下有效相对比较；新的精度合同若建立，应新增版本而不覆写旧结果。尚未有证据说FP32修正能恢复生成质量，或能消除全部跨步耦合。
