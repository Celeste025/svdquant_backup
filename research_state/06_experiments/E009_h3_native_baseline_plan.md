# E009 — 完整 resident H3 原生 NVFP4 正确性基线

2026-10-02，预注册。只建立用户主模型的真实完整baseline，不把部署移植或既有SVDQuant称作研究创新。慢速reference packing不报告性能。

## 固定输入与范围

- 模型为本地 `MiniMaxH3DiTComfyPruned`，不是另一个官方未prune配置；source见 `native_h3_next_baseline.md`。
- 唯一样本：`results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt`，原PTQ校准prompt1 step0；全部packed token和真实cu_seqlens，不裁token、不解码、不自由生成。
- 50个主干block×qkv/out/fc1/fc2=**200目标**；token_refiner另外2blocks/8linears及所有其他tensor保留原BF16计算，不误算208目标。
- 同一recovered环境torch2.11.0+cu128，同原torch BF16 SDPA/norm/RoPE；实际调用数与dtype作为检查，不用Sage或FP4 attention。
- 旧rank32 state不再校准。systems逐层按旧GPU BF16算术重建residual并导出legacy codes/scales；必须200W独立decode对原QDQ逐元素一致，不能从CPU B@A或RNE替代legacy基线。

## 执行顺序及门槛

1. 新runner的 `--phase resident` 可独立执行：原offload模型完整前向，保存50block与video/audio参考；同一模型逐个unwrap AutoWrappedLinear/RMSNorm为普通模块，保持原compute dtype，非wrapper参数/buffer只迁设备不改dtype。再次同输入前向，**每个block及两个endpoint逐元素完全相同**。没有AutoTorchModule、meta/CPU参数或buffer，resident forward禁止任何load_from_disk调用。失败则停止迁移，不量化。
2. `--phase full` 重做上述门槛后，读取完整export manifest及200文件，验证source/state/每层SHA和200roundtrip。原位换为200个NativeH3Linear；不同时驻留三份37.46GiB模型。非目标tensor、8个refiner linear的SHA保持不变。
3. 旧QDQ reference明确为**旧公式的完整重放**：原common.nvfp4_qdq(x_s)、从已验证legacy packet独立decode的W、BF16 F.linear及corrected高精度LR，采用同一native模块的legacy_qdq模式，避免另驻留一整套BF16 weight。不是已量化激活喂给LR的旧bug。
4. 在完整旧公式前向捕获block0四种真实linear的原始x和完整平滑后/pre-QDQ x_s；供以后独立fastpacker验证。预指定前512行做四个代表检查：独立普通nn.Linear+common hooks vs module legacy完整输出须exact；同一packet的native主支 vs BF16独立decode main、排除bias/LR，**每层NMSE≤1e-4**且有限。至少一个实际SM120 E2M1 kernel profiler证据。
5. 所有前置通过后仅一次完整native前向，**200实际NVFP4 GEMM**，全部50block及video/audio有限；保存native/BF16、native/旧公式QDQ完整endpoint和逐block误差，video/audio分别报告。不要求native与旧QDQ bitwise一致，不用单层小误差替代完整传播证据。
6. 原offload、resident、legacy公式、native各阶段真实SDPA调用数量一致，QKV均BF16。主块统计另外保留video/audio/text位置，pad_or_unassigned不进入有效token主汇总。

## 预算与产物

GPU1；每次named tmux+日志+外层timeout3600秒。resident BF16约37.46GiB权重，单份顺序处理；超过60GiB停止保存partial，不切帧/token。resident-only成功可立即交付迁移证据；full须等待export完整，不能边读未完成export边宣称验证。

参考block tensor、endpoint、真实xs与export位于 `/data1/models/svdquant-wjq/research/20261002/E009/`，小JSON与日志在repo。完整manifest、源代码、模型/输入/state哈希固定；失败保留，不覆盖旧报告或已冻结源码。

只有完整native参考建立后，H3专用fastpacker真实四层与ties/zero/subnormal/tail逐byte一致，才另定resident三臂性能协议。本轮无性能、质量或heldout结论；不新增prompt/step网格。

Runner：`scripts/research/bench_h3_native_nvfp4.py`；systems独占 `h3_native_nvfp4.py` 与exporter。不修改common/E003/E005/Wan已验证文件。

## 执行前置修复记录（不改研究门槛）

第一次resident-only已完成offload参考后因 `rope.inv_freq` 留在meta设备而停止；报告 `E009_h3_resident.json`、日志及源码/计划快照保留在 `E009/p1s00_resident/`。DiffSynth只物化root buffer，未包裹的rope子参数FP32[16]未载入；它原forward重建频率，故offload参考可执行。v2从同一原始safetensors显式读取这一参数并核对shape/dtype，未知meta仍拒绝；不用空初始化绕过。完整run先重做相同50block和双endpoint精确迁移门槛。

Native E005 nibble8保留负值round-to-zero的 `-0`，common signed15 QDQ为 `+0`。四代表activation独立decode要求torch.equal数值exact，另外只允许两者同为零的符号bit差并报告数量；所有非零必须逐byte相同。完整linear与模型输出仍要求原定SHA门槛。此为已知编码约定的校验修正，不更换量化公式或放宽非零数值门槛。
