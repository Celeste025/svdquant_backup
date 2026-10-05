# E007 — 完整 rCM-Wan 原生 NVFP4 基线正确性

2026-10-02，运行前预注册。E006 已 STOP；本实验不做其 continuation。

目标是建立可审计的完整原生 W4A4 基线，不将移植工程包装为研究创新。首版软件 packing 只验证正确性，**不报告加速**。

## 唯一样本与模型

- BF16 teacher：`/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer`，不是原始 Wan teacher。已核对配置：30 blocks、12 heads×128、hidden 1536、FFN 8960、patch `[1,2,2]`、in/out channels 16。
- 旧 SVDQuant：`/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16`，使用同包的 model/scale/wgts/smooth/branch，旧 rank32，不重新校准。
- 固定输入：`/data1/models/svdquant-wjq/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches/0001-00000-0.pt`。这是原校准集 prompt0001、step0、guidance0，不能称 heldout。
- 输入 BF16 latent `[1,16,20,60,104]`，对应 77-frame、480×832 视频及 **31,200 个视频 patch tokens**；现成 BF16 text embedding `[1,512,4096]`。不加载或重跑 text encoder，不生成新样本，不解码视频。
- 环境：`/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python`，torch2.11.0+cu128、Diffusers0.33.1。三条路径都保留同一原始 torch SDPA/QK norm/RoPE，实际调用记录 dtype、次数；不用 Sage/FlashInfer attention 混入线性层对照。

## 冻结的顺序与检查

1. 加载原 rCM BF16 transformer，记录配置、参数量、源码和所有输入/权重文件 SHA256。固定原始 raw call 运行 BF16 teacher，捕获30个完整 block endpoint；与缓存参考报告误差（若 NMSE>1e-6，停止并检查缓存/模型契约，不能改样本）。再次相同 BF16 forward 必须逐元素完全一致。
2. 通过原 `load_quantized_transformer` 重建旧 QDQ+BF16 GEMM 路径。必须正好300个 block projection 和300个 activation Quantizer hook；保存全部30 block endpoint 与最终 denoiser output，报告相对 BF16 的差异。
3. 原生转换使用 checkpoint 已保存的 weight scales 恢复 codes，不对 QDQ 权重重新寻找 scale。**全部300权重**独立解码后的 BF16 重构须逐元素等于旧 QDQ 权重，任何不一致立即停止。
4. `native_bypass` 必须实际恢复旧 activation QDQ 与 BF16 F.linear+bias，并保留 smooth/LR 的原顺序，完整输出和30 block endpoint与转换前旧QDQ全部逐元素相同。不是虚设 identity hook。bypass 的临时 BF16 weight 不进入原生部署常驻内存。
5. 代表层固定为 block0 的 self-attention q、FFN fc2、cross-attention k/v。旧QDQ forward 时在 activation quantizer 入口捕获**平滑后、未QDQ**真实输入，各取按原顺序前512行（cross KV就是完整512 text tokens），不选极端token、不补采样。native main 与相同 packed codes/scales 独立解码的 BF16 F.linear 主支比较，均不含 bias/LR；**每层 NMSE≤1e-4**且有限。保留旧 rounding 的 activation packer，不能把 RNE 改动误作 GEMM 误差。至少一层 profiler 证明实际 SM120 FP4 kernel。
   另外保存 q、fc2、cross-k 的完整 pre-QDQ 输入到 `/data1/models/svdquant-wjq/research/20261002/E007/pack_inputs.pt`，供独立快速packer做真实输入逐code/scale一致性验证；不改变本轮300层reference native路径。
6. 以上通过后，对同一完整 raw call 运行全部30 block、300 native main GEMM。不得截视频 token、跳层、混入BF16主支fallback。保存完整 endpoint NMSE：native/BF16、native/旧QDQ；以及30 block逐层同位置差异、有限值、实际native调用数、SDPA调用证据。主支之外低秩与保留层为BF16。

## 解释边界与止损

- native GEMM 将块级/全局 scale 用于硬件算术，旧 QDQ 先重构并舍入 BF16 再 GEMM。因此不要求完整 native/旧 QDQ bitwise相同，不把差异直接解释为质量损失或收益；先报告同codes单层数值契约与完整传播误差。
- bypass、300权重、单层契约、非有限值、native调用路径任一不通过则停止正确性结论并保存失败记录。只修实现契约，不改样本或阈值追结果。
- 所有误差使用 FP32 张量差和 FP64 能量累计；不以单层 NMSE声称最终质量，不以这一个校准样本声称泛化。
- GPU0 named tmux，日志 `results/logs/E007_wan_native_correctness.log`，硬 timeout **3600秒**。逐阶段、逐 block保存状态；显存超过60GiB则停止保留partial，不裁token。
- 首版运行墙钟只作预算记录，包含CPU capture/软件pack/检查，不作性能比较。**快速packer经过相同契约验证后**才另行冻结完整forward latency/peak-memory协议；当前runner不自动开始性能计时，也不解码/自由rollout。

Runner：`scripts/research/bench_wan_native_nvfp4.py`；native library由systems agent独立负责 `scripts/research/wan_native_nvfp4.py`。计划与两者源码哈希进入结果。
