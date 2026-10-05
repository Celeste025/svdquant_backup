Qmean 附加 query 行：窄技术与近邻核查（2026-10-03；无 GPU、无 kernel 开发）

**判断：合法、可做一个数值反证，但当前不是“保持原合同的免费 fusion”，也未建立方法贡献。** 检查的 SageAttention/FlashInfer/MpFA 路径没有直接实现附加 query 行即时生成修正；覆盖状态为 partially verified，差异目前属于实现与精度取舍。最值得先决定的是复用 FP4 K 带来的新误差，而非继续搜索或直接写 kernel。

已有碰撞与准确边界：

- FlashInfer 的 `_preprocess_qkv` 先求每 128 行 Q 均值，再以 FP32 operands 计算 `μ @ Kc.T`，保留 `[B,H,Mpad/128,Npad]`。`delta_correction.cuh` 已在 attention 内广播该行并初始化 score accumulator。因此“把 correction 加回融合进 attention”已存在；剩余是其生成与存储。S007 主段 0.818 GiB 是该物化张量的记录，不等于整模峰值或可实现加速。[固定预处理源码](https://github.com/flashinfer-ai/flashinfer/blob/1b578de52924c973bd229fe4fd147499ead2781d/flashinfer/nvfp4_attention_sm120.py#L128)、[消费源码](https://github.com/flashinfer-ai/flashinfer/blob/1b578de52924c973bd229fe4fd147499ead2781d/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/consumer/delta_correction.cuh#L35)。
- SageAttention3 官方 API 第 79–84 行仍先计算 `delta_s = matmul(qm,k.T)`；已核 PR394 的 Q chunking 降低同时驻留的 correction，但仍逐 chunk 物化。这不是附加行即时生成。[官方 API](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell/sageattn3/api.py#L75)、[PR394](https://github.com/thu-ml/SageAttention/pull/394)。
- MpFA Algorithm 1 第 7 行在 attention 外做 BF16 GEMV，主循环加载该向量，再以 BF16 rank-one MMA 将其广播至 score；没有借用 K tile 现场生成 `μKc.T`。它直接覆盖“额外 MMA 消费低秩 correction”，没有直接覆盖本提议的生产者改变。普通矩阵堆叠与低秩修正的代数也不构成新贡献。[MpFA §4–5 / Algorithm 1](https://arxiv.org/html/2609.33135v1#S4)。

**代数推导：便宜版本会改变关键误差合同。** 令实际预处理后的均值为 μ、centered K 为 Kc；固定现有 Q/K packets，反量化后记 Q̂c、K̂c。暂忽略两条执行路径的浮点累加次序差异，当前 score 为 `Sbase = Q̂c K̂cᵀ + 1 μ Kcᵀ`。附加均值行得到 `Saug = Q̂c K̂cᵀ + 1 μ̂ K̂cᵀ`，故

`Saug − Sbase = 1 [(μ̂−μ) K̂cᵀ + μ (K̂c−Kc)ᵀ]`。

这个差随 key 改变，不能靠 softmax 的行常数不变性消掉。即使 μ 完全精确，第二项仍在；把 μ 分成两段 FP4 只降低第一项。理想中心化恒等式下，它使 K 的量化误差从由 Qc 加权重新变成由原 Q 加权，可能抵消 Q smoothing 的一项收益；“可能”的实际量级尚未测。保持现有未量化 K correction 合同，需要额外高精度 K 路径或足够准确的 K residual sidecar/重建。由 FP4 K 反量化成 BF16 并不能恢复丢失的信息；双段 K 则是另一合同，不能悄悄加入本方案。

**SM120 合法表达与成本（源码事实 + 有界推断）。** 本地 `kernel/traits.h:66–130` 的 CTA 为 M64/M128，QK atom 是 `SM120_16x32x64_TN_VS_NVFP4`；底层 CUTLASS `mma_sm120.hpp:3185–3230` 是 `m16n8k64`、E2M1×E2M1、FP32 accumulator、E4M3 scale；本路径每 16 个 K 维元素一 scale。给一条 μ，或 μ_hi/μ_lo 两条行，各自配置 scales，再取结果广播，在数学及 operand 形状上合法；可补齐 16 行或重新组织有效 Q 行。M16 不构成不可行证明。

但当前满载的 128 条 Q 没有天然空行。朴素做法多一个 16-row QK stripe，理想算术量为 QK 部分的 16/128=12.5%，不是全 attention 的 12.5%，也不是实测时延；若 M64 CTA 独立重算同一 128-Qmean，则对应为 16/64。还要付跨 warp 广播、寄存器/共享内存、同步及排程成本。附加行必须在 softmax/PV/output 前取走，不可变成真实 query 或 key。当前现成模板只接受 M64/M128，不能直接改成 M129 后调用；需要调度修改。

BF16/FP8 均值不能作为混合 dtype 的一行直接塞入当前 E2M1×E2M1 指令。存成 BF16/FP8 后再转 FP4，不会保留原精度；若保留精度，必须走另一兼容 MMA/operand 路径。BF16 μ 与原 BF16 K 的额外 MMA 可避免上述 K4 误差，但要读取/保留高精度 K（以及额外乘法和广播），这是常规在线 GEMV/GEMM fusion 的工程取舍；不能因为省去二次张量便默认更快，也不能只因它常规就断言没有部署价值。FP8 路径同理需明确 K 表示、尺度和指令；本轮不设计新 kernel。

**最小有决策价值的下一实验（仅定义，未运行）。** 固定一份现有真实 H3 层 Q/K/V、原 128-Qmean 分组与全部量化 packets；按固定次序逐 Q block 消费全部 K，避免保存完整 logits。仅比较原 correction、精确 μ×K̂c 的“均值无限精度”上限，以及一段/两段 FP4 μ×K̂c；共享 Q̂cK̂c、softmax/PV 数值实现。读出 softmax-scale 后、去行常数的 correction 差，概率分布差与 attention 输出差，并相对已有 block/global-mean 两路径定位误差大小，不设任意阈值。这样能区分 K4 误差下限与 μ 编码误差：若上限就丢掉所需的 block-mean 行为，双段 μ 不值得继续开发；若可接受，才值得一层原生 kernel 的总时间/峰值测量。前者测的是算法精度取舍，不是原合同 fusion 的正确性或速度；后者才回答额外 MMA 能否换回真实成本收益。单层结果不推整段视频质量。

本地复核根目录：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/`。关键相对文件：`nvfp4_attention_sm120.py:128–167`；`data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/traits.h:66–130`；同目录下 `compute/consumer/delta_correction.cuh:35–53`；`data/cutlass/include/cute/arch/mma_sm120.hpp:3185–3230`。本轮不以未检索到等价实现宣称新颖，不注册 active claim，不扩大 baseline。
