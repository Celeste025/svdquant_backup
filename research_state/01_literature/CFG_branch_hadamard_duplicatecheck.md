# CFG 分支 common/difference 量化：有限查重

2026-10-03。只读；1 个检索 query（`"GCBT" "Joint Branch-Space" quantization`），沿本地 P15 核查原文后发现直接覆盖，停止搜索。无实现、无 GPU、不改变 E046。

**结论：核心变换及线性执行顺序已有直接覆盖；不应作为新方法立项。** [GCBT，arXiv:2610.00930v1，2026-10-01](https://arxiv.org/html/2610.00930v1) §3.2 Definition3.1/Eq3 对 matched cond/uncond activations 定义 `Rᵀ Q_A(RX)`，明确沿分支轴而非 feature 轴。附录 A.3 Eq27–28 进一步允许先在变换坐标上用共享 W 做线性，再逆变换，并在逆变换后加共享 bias。§3.3 根据 guidance/二阶矩选择角度；固定 common/difference 是该一般构造的固定 45° 基。附录 B.4 还包含 Wan1.3B、81帧、50 FlowUniPC、CFG5/shift8。这不是仅有 feature-Hadamard 或 CFG-loss 的相似术语。

**精确覆盖的边界。** 提案采用 `x±=(xc±xu)/2`，正交基常用 `/√2`；二者相差整体幅度约定。实数线性代数下可在逆变换/scale 中补偿，但 NVFP4 global、E4M3 舍入、BF16 加减使它们不必逐位相同。原文所读段落未给出“固定 Hadamard + 每坐标独立 NVFP4 global + 本地 SVD 原生 kernel”的同一实现证据；不能谎称已跑过相同二进制合同。这个差别目前只是实现/setting，未产生独立贡献。此前核查的 [DSAQuant](E045_terminal_cfg_nearest_work.md) 覆盖 mismatch 与 CFG-drop，**不是**本次判定直接重复的依据。

**本地实现仍有实际障碍，不能用‘仍两主 GEMM’代替成本与正确性证明：**

- **同步执行。** 现有 cond/uncond 是两次完整 DiT。逐 Linear 配对须改为按层协同推进，并保持同样 timestep、样本、token位置、mask 与各支 cache；需额外驻留另一支激活。每个共享线性之后先恢复两支，再进入原 norm/RoPE/attention/激活函数。换基不能直接穿过非线性或 attention。
- **统计域。** `x+`/`x−` 各 pack/global 才是提案；将它们拼成一个 batch 交给当前 tensor-global quantizer 会改变合同。独立 scale 也不保证差分用满有效码本：group outlier、E4M3 下溢和 BF16 相减仍需面对，不能由量级小直接推定精度更好。
- **bias、权重、LR。** 共享量化 W 可复用，权重误差不会因换基消失。不能直接对两个坐标调用带相同 bias 的完整 Linear：应在逆变换后给 cond/uncond 各加一次 bias。SVD 的 smoothing/高精度低秩分支仍必须存在；可在原始平滑输入上分别计算 LR 后相加，或在线性精度合同明确时连同 LR 换基/还原。**不能让 LR 改吃量化后或 packet-decode 输入**，也不能漏计两支 down/up。
- **开销与误差目标。** 变换/重构需要加减、读写、可能的新中间 buffer；两主 GEMM 不代表延迟/峰值不变。以半和半差写最终 CFG：`f_cfg=f+ +(2w−1)f−`；w=6 时 contrast 系数仍为 11。该变换重新分配误差，不会代数上消除 guidance 放大，common drift 和 contrast error 都须保留。

可作为成熟分支编码基线的一个固定参数实现候选；本次不建议借“无需搜角度”或“保持 CFG6”重新命名论文方法，也没有核实其在本地 NVFP4 上一定有效。

root补核：GCBT §4.3 的 runtime 段明确使用 Nunchaku 测 SVDQuant，报告约1–2%额外时延和近似不变的峰值显存。因此不能再假设它只有fake-quant精度验证，或把“接入native引擎”本身当作未覆盖贡献；本地NVFP4精确global/二进制合同仍需另核，但这不自动形成创新。原文：https://arxiv.org/html/2610.00930v1#S4.SS3
