# 分布保持的量化校准：continuity / Stein 有限碰撞

2026-10-02；应用 `research-target-reframing`，仅 primary 文献与数学核查，无 GPU、无新 claim。**判断：KILL 朴素 strong-form / STE 散度目标作为新方法；PARK 更宽泛的分布校准方向。** 局部或固定 seed 的端点 NMSE 不等于分布质量，但这个事实、gauge 自由度和候选残差目标均有直接先例。当前没有找到足以支持一次新视频实验的剩余贡献。

## 最危险的直接覆盖

| Primary source / 已读范围 | 覆盖与边界 |
|---|---|
| [Horvat–Pfister，ICLR 2024](https://arxiv.org/html/2402.03845v1)，§3 Eq11–13、§4 | 明确给出 `div r + r·∇log p=0` 的 gauge 条件、加权正交分解，并指出 score L2 同时惩罚不影响连续分布的分量。**不仅“相关”，已覆盖候选核心直觉及公式。** 不是原生量化器或有限步保持定理。 |
| [Improving Flow Matching by Aligning Flow Divergence，ICML 2025](https://openreview.net/pdf?id=FeZimuj6SG)；[后发 arXiv 正文](https://arxiv.org/html/2602.00869v1)，Prop3.1 Eq8、§4 Eq14、Appendix D Eq35 | 概率误差 PDE 的 forcing 正是 `−p[div(u−v)+(u−v)·score]`；提出条件化 continuity residual 的绝对值/平方目标及 Hutchinson、stop-gradient 实现。实践再加 CFM loss。**把此目标接到量化校准，单独只有应用差异。** 条件目标是上界，不等于精确 marginal gauge seminorm。 |
| [DNS，ICML 2026](https://proceedings.mlr.press/v306/yang26aa.html)；[Q-Drift，2026-03](https://arxiv.org/html/2603.18095v1)，摘要及§3/附录 | DNS 明确以边缘分布而非轨迹保持为目标，用 timestep shift，覆盖确定/随机采样及 Gaussian conditional flow；本次 DNS 只核实会议摘要，未完整审查证明。Q-Drift 用隐式 Gaussian 扰动及方差标定推导 drift 修正；附录忽略跨步协方差是显式近似。它们不证明实际确定性 NVFP4 误差满足假设，但“量化应保分布”不是空白问题。 |
| [DMD2，2024](https://arxiv.org/html/2405.14867v2)，摘要/方法；[LongLive-2.0](https://arxiv.org/html/2605.18739v2)，§2.2、§4、Appendix H/I | DMD2 去掉绑定 noise–image pairs 的回归，以 real/fake score 差做分布匹配；LongLive 已把不变的 DMD 目标用于 NVFP4 backbone + LoRA 视频模型。**分布目标 + W4A4 本身已落地。** 对照必须包含普通 task/teacher velocity loss 和 DMD，而非只打层级 NMSE。 |

范围校正：[NVIDIA QAD](https://arxiv.org/abs/2601.20088)主要是 LLM/VLM logits KL，不能把其 KL 当作视频密度 KL；[Q-Diffusion](https://arxiv.org/abs/2302.04304)/[PTQD](https://arxiv.org/abs/2305.10657)分别提供多 timestep PTQ、量化误差补偿等更简单基线。搜索中的 [Stein Points](https://proceedings.mlr.press/v80/chen18f.html)是用点集逼近概率测度，不是神经网络 W/A 位宽量化，不制造标题碰撞。

## 正确的数学命题与实际阻断

固定条件 c，令 p_t>0 满足 ∂_t p_t=−div(p_t v_t)，δ=v_q−v。若同初始分布、边界无通量/充分衰减且解适定，则

`div(p_t δ)=0  ⇔  R_t:=div δ + s_t·δ=0`,  `s_t=∇log p_t`

使 p_t 同时满足两条连续方程。它允许不同 seed 轨迹，但不是任意大扰动、有限样本小 loss 或有限步采样器的质量保证。仅最小化 `(E_p R)^2` 完全无效：在适用 Stein identity 的边界条件下，`E_p R=0` 对任何合格 δ 都成立。`E_p[R²]` 或足够丰富的弱测试函数才有辨别力。

1. **score 不因线性路径而自动已知。** 对 `x_t=(1−t)x₀+tε`、独立 Gaussian ε，只有精确 conditional-mean velocity `v*=E[ε−x₀|x_t,c]` 才给 `s_t=−[x_t+(1−t)v*]/t`（t>0）。用近似 teacher、CFG 或短步蒸馏输出代入，不保证得到该 teacher 实际 pushforward 的 score；低噪声端还放大误差。用单样本 conditional score 替代后再平方会增加条件方差惩罚，不能保留原 gauge 零空间；不等于免费、无偏目标。
2. **原生量化器的边界通量不能用普通 JVP/STE 忽略。** 动态 NVFP4 code/scale 切换存在不连续面；几乎处处导数漏掉分布意义的表面项，STE 又是另一函数。反例：圆周均匀密度、teacher 常速 3；δ 在两半圈分别为 +1/−1。普通 `div δ=0`、score=0 几乎处处成立，但跳变通量非零，速度 4/2 不保持均匀分布。即使估计出的局部残差为零，也不能推出保持密度。
3. **连续条件不等于部署求解器。** `p=N(0,I)`、δ=Jx、Jᵀ=−J 的 exact flow 保分布；Euler 一步协方差为 `I+h²JJᵀ`，已改变分布。rCM 少步或 H3 实际 solver 必须单独面对这种误差；抑制速度 L2 也可能是在控制离散误差，不能一概称“过约束”。
4. **弱形式可合法但不免费。** `E_p[δ·∇f]=0` 对所有合格 f 等价于弱通量条件，可避免对 quantizer 求导；有限 critic/核/小 cache 只约束有限统计，不能认证高维条件视频分布，仍可能漏掉模式坍塌。所有边缘时刻的保持也比仅保持最终分布更强。新的目标没有消除 teacher、采样覆盖或 critic 复杂度。

## 两次限定 query 的 seed coupling 补查与执行决定

若固定正交 R 且 z 为标准 Gaussian，则 `F_q(Rz)` 与 `F_q(z)` 本来同分布：仅改推理初始 seed 耦合不能改善总体质量。训练时匹配 `F_q(z)` 与 `F_teacher(Rz)` 是受限 coupling regression，DMD2 本就不要求一一 seed 对齐；[InstaFlow/ReFlow](https://arxiv.org/abs/2309.06380)也已改变 noise–image coupling。本次未核实直接“正交 seed 重配量化”应用，但不足以推定 novelty；若 R 依赖 z，逐点正交更不保证 Gaussian 测度保持。此补充同样 PARK，不开第三条路线。

**最便宜且合法的检验目前是上述解析反例，而不是新 JVP toy 或视频校准。** 它们足以否定朴素保证。重新启动的前提必须是一个同时明确原生离散边界、实际 solver、score/critic 成本的具体干预，并说明为何不等于已有 FDM/DMD 或简单 teacher loss；当前没有这个干预。不把 E007/E009 的同码轨迹差异解释为质量改善证据，不启动 GPU。
