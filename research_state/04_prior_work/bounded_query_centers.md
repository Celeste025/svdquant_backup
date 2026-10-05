受限 query 中心：近邻与边界（2026-10-03）

**当前判断：值得 E028 用已有三层数据做 CPU 谱诊断；不登记新方法。** 目标不是固定原均值 μ 后近似 `μKcᵀ`，而是选择受限中心 `c_i=A_iB`，同时量化 `Q_i−c_i` 并恢复 `c_iKcᵀ`。理想算术下，两者仍精确重构原 score（忽略 K 全局平移产生的 softmax 行常数）。全局共享中心是单中心端点，自由块均值是无此共享表示约束的端点；这不意味着当前全局均值本身是跨样本固定的 basis。

成本动机来自 E027：完整 QKV→output 为 31.780 ms / 3.518 GiB，已有 Q-chunk4096 为 33.975 ms / 2.601 GiB 且输出完全一致；独立 BF16 M16 stripe 为 14.815 ms，不能与 attention 简单相加，也不是融合时延下界。受限中心试图只预计算 `T=BKcᵀ [R,N]`，由每 Q 块的 A 组合修正，避免完整 `[Ng,N]` correction 和逐 Q 块高精度 K 重读。

| 最近构造 | 已有内容 | 对本问题的边界 |
|---|---|---|
| [SageAttention3 Algorithm 1，第 5、8 行](https://arxiv.org/html/2505.11594v1#S3) | 每块求 `μ_i=mean(Q_i)`；主乘法使用量化 `Q_i−μ_i`，再加 `GEMV(μ_i,Kc_jᵀ)`。本地 FlashInfer 将修正预先物化，consumer 广播消费。 | 同步中心化与恢复已是已有构造；所核材料未约束所有 μ_i 落在小型共享 basis/中心字典内。 |
| [MpFA Algorithm 1，第 3–7、12 行](https://arxiv.org/html/2609.33135v1#S4) | 使用全局 Q/K 均值，在 attention 外生成 BF16 correction 向量；在主循环中用 rank-one BF16 MMA 广播，再累加 FP4 QK。 | 直接覆盖单共享中心及 MMA 消费低秩修正。其 rank-one 指广播结构，不是对块均值矩阵做 rank-R 表示选择；未直接覆盖这里 R>1 的中心族。 |
| [HeadQ v2 §4，Eq16、18、21–22](https://arxiv.org/html/2605.03562v2#S4) | 从校准 post-RoPE queries 得到固定 per-head basis U；保存 key 残差侧码 `Z=Qq((K−Khat)U)`，读时以 `(qU)Zᵀ` 加到 logits。 | “校准 basis＋低秩侧码＋logit 修正”的一般接口已有强近邻，不能作为新贡献。该文 §8 明确 Q 量化未研究，也未验证同时选择 Q 中心并恢复中心项；未研究 packed kernel/时延。 |

**HeadQ 状态须保留：** [官方 arXiv 页面](https://arxiv.org/abs/2605.03562)记载 v3 于 2026-05-19 被作者撤回，理由为发布后发现 ethical concerns。这里只记录可复核的公开构造，不援引其性能或实验证据，也不据此将本问题判为已完成。整体覆盖状态为 **partially verified / conceptual overlap high**：所核新消费接口尚未被直接覆盖，**不等于新颖性证明**；矩阵结合律、低秩分解或中心聚类本身均不是贡献。

E028 的数值解释必须使用正确的中心合同。固定 K packets，令 `E_K=Khat−Kc`，定义 `εQ(c)=DQ(Q4(Q−c))−(Q−c)`，将实际中心化舍入纳入该误差。省略共同 attention scale 时：

```text
S_c = QKcᵀ + (Q−c)E_Kᵀ + εQ(c)Khatᵀ
S_c − S_μ = (μ−c)E_Kᵀ + [εQ(c)−εQ(μ)]Khatᵀ
```

因此 `μKcᵀ` 或其 K-score 谱不足以预测同步改中心的收益：它会把本应代数抵消的中心项当作残余误差。E028 只做已存 p36/s14 的 blocks 0/24/48 CPU 谱，至少区分每 head 的 `μ_block−μ_global` 表示谱与 **K4-error geometry**；后者对应上式第一项，仍不包含重新量化引起的第二项。离线 SVD 仅是同样本最佳低秩逼近的诊断，不是拟议在线 SVD，也不是量化精度最优性证明。

三个层、同一步和长度的数据不能回答固定校准 basis 跨 prompt/步数/长度是否稳定。post-RoPE 数据包含真实位置旋转，但目前不知道有效 R 是否随位置覆盖或轨迹变化而上升；秩始终不超过 D，不代表所需 R 很小。当前 Ng=177、D=128，取 R=D 的精确代数分解仅将 177 行降至 128 行；真正机会依赖小 R。

消费成本同样未解决：保存 T 的容量降为 `R×N`，但每 Q 块从读取一条修正变为读取并组合 R 条。缓存复用、系数精度、FP32 组合或 BF16 重舍入、寄存器与同步，决定其是否胜过已经可用的 Q-chunking。谱集中仅支持后续评估，不能推出 native 加速或视频质量；谱不集中也不证明所有非 PCA/量化感知中心无效。

本轮到此停止搜索，不实现 kernel、不运行 GPU。**E019 不补测：已有修复且无新的部署约束；原 KV 分片实验保持“未执行，非科学阴性”。**

## E031 前有限价值复核（2026-10-03）

本轮仅补两项定向检索并复读上述三篇一手论文；不新增方法、kernel 或 GPU 实验。E031 的 fullSVD/range0/range1 是经典自适应中心基线，结果在本记录撰写时尚未提供。

**覆盖状态：概念高度重叠，具体系统接口仅部分核实。** [Sage3 Algorithm 1](https://arxiv.org/html/2505.11594v1#S3) 已在 tile 循环写出块中心 GEMV；故避免完整二次 correction 物化本身也是已有 tiling 选择，不能把本地 FlashInfer 的预物化实现局限当作整个领域的必要限制。[MpFA §4.2–5.1，Eq7–11](https://arxiv.org/html/2609.33135v1#S4) 明确是全局共享 Q 中心、外部 GEMV 生成单 correction 向量，再由主循环的 BF16 rank-one MMA 消费。它直接覆盖中心恢复与硬件消费协同，但不直接提供所有 Q 块各有 A_i 的 R>1 受限中心族。[HeadQ v2 §4、§8](https://arxiv.org/html/2605.03562v2#S4) 提供固定校准 query basis、K 残差 sidecode 与读时低秩 logit 修正；其文本明确不研究 Q 量化、adaptive basis refresh 或 packed-kernel latency。保留该文已撤回状态；不依赖其性能/实验证据。本轮未核到 MpFA 的可审查独立实现仓库，关于它的边界以论文明确构造为限。

所核范围内仍未见一个等价的、**随当前输入选择 R>1 中心族，同步量化 Q−c，并以 A/T 因子在线消费而不展开 Ng×N correction** 的实际实现；这不是“没人做过”的证明。仅将 PCA/range finder、矩阵结合律和 producer FMA 组合起来，算法贡献预期很薄。可保留的是尚未验证的系统结果：受消费成本限制的中心表示，是否在真实精度—时延—显存前沿上形成有用的新点。不能把需要修改 barrier 本身包装成科学贡献。

**限投入决策：** 完成 E031；若廉价经典适应器仍保留 block 相对 global 的优势且有实际成本余量，至多推进一种已定义 producer consumer 原型并测完整路径。独立算子时间相加不是融合时延下界，E031 没有 factor consumer 也不能证明部署收益。若完整路径被已测 global 和精确 Q-chunk4096 的精度—时延—显存取舍支配，则停止这条实现路线，不再靠增加 basis/几何变体续命。单个经典 SVD 实现慢，只否定那个实现，不是所有适应器的不可能性结论。

改变“太薄”判断需要实际证据，而不是更换术语：包含中心统计/适应、Q pack、T 构造和消费的完整计时与峰值内存；对照同中心的 expanded correction，隔离 consumer 的贡献；随后用少量预先固定的其他输入/步数确认适应性收益和失效边界。若最终只剩已知低秩近似加一个容量数字，应作为工程结果收尾。若成熟基线加真实 consumer 稳定改善部署前沿，且能解释何种形状/资源约束下组合才有收益，即便算法仍是成熟组件，也值得进一步形成系统证据。无需在这一步要求视频质量已提升或任意百分比门槛，但算子 NMSE 不等于最终视频质量。

## One-hot 共享中心：E032 前基线边界

E031 已完成并经独立核验：fullSVD 约 211 ms，range0/range1 约 1.84/3.61 ms；raw preprocessing 为 10.35/12.15 ms，对照 block 为 8.43 ms。range1 保留 global→block 误差优势的 90.66%–94.55%，但 factor consumer 成本仍未知。以上是已报告的本地实验事实，不证明部署收益。E032 已冻结 K16、6 次 Lloyd 的共享中心实验；本记录不添加实验臂、不写 kernel，且没有 E032 结果。

经典替代表示是每个 Qtile 只选一个 codebook 成员：`c_i=C[id_i]`，同时量化 `Q_i-c_i`，预计算 `T=C Kc^T`，在主循环按 `id_i` 读取一行恢复项。聚类/VQ 和这个代数恒等式均不登记新 claim。它比连续 A 的表示能力更受限，但可避免每 tile 读取、组合多条 basis 行，是选择 PCA consumer 前应考虑的强基线。

本轮限定检索未找到 FP4 attention 对“原 Qtile 均值聚类＋同步 Q 残差量化＋correction 行共享”的完整等价实现；这不是不存在先前工作的证明。最强一般构造碰撞是 [Fast Transformers with Clustered Attention §3.2–3.3、Eq3–6/10](https://arxiv.org/pdf/2007.04825)：已有 one-hot query 分簇、centroid×K 和共享 attention 结果，改进版对 top-k keys 单独重算。它用簇代表近似 attention 分布/输出；本合同则保留每个原 query 对全部 keys 的低位残差乘法，再加入共享中心项，不将整条 score/attention 行替换为 centroid 的行。两者不能视作完全同一算法，也不能把 centroid 表复用本身当新意。

[VC-Attention §3.2、Appendix B](https://arxiv.org/html/2609.15810v1#S3.SS2) 是更近的低位构造：在线聚类 V，按标签排序 KV，然后按固定硬件块减去 V 均值并在线恢复；Q 保持原序。它没有直接给出原 Qtile Qmean 的 codebook 与 correction 行共享。已核 Sage3/MpFA 的块均值/全局均值仍是此基线的两个参照端点。

接口判断见 [consumer notes](../06_experiments/restricted_center_consumer_notes.md)：保留单行 TMA/consumer，只需增加 codebook 行数和 center_id 映射。精度、完整聚类/preprocessing 成本及实际 cache 行为仍待测，不从容量或读取行数直接推出速度。
