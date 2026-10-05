# KV token permutation 的有限 prior-work 碰撞

2026-10-02。只读 primary sources，未实现或运行实验。判断对象是 noncausal attention 中逐 head 共同置换 K/V 的低比特分组效应，区别于 channel rotation、固定 MMA layout swizzle，以及改变注意力 mask 的近似算法。

**决策：PARK 普通 quantization-aware KV token 排序/聚类方向。VC-Attention 已直接覆盖其核心。** “数值分组与稀疏/并行局部性不可同时满足”目前只是未验证推测；本轮不建立新 claim，也不据此启动 GPU 实验。

## 四个最近邻

| Primary source / 核实范围 | 已有内容 | 对候选的覆盖与边界 |
|---|---|---|
| [VC-Attention，2026-09-14，§3.2、§3.4、Appendix A](https://arxiv.org/html/2609.15810v1) | 每 batch/head 对 V 做在线 k-means，按 cluster label 排序并共同置换 K/V，Q 不动；利用 noncausal 不变性。每 128-row block 减均值，残差接 E4M3 或 NVFP4 quantizer，online softmax 中还原均值。比较了 sequence、static cube、k-means、balanced k-means，并计入重排/复用成本 | **正面覆盖。** 已超越按 V norm 排序。128-row mean block 不是 NVFP4 16-token microblock；粒度不同不解除重叠。未从所读实验核实其与 sparse-SP 联合执行的质量/通信 Pareto frontier |
| [SageAttention3，2025-05，§3.1–3.3](https://arxiv.org/html/2505.11594v1) | NVFP4 QK/PV，P 的两级 scaling；固定 K/P 排列用于匹配 MMA accumulator/operand register layout，复用 16 元素 max reduction | 其 K permutation 是硬件布局处理，不能误称 data-dependent V regrouping。证明 P 的分组/online scaling 需按真实 kernel 理解，不能仅量化一张离线归一化 P 就代表原生路径 |
| [ScaleSearch，2026-05-12，§3–4 的 ScaleSearchAttention](https://arxiv.org/html/2605.12464v1) | 搜索 BFP scales；attention 结合 Q/K channel-space Hadamard/可逆变换与 sink-aware mixed-precision KV blocks | 所读方法未发现按 V 内容共同重排 token 的算法。它是必需的 scale-selection 对照，不能把改变 scales 的收益误归 token 排列，也不能将 channel 变换当同一操作 |
| [SparSP，2026-09-26，§4–5](https://arxiv.org/html/2609.32197v1) | 利用时空依赖将 KV blocks 放置到各 GPU，按实际需求路由；明确保持 native KV block 的成员，只改变 block ownership/order | 已覆盖为稀疏通信而重排的系统动机。它也暴露一个反对“必然冲突”的简单事实：宏观 block ownership 与块内 microgroup 成员存在不同层级自由度。论文未验证与 VC-Attention 联合量化优化 |

VC-Attention 正文声称可以与稀疏等方向组合，但该声称本身不是联合系统兼容性实验。其已做的内容足以否定普通“利用 attention permutation invariance 改善 V 量化”新颖性；尚未核实的组合部分也不能被直接认领为我们的贡献。

## 必须保持的事实边界

- 对完整非因果注意力，令 Π 为 token permutation matrix，K'=ΠK、V'=ΠV，则 P'=PΠᵀ，因而 P'V'=PV。此为精确算术恒等式；应在 RoPE 等位置变换完成后置换，或同步移动位置元数据。有 causal/custom mask 时也须同步置换 mask，H3 不得跨 cu_seqlens 的独立段混排。
- Q/K 的 channel microgroups 与 V/P 的 token-axis microgroups不同，因此 token 分组可能改变 PV 量化误差。这只是机制入口，VC-Attention 已使用其中 V 一侧。不能因为未核实其显式联合优化 P/V 的目标，就将“连同 P 一起考虑”视为足够贡献。
- 任意排序也会改变 online-softmax 的遍历顺序、分块和有限精度 reduction。native 输出变化不自动等于 V microgroup 变好了；BF16 matched-permutation 对照、scale recipe 和 P 的真实 online 语义都必须固定/分开。
- 数值有利的 global clustering **可能**打散时空/稀疏块并增加跨 rank gather；但“必然、显著、无法局部化”尚无证据。对 block-sparse mask，若只在每个完整 KV 大块内部重排、保持块成员与 ownership，原有允许访问的 token 集和需要传输的块可以保持不变；内部 16-token 分组仍能变化。这是一条可能解除冲突的普通层级方案，不是新方法。

## 如未来仍要核实冲突：唯一廉价证伪门槛

**现在不执行。** 只有 native profile 指向该接口、且已有可重用 V snapshot 时，做一次离线分组诊断即可，不跑模型、不写新 attention kernel：

取预先固定的少量 heads/早晚步，在相同 NVFP4 scale recipe、同一 block-mean 规则下比较原序、V-norm 排序、全局 V clustering，以及**仅在原 sparse/通信大块内部做同样 clustering**。把重建 V 逆置换回逻辑顺序，计算误差与额外 preprocessing 时间；宏观 blocks 的成员、mask 与 rank ownership 固定。先检查本来就有没有比 norm 排序更好的收益，再检查局部分组能保留多少全局改善。

若局部分组保留全局改善的 ≥90%（本轮投入门槛，非统计定律），就直接推翻“数值收益必须牺牲空间/通信局部性”的强假设；若全局排序本身没有稳定改善，也停止。这个负向门槛只检验 **V regrouping 的必要冲突**：没有保留改善并不证明冲突成立，也不证明 P/V 输出质量或真实通信性能受损，仍不能升级为论文方向。不得只凭 token 移动距离或 V reconstruction NMSE 发起大规模生成。

当前已有正面 prior 足以 PARK；上述门槛仅作为未来证据触发时的最低成本排除项，不加入当前实验队列。
