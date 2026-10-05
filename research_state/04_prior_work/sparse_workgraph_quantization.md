# N2：投影量化与稀疏工作图的有限碰撞

2026-10-03；只读本地记录与四项直接一手工作，无 GPU、安装或新 claim。按 claim-prior-work-triangulation 区分已覆盖问题与未观测事实。**宽泛的量化×稀疏误差、学习 router、动态预算校准均已覆盖；本轮未核实到同一 W4A4 QKV 仅换投影来源 mask、再测实际工作量的直接实验，但这不构成新颖性证明。**

| 最近工作 / 已核范围 | 路由输入与预算合同 | 已有校正；与 N2 的边界 |
|---|---|---|
| [QuantSparse v4 §3.1–3.4、§4.4](https://arxiv.org/html/2509.23681v4)；[作者仓库](https://github.com/wlfeng0509/QuantSparse) | Eq.4 明确量化 X/W 后产生 Q/K 投影误差。报告 15%/25% density 设置，但本轮原文未明确可审计的推理 selector dtype、top-k/top-p 实现；仓库仍仅 README/图片并注明代码待发布，不能补猜。 | FP teacher pooled attention 蒸馏及 salient top-k **query 选择**，另有跨步二阶残差校正。该 teacher 选择是训练监督，不等于推理时固定低位 QKV 交换 teacher mask。已直接覆盖投影量化与稀疏共同失真，不能说前人只量化 attention 操作数。 |
| [SLA2 §4–7、Eq.15–16 / Algorithm 1](https://arxiv.org/html/2602.12675v1) | 原 Q/K 先池化、可学习投影，再逐行固定 top-k%；§5 描述原输入 FP16，稀疏分支随后采用 INT8/FP8 风格 attention 量化。论文没有把 router 描述成从低位 packet 解码；[作者仓库示例](https://github.com/thu-ml/SLA)也接受 BF16 QKV，不能把论文 FP16 记法当唯一部署 dtype。 | 以 full-attention 输出训练 router/混合参数，SoftTop-k 后硬路由整模微调及 attention QAT；并非 teacher-mask 运行期替换。固定 k 下边身份会变，基本块数不随分数改变；无法据此推出 top-p 预算膨胀。 |
| [Sol-Attn §2–3、Eq.4–5](https://arxiv.org/html/2607.24027v1) | 当前 pooled Q/K 的 proxy logit，以每行均值/标准差设 `τ_i=μ_i+βσ_i`，动态选块；β 对应模型平均 density 校准。所读论文未指定 router 的存储/累加 dtype，本轮未审作者 kernel，标为未确认，不能称量化路由。 | 已覆盖 top-p 密度难控制、动态预算、路由开销和遗漏块近似纠正；不是量化特定预算校正。未见固定 W4A4 projection 后交换 teacher mask 的实验。若将来只需普通标准化阈值或调 β，则已有直接方案。 |
| [SpargeAttn 固定作者源码 `ae5b629e` utils.py L162–214、270–279、371–410](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L371) | 路由来自当前**浮点** Q/K，不是 INT8 decode：pool FP32 reduction 存回输入 dtype；guard 的单位向量转 FP16 做余弦统计；同时另产 INT8 packet。CDF 累计选块或固定 top-k 二选一，再把低相似度 Q/K 块对应整行/列强制保留。 | 因 guard，名义固定 top-k 也可有动态实际块数。现成阈值、CDF/top-k 和高精度 pooled selector 已是普通对照；未见该路径保护上游投影误差的 teacher-mask 纠正。旧尾块/全长消费者限制仍在，源码合同不等于本机已可运行。 |

**因果边界。** 同一真实 block 输入 X，经 BF16 或 native W4A4 projection，再经原 norm/RoPE，取得 Q₀/K₀ 与 Qq/Kq。保持 router 算术、分组、阈值和有效长度相同，令 M₀=R(Q₀,K₀)、Mq=R(Qq,Kq)，再比较同一消费者 `F(Qq,Kq,Vq;M₀)` 与 `F(Qq,Kq,Vq;Mq)`，才能隔离 mask 的作用。“低位来源 mask”在这里仍可由 BF16 张量生成，不能混写成 router 自身用 FP4。提高 R 的算术精度不能恢复投影已丢失的信息。M₀ 只是干预，不是针对 Qq 的最优 mask，也不是免费部署方案；少选块不自动等于省掉无用工作。

**尚未观测的残余。** 是否在真实配对输入上，量化通过 CDF/guard 改变实际 active blocks、head 负载或 KV union，并在固定低位 QKV 消费时改变误差—完整成本取舍？应先区分块数量效应与固定预算下的边身份/局部性，不以 mask Jaccard 或极端 head 当系统结果。已有 [N2](../01_literature/native_systems_frontier.md) 所列 FVAttn/SparSP 已按实际 mask 做调度/需求传输，本轮不重复查新。普通预算校准或已有调度能解释现象时，不能把现象重命名为新方法；也不恢复旧任意 10% 门槛。

**当前决定：未测、未 ready，并非阴性。** [本轮入口审查](../02_problems/sparse_workgraph_entry_audit.md)发现 FastVideo 通用 Triton sparse consumer 候选，但现成 H3/Wan VSA 是 fixed-top-k；真实 top-p 的 BSA/NABLA 尚不满足本项目布局/尾部完整合同。固定预算重放若后续获准，只回答该预算下边身份的影响；不能冒充原 N2 的动态预算实验。不为本 note 新写 router/kernel、不开展方法网格。

E035准备时补充：后续本地只读核查确认已安装cuDNN BSA有SM120、外部variable-count索引和每KV块有效前缀支持，CPU导入成功但未GPU验证。原H3/BSA完整路由仍不是现成top-p合同；E035明确只做contiguous128/prefixdense的累计概率诊断及同输入投影捕获，暂不消费稀疏mask。该新入口不改变上述prior-work覆盖判断。
