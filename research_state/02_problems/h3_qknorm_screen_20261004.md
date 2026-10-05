# H3 QK-norm 功能方向筛选

2026-10-04；claim-prior-work-triangulation；3 个定向 primary queries，随后读取原文。0 GPU、0 新前向、无代码修改。已读 current_state 与 F2 草案；F1 已由 E062 停止，本页不重开。**结论：支持一次有界的实际反例检查，方法仍 PARK。** 宽泛的“线性重构不等于 attention 功能、改用联合 QK loss/低秩”高度重叠；精确到 **pre-QK-RMSNorm 的真实 W4A4＋共享 rank32**，本次未核实直接覆盖，但不据有限检索宣称新颖。

## 最接近的三个方法

| 主来源 | 已直接覆盖 | 未直接覆盖本问题的部分 |
|---|---|---|
| [QuantMLA §3.1–3.2，Appendix C](https://arxiv.org/html/2609.36760v1) | 以功能误差校准量化：content 用 attention-output 重构，RoPE 用完整 positional QK 重构；采用实际 query 保留跨频率抵消。给出 RoPE 兼容的二维旋转/互逆缩放，以及 RMSNorm 正交等变的离线融合。 | 量化 MLA 缓存，RoPE query 保持非量化；RMSNorm用于等价变换，不是测量 pre-norm 投影径向误差；没有固定共享 rank 的 W4A4 Q/K 双侧补偿。本地若只照搬 QK loss＋RoPE约束，应视为高度增量。 |
| [HeadQ §3，Eq3–13](https://arxiv.org/html/2605.03562v1) | 将 key误差分成 attention可见与softmax零空间，使用 query基学习低秩key残余，并在logit侧校正；明确 storage-MSE可与功能排序相反。 | 主要是key-cache量化、固定query；零空间来自query投影/softmax常数，不是QK RMSNorm的输入依赖径向方向。没有一份packed W4＋原rank32因子的部署合同。它直接堵住“发现MSE排序错位”作为完整贡献。 |
| [SlimDiff §2.4 Type-II，Algorithm2](https://arxiv.org/html/2509.21498v1) | 在扩散模型中联合分解Q/K，按timestep相关输入协方差加权，对双线性复合`Wq Wkᵀ`做低秩压缩，而非独立保护Wq/Wk。 | 所核公式是未含QK RMSNorm/RoPE的双线性算子；属于结构压缩，不是量化残余校正。说明“联合QK、数据感知、低秩”已存在，不能仅用H3/NVFP4换设置认领。本文未复证其最优性推导，不把作者定理当本地数值保证。 |

角度量化也非空白：[NSNQuant §3.3–3.4](https://arxiv.org/html/2505.18231v2)以cosine distance训练KV向量码本，并作尺度恢复；这保护方向且保留幅度作用，不等于模型自身RMSNorm消除径向误差。一般 block/QAT/QAD 通过完整计算图已隐式包含norm及QK耦合，不能描述成“现有方法都忽略它”。本次未找到明确以“QK RMSNorm径向无效导致W4A4低秩容量浪费”为目标的直接方法；after-QK-norm/PTQ全领域查全程度仍为 **partially verified**。A3检索命中联合QK/RoPE低秩原稿，但OpenReview全文被验证页阻挡，不据摘录扩大结论。

## 保留下来的问题究竟是什么

**可证伪问题：** 在固定执行合同与输入下，legacy rank32 相比同smooth的fresh rank0，是否反复出现原始Q/K误差明显改善、实际norm→RoPE→attention功能误差却未相应改善，而且这项错配与可干预的方向分配有关？有用之处是决定有限高精度预算该保护什么；数学上的径向不变性只是已知前提，不是发现。

这里有一个容易过度解释的细节。令teacher head为q，native为`q̂=(1+α)q+e⊥`。在`1+α>0`且忽略ε时，其归一化方向由`q+e⊥/(1+α)`决定。**纯径向且不翻号时近乎无效，不意味着混合残余中的径向分量可以无条件删掉。** 径向改变会重标切向偏差；负缩放还会翻方向。实际learned Γ、ε、RoPE和softmax都应直接重放。因此径向能量大、原始cosine改善、或oracle删除径向，单独都不能证明容量浪费。

## 对根计划的判读建议

- **对照合理：** 两个既定teacher输入×block0/24/49；完整同输入native rank32对比同smooth、从原Ws重新量化的`Q(Ws)` rank0。删除旧`Rq`的LR会改变目标有效W，不能充当rank0。rank32与rank0之差同时包含主支重新量化，故只称配方差，不能全归为“LR修正向量”。结论先限于legacy配方，不外推完整官方SVDQuant校准。
- **指标形成同一条证据链：** 各臂相对同teacher的raw Q/K SSE；径/切向SSE的**带符号净改善**；实际norm后的Q/K误差；post-RoPE logits按每query去有效keys常数后的误差；固定teacher V及原attention算术的输出误差。不能让V改善/损伤混入QK机制，也不以未归一化score能量或latent元素数当独立证据。
- **这次能决定：** 若raw收益与功能收益一致，或主要没有raw QK收益，停止当前“径向预算错配”解释；不换层/提示寻阳性。若同一预选层在两状态都出现错配，且数值控制可靠，允许下一次有针对性的方向干预。所采样层/状态有限，不推总体或视频质量。
- **下一步才区分非平凡残余：** 在同预算下，比较普通Q/K/V权重调整、输入协方差加权与完整norm→RoPE的联合拟合。若普通方案就恢复功能，记录成熟校准收益即可。潜在剩余是：输入依赖的切空间及双侧误差抵消，使固定独立度量不能合理分配共享rank；需要一个可部署的联合表示/求解器，并超过直接attention/block重构基线。不是只把损失写在norm后。

**没有要求实验前已有决定性视频阳性。** 上述局部实验能低成本排除或保留真实容量错配，信息价值成立；阳性只形成机制入口。论文级结果仍需证明可干预性、成熟同预算强基线之外的收益及独立视频质量，不能以近邻使用不同模型就自动认领残余贡献。

执行前补充：本地[既有前沿核查](../01_literature/native_systems_frontier.md)已记录 GaugeQuant 的 RoPE 可交换逐对旋转及 QuaRot/SpinQuant/ReQAT 等价变换；普通共同 Q/K 旋转、借旋转改变舍入来求互补不单独构成新方法。E063 先做必要条件筛查，不执行 attention 四角；只有原始径向收益错配通过，才考虑完整功能诊断。实际模型为 ComfyPruned，QKV 使用 `[rows,3,heads,dim]`，须与真实 norm 输入核验，不能引用基础类的不同布局。

执行后结论：E063及独立复核已complete。两状态×三预选层的raw收益仅0.454%–1.383%，主要来自切向，norm后仍同量级改善；必要条件三层均不通过。依原计划停止径向容量错配候选，不执行上文下一阶段attention/同预算拟合。见[报告057](../reports/057_20261004_h3_qknorm_radial.md)。上文GO仅记录执行前诊断许可，不是当前方法立项状态。
