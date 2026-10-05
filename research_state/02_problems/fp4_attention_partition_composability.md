# FP4 attention：概率归一化与分区可组合性

2026-10-02；独立定向核查。已读本地E016所用FlashInfer SM120 consumer、官方Sage源码及下列一手论文；没有GPU实验，没有修改冻结源。本页不登记active claim。

**判断：固定Q/K/V packets后，P量化仍可能产生超出普通浮点累加顺序的分区依赖；row-sum守恒不等于分区可组合。当前尚未测量H3实际量级。** 整数log2参考max和逐tile两级缩放已有直接近邻，不能把它们本身作为新方法。已有修复不免除对当前部署合同的测量；新方法claim还需要现成方案未解决的实际成本或数值约束。

## 1. 源码事实：本地P4 consumer的合同

本地源码根目录：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/`。

- `jit/nvfp4_attention_sm120.py:45–51`指定Q/K tile为128并启用ping-pong math order；当前非因果主路径先从完整N128 score tile更新running max，再消费两个N64 slot。
- `data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/consumer/softmax.cuh:317–413`：更新running max，按其变化缩放旧row_sum；P带固定`448*6`放大，先把未量化P计入row_sum，再除以未舍入FP32组scale供E2M1转换。倒数有`1e-8`下限。
- 同目录`compute/mainloop.cuh:778–849`分别把FP32组scale转E4M3、归一化P转E2M1，然后执行原生PV；`:875–903`明确KV tiles按高索引到低索引消费。
- 同目录`compute/epilogue/lse_writer.cuh:80–86`从未量化row_sum去掉固定放大，输出对应scaled scores的普通log-sum-exp；它不是represented-P的log-sum。
- Python `nvfp4_attention_sm120_fwd`接受独立Q/K长度和`return_lse=True`，因此未来固定packets的单卡分片重放不要求新写attention kernel。

[官方softmax consumer](https://github.com/flashinfer-ai/flashinfer/blob/1b578de52924c973bd229fe4fd147499ead2781d/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/consumer/softmax.cuh)可作为公开入口；以上行号对应本地已读文件。

## 2. 代数推导：依赖具体落在哪里

以下为源码合同的理想算术化简，不是GPU实测或逐位等价证明。固定缩放后的score `z_j`、P的16元素microgroup成员及其最大值`a_g`；令`m`为该tile被消费时的running max，忽略倒数下限、exp近似与下溢：

```text
p_j = 2688 exp(z_j - m)
s_g =  448 exp(a_g - m)
q_j = R_E2M1(p_j / s_g) = R_E2M1(6 exp(z_j - a_g))
p̂_j = q_j R_E4M3(s_g)
```

因此E2M1码在此理想化下与running max无关；其公共因子已经消掉。剩余的结构性依赖在E4M3 scale：通常`R8(c s) ≠ c R8(s)`。改变KV分区或遍历路径可改变`m`，即使Q/K/V与correction完全固定、LSE合并使用高精度，也可能改变有效P权重。scale下溢、倒数下限、exp近似及FP32累加是另列的来源，不能全部归给E2M1码变化。

若所有参考max在同一整数log2格点，reference之间只差整数2幂；在相关值及舍入结果保持正常、有限范围且分组不变的条件下，二进制浮点舍入具相应2幂齐次性。这能消除上述scale相位依赖。它**不保证**消除subnormal/饱和、exp实现误差、阈值附近的码变化或FP32 reduction顺序误差，更不保证全模型逐位不变。

## 3. PNQ的分母与merge必须属于同一算子

令`w_j=exp(z_j)`，`ŵ_j`为回到共同reference后的近似未归一化权重。对分片r记`Z_r=sum(w)`、`Ẑ_r=sum(ŵ)`、`N̂_r=sum(ŵ V)`。

- 当前native合同：`O_r=N̂_r/Z_r`。使用真实LSE的权重`Z_r/sum Z`合并，可得到`sum N̂/sum Z`；但前提是各分片得到的`ŵ`与单段合同一致。
- PNQ合同：`O_r=N̂_r/Ẑ_r`。必须用represented-LSE，即`reference + log(sum represented P)`对应的`Ẑ_r`合并。仍用真实LSE会得到`sum[(Z_r/Z) N̂_r/Ẑ_r]`，一般不等于全局PNQ输出。即使量化器已齐次、各片row-sum精确为1，这个合同混用也会产生分区依赖。

归一化一致性和reference齐次性是不同条件；满足前者不自动满足后者。represented-LSE本身仍取决于各片如何构造represented P，不能把换LSE当作完整修复证明。

## 4. 无下溢的解析反例：仅说明机制可能存在

两个等长、各128-token的KV块，每块内部score恒定，分别为`0`与`log(1.3)`，V分别恒定为1和0。固定microgroups，单段路径先消费高score块。这里全部E2M1码为6，两个P scales为：

```text
低score块：R_E4M3(448 / 1.3) = 352
高score块：R_E4M3(448)       = 448
```

两者均在正常范围。完整PNQ路径输出`352/(352+448)=0.44`。每块单独作为分片时，其本地P scale均为448，量化正好精确；本例represented-LSE与真实LSE相同，合并输出`1/2.3=0.4347826…`。二者row-sum均为1，输出仍相差`0.00521739…`。

这是在理想累加下成立的score空间构造，排除了“只能是FP32 non-associativity”的过强否定；不是原生packet实例，也不是H3量级或生成质量证据。不要把该反例数值作为后续实测门槛。

## 5. 最近具体碰撞与未覆盖边界

| 一手来源 | 已覆盖 | 这次没有核实到的内容 |
|---|---|---|
| [MXAttention，Appendix H Eq99–105](https://arxiv.org/html/2607.24377v1) | PNQ用同一量化P更新分子分母；展开沿既定遍历的有效P，证明归一化一致。 | 没有证明换partition后有效P相同；没有当前固定NVFP4 packets的分区量级实验。不能把row-sum保证升级成partition invariance。 |
| [Hardware-Aware FP4 FA4，§4.3 Eq19–21、§4.4](https://arxiv.org/html/2609.04105v1) | represented denominator；另有固定/采样anchor和范围guard。 | 所读内容未直接验证当前NVFP4 E4M3路径的分区不变性。其NV/MX混合合同也不同于本地P4/V4。固定anchor可能消除部分依赖，须作为具体对照。 |
| [BAPS，§3.3、Algorithm1第4–8行](https://arxiv.org/html/2602.02071v1) | 明确先乘`log2(e)`，对rowmax取ceil，随后以整数2幂rescale；该段引用Softermax。 | 动机为减少递推乘法误差，未直接给NVFP4 E4M3 scale相位与分区可组合性实验。但整数log2参考max本身已经覆盖。 |
| [Shift-Accumulate Attention，§II-E Eq6](https://arxiv.org/html/2609.09208v1) | 明确使用`M=ceil(max(s)*log2(e))`和shift-exact更新；还对概率作PoT量化。 | 数值格式与目标不同，未直接测本地NVFP4问题；不因此把相同max构造认作新方法。 |
| [SageAttention3，§3.2 Eq5、Algorithm1第10行](https://arxiv.org/html/2505.11594v1) | 论文规定先取每tile的FP32 `sP1=rowmax(Ptilde)/(448*6)`，量化`Ptilde/sP1`，PV后乘回`sP1`。 | **我们的推导**：固定tile时该式可在低位编码前消掉任意reference公共因子；论文未用partition invariance表述。当前[官方softmax源码](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/sageattn3/blackwell/softmax_fused.h)与本地均采用固定2688/running max，未见此逐tile FP32因子恢复。论文配方是必要的“为何不用更简单方案”对照，不能与当前实现混称一致。 |

MpFA的global-mean correction及Sage的Kmean/LSE修正解决另一层合同，不自动处理上述P scale依赖。VC-Attention的V均值恢复和FP8 ExpCast也不能被泛化为已证明NVFP4分区可组合；本次未找到这一具体保证。此处新颖性状态为**partially verified**，不是“无人做过”。

## 6. 最小可证伪测试：先看实际量级，再判断贡献

先用已有、预先固定的少量真实QKV状态做CPU数值重放，不添加prompt或完整生成。固定全局一次产生的packets、QK correction、有效长度、128-token tile与16元素microgroup成员；只改变单段原逆序与两个连续分片各自逆序的计算路径。分别记录有效权重、输出、E4M3 scale舍入与范围事件，留同路径高精度/无scale舍入控制。

数值控制仅用于归因：当前合同、FP32 P scales、PNQ、BAPS式整数max、Sage论文式逐tile FP32缩放。PNQ必须分别检查真实LSE和represented-LSE，避免把错误merge协议当机制阳性。比较分区差异与匹配高精度分片误差、总量化误差及实际输出方向；不预设任意“显著”百分比。

CPU结果若值得原生核验，再用已有packed API在单卡完成P=1/2的attention重放；只切分合法packed视图和correction列，不重新量化各片Q/K/V、不重算各片均值，不先搭多卡后端。还须核对本地permutation与scale布局，保持逻辑token及microgroup身份。

目前没有H3实测，不能说该问题有害、成本不可接受或可投稿。若现成参考max/逐tile缩放就能有效解决，仍应记录具体部署发现；新的算法或系统claim则需要进一步指出这些已知方案尚未解决的数值边界、实际资源成本或并行约束。

## 7. 策略复核：停止新方法立项，仅保留一次封顶合同审计

2026-10-02；读完[报告017](/home/wjq/workspace/svdquant-exp/research_state/reports/017_20261002_query_group_phase.md)后的独立复核。这里只补核一个最关键一手来源：[Hardware-Aware FP4 FlashAttention-4，§3.2、§4.4、Appendix A.4–A.5](https://arxiv.org/html/2609.04105v1)。本节收紧上一节的立项判断，不修改既有实验或否定未测量的数值现象。

**新增的精确碰撞。** §3.2 Eq11直接给出 `Q_E4M3(G·a_B/6)/G`：低幅P块的E4M3 scale可舍入为零；row-level stabilizer能修范围，但其scale与inverse correction向online路径增加状态。A.4记载稳定化修好了真实模型失败，却消耗了期望速度收益；A.5明确将“robust global factor把工作放回暴露的P生产关键路径”“稳健NV/NV太慢、最快shiftless方案依赖分布”列为放弃NV/NV、采用MXFP4 P/V的理由。§4.4甚至已处理represented denominator中乘法次序导致的subnormal flush。故**“低位scale范围与稳定化成本冲突”本身也不是尚未覆盖的新残余**。这些结论来自其GB200/B300合同，不直接证明SM120同样慢；但换到SM120也不自动构成方法贡献。

**为何现在停止新方法候选。** 固定canonical KV tile及microgroup时，Sage论文的tile SF1可把每tile有效权重写成 `exp(a_tile)·QDQ(exp(z−a_tile))`（省略固定编码常数）。其中`a_tile`不随Ring遍历/分片reference变化。各片只是在共同reference下求这些固定tile的分子/分母和，再以匹配的normalizer合并；理想算术下已具可组合性。PoT max是另一个已知正常范围构造。PNQ需要represented-LSE，而非真实LSE，是与所选算子匹配的merge合同，不是新的归一化方法。上述论证保留范围、exp与浮点归约误差边界，不能宣传逐位不变。

“任意partition”目前也没有自然产生新的系统需求：把已有canonical tiles分配到不同rank，与切碎/改变quantization microgroup是两件事。前者可沿用tile SF1；后者已改变近似算子，不能要求它无条件保持原输出。本项目尚没有一个必须切碎这些组、又无法用已有padding/packet metadata保持其身份的真实调度约束。不要为立题先搭Ring后端或制造此要求。

**仅允许一次封顶合同审计。** 使用已有E018状态与packets，在固定tile/group身份下重放单段与两片；包含已知tile SF1/PoT、正确分母以及高精度控制。不新增prompt、视频、GPU框架或一轮新的kernel对照。若现成方案去掉reference项，结案为现有实现/文档合同发现；若残差来自已知范围或浮点问题，也先按已知机制结案。只有发现一个可复现、在正确已知方案下仍存在的具体代数遗漏，或拿到实际部署要求与关键路径证据，才另行评估；不能仅因非零差异或较大误差续预算。停止理由是已有解法与缺乏新的约束，**不是“尚未证明最终视频质量受损”**。

**下一项的诚实边界。** 当前没有足够证据推荐一个应立即持续投入的新论文问题。6张空闲72GB卡是可行性资源；Wan原生NVFP4下额外LR成本大于收益，支持“无在线旁路”的执行预算；H3局部误差排序与辅助视频评分不一致，支持拒绝用该代理直接选配方。这两条还没有证明现成QAT/量化蒸馏无法满足该预算，也没有共同指向一个新机制。唯一可以保留的下一类问题是“固定一个原生主GEMM、无在线LR/高精度回退的执行图，能否通过训练恢复独立评价质量”；但直接QAT/蒸馏本身就是现成方案，在其具体失败约束出现前，不把这句话注册为新方法或承诺顶会贡献，也不自动开启大训练网格。

## 8. 模态概率质量：VC-Attention直接覆盖修正，保留有限读出

2026-10-03归档前次定向核查；本节只保存先前工作结论和E019可复用读出的定义，不报告新实验，也不更新E019运行状态。**停止此方向检索，不扩成已有方法网格。** 全行row-sum守恒不保证text/audio/video三个KV区间各自的质量正确，但这一差别不自动产生新方法。

**直接碰撞。** [VC-Attention §3.2 Eq6](https://arxiv.org/html/2609.15810v1)将V按块分成均值和残差，使用

```text
A_i <- alpha_i A_i + Ptilde_q,ij R_q,j + r_ij mu_j^T
l_i <- alpha_i l_i + r_ij
r_ij = Ptilde_ij 1
```

这里`r_ij`来自**量化前**的概率tile；低位PV只计算残差，均值项用高精度块质量恢复，并沿同一个online rescale累加。因而“模态/组V均值＋精确group mass校正”已有直接代数覆盖；把硬件/聚类组改成三个模态区间，不能立新方法claim。其保证是块常数分量的恢复，不是全部残差路径精确，亦不是`P_q`本身已保持各组质量。Eq6的分母也累加未量化`r_ij`；若与PNQ组合，必须重新核对分母，不能直接混接后继续声称原保证。

[MXAttention的PNQ，Appendix H Eq102–105](https://arxiv.org/html/2607.24377v1)把最终有效represented P的**全行总和**用作分母，确实没有保证任意子集的质量等于量化前softmax质量。“每区间内PNQ，再乘高精度区间质量”是分组归一化与softmax分解的自然组合。本次未找到完全同名的单篇方法，不能把检索空白当成新颖性。**两条自然修复均不进入新方法立项；VC的均值修正路线属于直接碰撞。**

**E019仅可保留的更具体observable。** 固定score与decoded V，按真实packed标签划分KV模态，排除padding；使用已换算到最终共同reference的有效权重，不能直接相加各tile不同reference下的P。令`p=softmax(z)`，`w_hat`为量化有效未归一化权重，对模态集合`M_m`定义

```text
rho_m     = sum_{j in M_m} p_j
rhohat_m  = sum_{j in M_m} w_hat_j / sum_j w_hat_j
delta_m   = rhohat_m - rho_m
```

这组读出能直接检验“PNQ总和为1，但模态份额偏移”。同时保留PNQ前原合同的各模态质量、原始`rho_m`、绝对signed偏移，以及参考概率被零化的质量；若给相对变化，须同时展示原始质量，不能让极小分母制造效果。区分三种来源：E4M3 scale整组归零、非零scale下E2M1组内尾部归零、其余非零值舍入带来的再分配。按质量统计，不能以零元素比例替代。

NVFP4采用microscaling，**小绝对概率并不自动被抹掉**：一个整体较弱但组内平坦的模态仍可能被有效表示。上述读出尚不能证明弱模态受系统性压制、跨模态gate损伤、条件信息丢失或生成质量变化；也不能将E017的评分排序反转归因于它。若发现质量偏移，也只是现有算子的有限诊断，不据此追加VC/PNQ/模态修正等GPU或生成对照网格。
