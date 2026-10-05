# E019：固定 FP4 packets 的分片合同审计

2026-10-02，GPU前登记。本轮是一次封顶的实现/数值合同审计，**不作为新方法立项**。更直接的近邻已覆盖E4M3概率范围与稳定化成本；固定128-token tiles下，Sage论文tile SF1在理想算术中已消除reference依赖。见[问题与最新策略复核](../02_problems/fp4_attention_partition_composability.md)。不以“分片后不同”本身称贡献。

## 固定输入和问题

复用E018的E017 block-mean自由轨迹p36/seed59526/s14、blocks0/24/48 post-RoPE Q/K/V。这里采用E018已经计算并保存的**global-mean FP4包**作为共同输入：Q/K/V codes、E4 scales均固定，不重新按片求均值或量化。Q长度pad22656，有效KV长度22539，56heads×128。只改变KV分区：完整177个128-token tile反向遍历，或从11264=88×128处分成两片各自反向遍历，然后按LSE合并。切点保留P16、V16 microgroup及128tile成员；右片padded11392/valid11275，117个尾padding始终屏蔽。

研究读出：这种分区是否改变有效概率/输出？有多少可由E4 scale参考点依赖解释，多少属于已有稳定化方案、输出格式或native数值顺序？只讨论固定输入的数值合同，不推断生成质量、并行吞吐或新方法价值。

## 原生与数学合同分开

E018只保存了correction SHA。本轮先在原SM120环境以官方global preprocessing恢复完整FP32 correction，必须逐tensor SHA重现旧记录，再保存。不得以CPU均值偷偷替换GPU合同。

每块原生调用严格四次：full BF16+LSE一次，输出必须与E018 global phase0全tensor SHA相同；full FP16、left FP16、right FP16各一次。后三次降低分片输出舍入的干扰，但仍不是FP32输出，不据此将全部差异归给P4。使用原consumer，causal=False、scale=128^-0.5、return_lse=True。LSE为自然对数，沿共同global K中心化后的同一score定义。Q包保持完整；K包按对齐行切片，V scale按物理64×4 tile切分，不能直接截表面最后一维。CPU必须验证两片解码与原包对应逻辑片完全一致。

数学分析固定8个head `[0,8,16,24,32,40,48,55]`；video query取latent帧 `[2,18,34]` ×帧内raster位置 `[0,287,575]`，加video起点1227，共9行/head、72行/块、216行总计。选择在看本轮结果前固定。保存实际packet解码的Q/K/V及恢复的FP32 correction列，用FP64 `Qhat@Khat.T+correction` 再乘scale构造共同score。**这是固定包的数学score重建，不是捕获的native MMA accumulator或逐bit exp模拟**。没有重写H3模型。

## CPU归因对照

所有对照共享同一score、V、128tile与16microgroup。完整N128先更新running max，不能改为两个N64分别更新。为了隔离E4存储，主对照共享固定E2M1码 `R2(6 exp(z-a_group))`，不分别从低精度p/s重算码。native倒数floor、exp近似和FP32累加属于数学参考未逐位模拟的边界，事件单列。

固定七个读出合同：

1. FP64无P量化，校验full/split代数归并。
2. E2码＋FP32 scale，保留非E4底噪。
3. E2码＋E4M3 scale；相同FP32 scale再转E4，分母用未量化概率的FP64和来隔离scale效应。native分母实际是FP32 online row_sum，不能混称数值精确。
4. 同一E4有效概率的PNQ，分片用represented-LSE正确归并。
5. PNQ但故意用true-LSE归并的负对照，不能把合同混用产生的误差算作新机制。
6. 已有ceil-log2 reference max；以canonical scale及ldexp构造2幂变换，避免libm独立exp的halfway误差伪装成非齐次性。
7. 已有Sage论文tile SF1：整128tile的max作为P局部reference，PV后恢复tile因子。不可误用全局rowmax或N64 max。

每row记录full/split有效权重L1差、mass偏差、输出差能量及各自对共同无P量化参考的误差，保留 `ΔE=||delta||²+2<e_full,delta>`。统计normal/subnormal/zero scale与s<1e-8事件及对应真实softmax质量，tail padding不入分母/事件统计。原生FP16 full/split用FP64 LSE权重合并，另报告无P数学结果先cast FP16再合并的格式底噪；不把融合输出变化自动当作变差。不设事后“显著百分比”门槛，不将216相关行当独立生成样本。

## 资源与结束条件

0次DiT、0视频、3块×4=最多12次原生attention。物理GPU5启动前稳定空闲检查，tmux与独立日志，GPU阶段共享900秒/60GiB上限；CPU数学归约六线程、不得初始化CUDA。数据在 `/data1/models/svdquant-wjq/research/20261002/E019`，小报告在 `results/research/E019`。源/协议在GPU前冻结；E005–E018不改。输入或packet parity失败时保留记录、停止解释，不默默换层/切点/样本。

结束后给出实际数值与已知控制效果，即可关闭此合同审计。即使原生有差异，也不自动升级研究主张；若现成方案解决且无新约束，不开发新kernel或多卡框架，不扩大成基线网格。
