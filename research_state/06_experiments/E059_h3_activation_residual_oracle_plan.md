# E059 — 保留native GEMM的激活残差移除诊断

2026-10-04，exploration；运行前记录。不是新方法或部署优化。

## 动机和决策问题

E056–E058建立H3两个窗口的持久方向与实际两步增强，但不能归因固定W还是动态A。交互有抵消作用，故不以压小交互范数为目标。若相干方向有实质部分能通过激活侧改变，才值得研究一份固定packed W预算下的干预；若没有，就停止当前activation时序候选，不购买双权重/缓存预算挽救。

传统software W4A16开关不够干净：已有E009单状态legacy QDQ/native差的能量，相对native量化误差能量为video0.1549/audio0.3134；相应范数比约0.394/0.560。这不是当前两个窗口的数值，但明确不能未经匹配归因。因此本实验保留同一native FP4主GEMM，只加明确的oracle补偿，不切换整个主算术路径。

## 干预与竞争解释

每个主linear沿用原BF16 smoothing得到x_s，调用原packer及native main（含bias），保持原BF16低秩支路。以实际codes/E4M3 scales/FP32 global定义解包A_q/W_q；禁止默认BF16 decode引入额外重构舍入。

令W_bar为不含tensor global的packed residual权重，预定
`delta = FP32_matmul(FP32(x_s)−FP32(A_q), FP32(W_bar).T) * g_W`，关闭TF32。
`main_corr = BF16(FP32(main_native)+delta)`，然后原顺序`main_corr+branch`。

- **zero控制：** 同样FP32加回接口，delta=0；完整6次native调用，raw/velocity必须与已保存同输入native逐元素一致。
- **oracle干预：** 200主linears全部执行上述补偿，网络内部后续activation允许随干预变化，这是总效应的一部分。

H_A：激活表示残差对持久的正方向有实质贡献，移除后跨步相干误差明显减少。H_W/other：相干方向仍主要保留，或A原本提供有利抵消；阴性不等于证明“W独占主导”。native main的有限精度误差仍保留，oracle不是精确W4A16，不作唯一W/A方差分解。

## 最小设置

只用SVD强基线，固定两个窗口各三个已有输入：第n步teacher（E014），第n+1步teacher（E015 SVD模态BB），第n+1步实际偏移输入（E015 SVD模态QQ）。source为p30 s5→6与p36 s14→15。共6个输入×zero/oracle=12次DiT。先全部zero通过，再执行oracle；不重采teacher、不生成视频、不改变attention/scheduler/权重/低秩。

第一步teacher与第二步teacher的BF16参照来自E014/E015；第二步偏移输入的BF16参照来自E057。只比较固定输入预测，不把它当oracle完整新轨迹。保留video/audio全部输出，决策以video为主。

## 数值与资源约束

每个oracle case的block0四linear：前2行、前32输出与FP64实际code/scale/global公式核对；预定FP32计算相对FP64校正RMS误差≤1e−3，max误差≤1e−2×参考RMS。另比较global先乘W的FP32次序，不根据效果改所选次序。零参考须误差为零。记录BF16加回舍入幅度，若其RMS≥校正RMS的10%，停止来源归因并明确控制分辨率不足。bias、LR不进入delta；global只计一次。

权重逐层FP32解包、行分块计算，不跨层保留FP32权重，不优化或改新kernel。最多一张空闲SM120、1800秒、12次完整前向、allocated<60GiB；CPU check先确认6份actual input。所有输出独立保存，执行源不覆盖。该oracle增加高精度GEMM，绝不是可部署收益。

## 读出及下一决定

令e_i=Q_i−B_i，o_i=O_i−B_i，d_i=Q_i−O_i，验证e_i=o_i+d_i；逐步报告三项能量和带符号交叉。主读出去通道均值的相邻点积`<e_0,e_1>`与`<o_0,o_1>`、cosine、各步能量，完整展开removed/residual的跨步交叉项。不能把剩余误差直接叫纯W误差，不能把跨项当独立份额。

若两个窗口剩余正相邻点积都下降≥50%，且任一teacher端点误差能量没有增加>10%，则只允许设计一个具体等预算activation干预；方法仍须碰撞先例，不能由oracle宣称SR/互补必然有效。若未同时满足、结果反向/异质或数值控制失败，停止当前activation时序方法候选，不扩prompt/step/精度网格。阈值是资源投入标准而非显著性或不可行定理。第二步偏移输入误差变化只作必要竞争解释，不用较好子结果改主判据。

## 先例/边界

TCEC/QuAKE已做历史补偿，Ditto已做跨步activation差分和旧输出恢复，De-biasing Diffusion的随机舍入方法全文细节仍未核实。来源开关、oracle与FP32精度处理均不是创新。若后续成立，候选必须仍是一份packed W4、原rank/DiT调用数、无完整FP16 master/第二份W/逐层activation历史；闭环input依赖先前舍入随机数，固定teacher态负相关不自动等于闭环无偏。
