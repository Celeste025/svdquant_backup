# E064 — 真实量化深度路径上的四角来源诊断

2026-10-04，执行前协议。H3 CFG1、既定 p30/s5 与 p36/s14；同时保留 plain 和 legacy SVD。此为来源定位，不是新方法；不以恒等式或已知跨层误差累积认领创新。

## 动机与竞争解释

E014 完整模型中 p30 plain/legacy 视频 SSE=.849813，p36=.998654，但两配方误差向量 cosine 仅约 .520/.467。E015 下一 teacher 状态仍有配方差，E061固定smooth初始化没有一致解决它。不同量化自然可以产生不同方向，这些数值本身不证明特殊机制；值得确定的是后续投入该针对当前块的局部量化残余，还是已经偏移的输入与量化块之间的作用。

- H0：当前块误差主要是 BF16 块传播输入差与 teacher 输入上的固定局部残余；额外输入依赖项小或抵消，不推进“量化反馈放大”。
- H1：在预选多个深度、两个固定状态上，量化块的输入依赖项有重复的明显正净作用，且在去通道均值后保留，才值得单独设计后缀干预。
- H2：该项虽存在，但主要是普通 residual carry、通道偏置或算术控制失败；不能作为新机制动机。

CLQ §3.2/3.4 已有量化前缀校准及跨层目标；[跨层误差补偿原文 §3.2](https://arxiv.org/html/2607.14630v1)已有 teacher 局部残余和量化有限差分的精确递推。因此本次分解没有理论新颖性；发现输入依赖也不等于提出比现有 block/QAT 更强的方法。前期 E062/E063 阴性不提供本假设阳性证据。

## 固定设置与四角

预选 blocks **0/12/24/36/49**，两状态×两量化臂，共20格。先分别完整运行 BF16、plain、legacy SVD，实际输入及最终 raw/velocity 必须复现 E014 同臂结果；所有非目标层、attention、smooth、rank和算术保持原样。各臂在自己的真实完整前向中捕获 block 输入/输出及完整实际 kwargs，不将后续输入替换成teacher。

每格 teacher 输入 hB、该量化臂自己的 hQ，定义完整 block（含 residual）函数 B/Q。四角为 BB=B(hB)、QB=Q(hB)、BQ=B(hQ)、QQ=Q(hQ)。BB/QQ 必须与完整模型捕获一致；QB/BQ 使用完整实际输入和相同模型 metadata 重新执行。量化始终使用现有真实native pack/GEMM，不用伪量化替代本次forward。

用保存的 BF16 输出转FP64后直接相减：

```
e_in = hQ-hB
qB = QB-BB
p = BQ-BB
echo = (QQ-BQ)-(QB-BB)
e_out = QQ-BB = p+qB+echo
rB = p-e_in
net_echo = ||e_out||² - ||e_out-echo||²
```

echo 是整个量化 block 的输入依赖差，含权重量化后的非线性、激活量化及BF16算术，不能单独归因activation/global scale。rB用于区分恒等 residual carry 与 BF16 非平凡传播。net_echo可负，不截断、不称独立能量份额；另报告 qB、p、echo、e_in、rB 的能量和交叉项，但不用echo范数代替损伤。对 video/audio/text 分别统计，video为主；在同一行域统一去各自通道均值后重复恒等式和净作用。

这里没有BF16小补偿加回，因此不是E059数值失败的重复；E059未得到的W/A来源问题仍未解决。

## 数值与执行合同

- block0上 hB=hQ 且 BB=BQ、QB=QQ，故 p=echo=0，两个量化臂都必须精确；不以容差放行。
- 每格 QB/QQ 均为4 native GEMM、4完整激活pack和2 BF16 SDPA；BB/BQ均无native、2 SDPA。记录调用数/输入签名。CPU数学检查与实际完整重放先过。
- 四角以完整形状前向，完整张量保存到DATA1，CPU逐行分块FP64统计，避免GPU大临时量。恒等式按相同数组复算；独立脚本不能只信runner统计。
- 不改历史执行源/结果。若实现需修复，已执行版本保留，新版本命名并保持原GPU截止；来源、布局、零控制或预算失败须如实报告不可判定。
- 若实施中发现计数需区分从完整模型继承的角和单独重放的角，记录实际，不把继承角称额外调用。

## 读出与投入决定

全部20格汇报raw及去均值后的net_echo/||e_out||²，另报rB和e_in，不能把block0零控制当机制结果。四个非零深度中，某量化臂至少两个预选深度在两状态都出现 raw及去均值后 net_echo/||e_out||²≥.20，才允许提出下一次后缀因果诊断；这是投入门槛，不是统计/物理定理。无论是否晋级，都保留逐格数据，不换层/提示追阳性。两量化臂比较只作来源描述，不把本地block读出解释成最终配方排序的充分原因。

若不满足，停止本轮“输入依赖误差具有稳定有害放大”的解释，不启动校准或训练。若满足，也只有来源候选：须先用固定后缀及普通传播/偏置控制证明作用，进一步W/A控制后才可能归因，且还需成熟同预算强基线、独立视频验证。

## 预算与产物

CPU check先执行；GPU前重查空闲。最多6次完整DiT（2 BF16、2 plain、2 SVD），最多80次局部block调用（20 QB、20 BQ、最多40对角identity replay）；总native GEMM/pack调用≤960（4×200+40×4）、完整SDPA≤772（6×102+80×2）。具体身份重放安排可少于上限，但必须记录。单张空闲SM120，allocated≤60GiB；保存张量≤32GiB，DATA1；绝对GPU墙钟1800秒含装载/拷贝/统计，不自动另开预算。无TE/VAE/训练/新视频/MJ，非性能benchmark。

源码 `scripts/research/probe_h3_depth_four_corner.py`；小报告 `results/research/E064/`；大张量 `/data1/models/svdquant-wjq/research/20261004/E064/`。source、参考、launch和实际tensor hash绑定。两臂实际路径需要分别捕获，不能交叉借用。
