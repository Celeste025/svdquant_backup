# H3 支路表示筛选：暂不立项

2026-10-04；按 claim-prior-work-triangulation 流程，仅写此笔记。3次定向检索＋root追加授权的1次正交约束检索，采用论文/作者源码；无GPU。**目前没有实际 main/LR 大幅相消的观察；没有符合“同一有效W、同rank/native预算”的新方法。** 最多保留一个工程归因问题，不续E059数值网格。

## 已知事实与最强反例

[实际native代码](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_nvfp4.py:161)：`m16=scaled_mm(Aq,Rq,bias,output_dtype=BF16)`；[194–213](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_nvfp4.py:194) 的LR是两次BF16 linear，最后BF16 `m16+l16`。bias在主支。把已经落地的两支转FP32相加再转BF16，不能找回支路落地前的信息；要测这一floor必须区分主GEMM数值、LR中间rank激活舍入、两支输出舍入、最终加法四项。

[E059已保存结果](/home/wjq/workspace/svdquant-exp/results/research/E059/evaluate.json) 是6次zero复现通过、0次oracle完成；首个oracle的block0/fc2小片加回舍入RMS1.9447、Δ参考RMS18.8100，比值0.10338触发预定停止。该量测是`main+activation correction`，**没有保存实际LR输出，不是两支相消证据**；小片前2行也不能代表video整体。

按仓库方向，`W∈R^(out×in), A∈R^(r×in), B∈R^(out×r)`，列输入输出为`Wx`。若理想截断SVD `L=U_rΣ_rV_rᵀ, R=W−L`，则`U_rᵀR=0`和`LᵀR=0`，所以**任意不同输入x、z也有 `(Lx)ᵀ(Rz)=0`**。无bias时，整体范数相消系数`(||Lx||+||Rz||)/||Lx+Rz||≤√2`；逐坐标抵消仍可能存在。故“SVD拆分天然造成大幅整体相消”不成立。

**不能把这一理想反例套到本地legacy。** [export manifest](/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/manifest.json) 绑定standard_8p64s的quant_state路径/SHA/config；[现有standard源码118–125](/home/wjq/workspace/svdquant-exp/scripts/ptq_minimax_h3_svdquant_standard.py:118) 是`Q0=Q(Ws); L≈SVD_r(Ws−Q0); Rq=Q(Ws−L)`，并非直接SVD(Ws)。[官方DeepCompressor calibrator](https://raw.githubusercontent.com/mit-han-lab/deepcompressor/main/deepcompressor/calib/lowrank.py) 本就支持compensate初始化及跨轮更新Q，残差初始化不是错误。现有本地275–281则重新初始化Q0、换随机SVD seed选择候选；本地[LowRankBranch](/home/wjq/workspace/svdquant-exp/third_party/deepcompressor/deepcompressor/nn/patch/lowrank.py:48) 用FP32 randomized SVD，所读[官方实现](https://raw.githubusercontent.com/mit-han-lab/deepcompressor/main/deepcompressor/nn/patch/lowrank.py)用FP64 full SVD。export没有生成standard脚本SHA；配置/结构一致是关联证据，不足重建生成时源码。这里不撤销既有同实现对照，也不称论文所有基线无效。

## 先例碰撞及正交性缺口

| 主来源 | 已覆盖／剩余边界 |
|---|---|
| [SVDQuant §4.2–4.3](https://arxiv.org/html/2411.05007v4) | 高精度LR＋低位残差、Nunchaku融合down/pack及up/main已覆盖。本文公式没证明本地分别BF16落地与其融合kernel同数值；本轮未审计Nunchaku accumulator，不能用“融合”代替证据。 |
| [LQ-LoRA §3.1–3.3](https://arxiv.org/html/2311.12023v2)、[LoftQ作者页](https://www.microsoft.com/en-us/research/publication/loftq-lora-fine-tuning-aware-quantization-for-large-language-models/) | 联合低秩＋量化分解、交替优化、加权重构已覆盖；“重新选择分解降低输出MSE”没有独立新意。 |
| [QERA §III](https://arxiv.org/html/2410.06040v1) | 按输入协方差求低秩输出误差最优解。把低秩容量用于量化误差或保护特定输出方向已高度重叠；未从所读公式确认显式约束`UᵀRq=0`。 |
| [ResQ §4.1，式2–4](https://arxiv.org/html/2412.14363v2) | 在正交**输入/收缩维**子空间分别高低精度编码并消除交叉计算。不能等同于约束SVD残差的**输出左子空间**，但“正交子空间保护＋低比特”已是主源先例。 |
| [QuaSAR §3.2–3.3](https://arxiv.org/html/2608.14149v1) | 已把W4A4补偿失败归因数值不稳定，并用截断伪逆及低秩/位宽预算处理；其问题在校准求解器，不是推理支路输出相消。不能仅因用已知数学就排除机制贡献，但须有新的因果证据。 |

本次有限检索**未核实**直接要求`UᵀRq=0`的主来源；这不是“无人做过”，更不是推荐理由。令U为B列空间正交基，Rq量化后确可向U泄漏；本地补偿初始化甚至在量化前也不保证主支正交。泄漏量大不等于有害相消，可能是重构所必需，必须测符号及实际误差。状态：问题层面高度重叠；精确正交约束的新颖性仅部分核查。

## “同一有效W”不等于相同W4A4算子

1. 因子变换`A′=TA, B′=BT⁻¹`保持BA，所以不能改变理想main/LR相消；只能改变LR内部BF16舍入。普通幂二平衡是必要零假设，不应改名成新分解。
2. 跨支移动`Rq′=Rq+C, L′=L−C`虽保持`Rq+L`，实际算子`Rq·aq+L·xs`仍改变`C(aq−xs)`。任何收益必须分离这一**激活误差重分配**，不能全归于浮点floor。
3. 投影`Rq′=(I−UUᵀ)Rq, L′=L+UUᵀRq`可在实数域保持总W且LR不增rank，但Rq′一般不再能由原NVFP4网格精确表示；重新量化会再泄漏。在运行时投影native输出又增加高精度计算。两者都不是已实现的同预算解。

## 唯一最小判别与投入门槛

**先不新增GPU。** 若主流程仍需判定工程混杂，最多另立一次2个既定source-teacher输入的native复现；只记录block0四linear，不改任何输出，不增加oracle网格。对完整video/audio/text分别累计main/LR范数、点积和`C=(||m||+||l||)/||m+l||`，bias单列；固定各模态32行保存xs、实际packet/global、LR中间量和实际两支，不能重打包子样本改变global。

CPU用实际packed数值及已存A/B算高精度m*、l*，比较`round16(round16(m*)+round16(l*))`与`round16(m*+l*)`；另记actual与前者差，避免把native GEMM差或LR中间舍入算作支路输出floor。只看近零坐标的巨大相对误差不能过关。两输入video均出现C>2且纯落地误差范数达到同输入线性层总量化误差的20%，才值得核对现成融合/FP32输出控制；阈值只是资源标准。否则停止，不以少量坐标阳性挽救。

即使过关，首先归为基线工程；只有在已有高精度合并控制下仍有独立现象，且能构造一份packed W4＋原rank32 BF16两因子、原200 native GEMM、原DiT次数、无额外master/历史缓存的表示变换，并分别控制有效W与`C(aq−xs)`漂移，才重新讨论方法。当前没有这样的构造或数据，不申请新实验链。
