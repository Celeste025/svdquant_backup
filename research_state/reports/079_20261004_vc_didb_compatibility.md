# VC-Attention / DIDB-ViT 与当前基线的兼容性

2026-10-04，只读核查，无代码接入、无GPU实验。VC论文v1、DIDB论文v2及本地官方Sage3 api.py。

| 方法 | 当前H3+SVD+Sage3 |
|---|---|
| VC的V-Smooth | 数学与原模型兼容；需要改量化预处理和attention内部均值补偿。论文已实现Sage类FP4路径，本地未复现。 |
| VC的ExpCast-FP8 | 针对E4M3概率编码；不能直接替换当前NVFP4 P路径。 |
| DIDB-ViT完整方法 | 二值ViT的结构/训练方法；不能直接替换H3的attention kernel。 |

VC对V聚类并同步重排K/V，然后分块减均值量化残差；用每块未量化概率和恢复均值。各块概率和不是1，不能仅在输出随意加一个均值；当前Sage3 API没有对应分块均值补偿接口。H3非因果有效长段可在RoPE之后保持K/V同步排列；padding/refiner仍须按原合同。SVD linear与此在结构上可组合，组合效果尚未验证。FP4实验只采用V-Smooth，ExpCast用于FP8。论文及作者页未找到官方公开实现入口；MiniMax官方integration仓库列为paper only，所以不能说可直接pip安装。

DIDB以W1A1主干做图像分类/分割，需要训练。它不只用Haar：还利用O_i=V_i+sum_j P_ij(V_j−V_i)分析差分丢失，加入可学习shortcut与局部负邻域项；Haar作用于输入X，经新增高/低频二值投影形成Q/K，并非给现成Q/K加等价旋转。整个方法改变模型函数，不能无训练迁移到现成H3。检索未确认作者代码仓库。

**对C007的更正：** 上轮只重点提到DIDB的Haar近邻，遗漏它直接讨论attention差分信息保留，故碰撞比报告078描述更强。不能认领“首次保attention差分/高频”。可继续验证的窄残余：固定预训练H3权重和全精度目标函数，在PV收缩维量化前同时编码P/V和差项，并以同bit/scale预算验证可见细节收益。与VC的V-only residual和DIDB结构性补偿都必须区分；当前未证明新颖性或收益。

来源：[VC论文](https://arxiv.org/html/2609.15810v1)；[作者介绍](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck)；[MiniMax官方集成说明](https://github.com/MiniMax-AI/awesome-minimax-h3-integration)；[DIDB论文](https://arxiv.org/html/2507.02222v2)。
