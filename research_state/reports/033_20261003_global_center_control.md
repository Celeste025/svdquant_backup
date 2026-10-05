# 033：更便宜的global强对照与下一项质量决策

2026-10-03。E037及独立CPU复核均完成；目标仍active，暂无可投稿核心贡献。

E036证明同一coarse16输出下，源侧FP4打包和分布式校正能降低两卡完整attention成本。E037补齐一个更便宜的现成选择：只使用一个global query中心。相同真实H3 block0输入、相同两卡token→head→token边界，统计、打包、校正、通信、attention及逆向输出均纳入计时。两种方案交错运行，每臂3次预热、10次测量；共52次native attention、零DiT。

| 方案 | 两rank较慢者wall中位数（范围） | BF16参照NMSE | 最大单卡allocated峰值 |
|---|---:|---:|---:|
| coarse16 | 21.592 ms（21.550–23.441） | .00285639 | 1.4342 GiB |
| global | 20.205 ms（20.167–20.429） | .00356130 | 1.3525 GiB |

Global快6.42%，但局部误差高24.68%，56/56 heads均更差。长耗时样本全部保留。两臂各自复现E034对应输出，没有发现布局异常；二者不具备相同精度，不能称等质量加速或相互严格支配。

应用层远端A2A发送总量由324.488降至287.971 MiB，均含154.875 MiB逆输出；另行统计的AllReduce逻辑peer-send量为.05469/.10938 MiB。没有把这些载荷量称为实测链路线速。独立CPU汇总8.17秒，CUDA未初始化。launcher7.18秒结束、worker退出、GPU释放。

**下一步E038：暂缓中心/通信网格，比较完整视频。** 固定四个公开VBench动作文本，每题两个seed，三臂均使用同一native SVD线性权重：BF16 attention、global、coarse16。沿用H3 20步、124帧、576×1024及同一VAE；共享实际初始噪声和条件，自由递推。问题只有一个：coarse16是否保住global丢失的实际语义或时序性质？MJ-VIDEO逐题/seed结果与时序辅助指标及可见样例共同报告；不把NMSE、运动幅度或奖励总分独立当质量结论。

若本批次没有清晰分离，或coarse自身退化，停止扩大中心表示路线，保留E036/E037工程事实。这是资源决策，不是证明所有生成都不需要多中心。E038单卡数值比较不提供完整两卡模型加速结论。

证据：[E037独立结果](../../results/research/E037/independent_summary.json)、[实验协议](../06_experiments/E037_sequence_parallel_global_plan.md)、[研究判断](../00_state/E037_research_decision.md)、[E038固定清单](../06_experiments/E038_center_video_manifest.json)。
