# 027：query 中心可压缩，但尚无可部署收益

2026-10-03。**同样本 rank16 中心在三层真实原生 attention 中，保留了自由块均值相对全局均值 91.66%–96.66% 的张量误差优势。** 这是当前新的正向数值信号；三种选中心几何没有统一胜者，不能据此宣称新方法、视频质量或加速。

E028先复用H3 p36/seed59526/step14的block0/24/48，CPU计算三层×56heads×三种几何的504条谱。E029随后固定rank16，用对应最优中心同时重做Q量化和恢复校正，真实K/V packets固定；加官方block/global两端点，共15次native attention、零DiT、零新视频。没有按结果挑层/head。

| 对BF16输出的 pooled NMSE | 层0 | 层24 | 层48 |
|---|---:|---:|---:|
| 全局均值 | .00356130 | .00956374 | .01025691 |
| 自由块均值 | .00241969 | .00685596 | .00641365 |
| rank16 普通几何 | .00246452 | .00706893 | .00657402 |
| rank16 K-score几何 | .00245782 | .00708181 | .00658275 |
| rank16 K4误差几何 | .00246680 | .00706462 | .00654315 |

所有候选head均优于global；相比自由block，504个候选head比较中32改善、472变差。各候选pooled误差增加1.58%–3.29%，所以是用少数中心方向保留大部分精度优势，而非超过原block精度。保留比例定义为(global−candidate)/(global−block)的NMSE差比例，不是视频质量百分比。

仅看score谱会高估集中程度：r16的K-score剩余能量为0.26%/2.88%/0.74%，K4-error为0.89%/8.07%/3.63%，普通几何为2.27%/11.90%/9.10%。不同几何各自优化不同basis。第48层K4-error加权残差3.63%，head中位数却7.48%、最高14.33%，不能用pooled值替代典型head；r4/8也并非三层统一高度集中。

E028 CPU13.10秒、无GPU；E02965.57秒包含I/O/SVD/归约，峰值6.04GiB包含oracle构造，均不当部署性能。独立CPU重新归约504条完整谱和15份完整输出通过，6个现场端点与历史输出一致；完整head分布见[谱汇总](/home/wjq/workspace/svdquant-exp/results/research/E028/independent_summary.json)和[原生输出汇总](/home/wjq/workspace/svdquant-exp/results/research/E029/independent_summary.json)。原始结果和小center/A/B因子保留。

当前basis由同一样本计算，E029仍物化完整校正矩阵；且逐元素BF16中心舍入会破坏严格rank17。实际消费者还要付出多行读取与组合成本：现内核每tile读一条128-float校正，直接换17行不能按容量比推断时间。可行实现方向是在producer warp中组合17行后写回现有一行shared缓冲，但流水线发布/等待次序必须正确，尚未实现。

E030跨文本迁移也已完成并独立复算。从已有p30/seed49771/step14重放一次DiT获得固定Euclidean rank16 basis，再测p36同一步三层。下表同时保留逐块BF16中心和真正可由17行T加系数A表达的FP32中心；它们仍在原kernel前完整物化correction。

| 固定basis迁移结果 | 层0 | 层24 | 层48 |
|---|---:|---:|---:|
| BF16中心 NMSE | .00272640 | .00788449 | .00735129 |
| 可分解FP32中心 NMSE | .00272537 | .00788232 | .00734960 |
| BF16中心相对同样本Euclidean误差增加 | 10.63% | 11.54% | 11.82% |
| BF16中心保留global→block优势 | 73.13% | 62.02% | 75.60% |
| 按head保留率中位数 | 71.17% | 50.24% | 55.29% |

固定basis明显不如同样本basis：三层分别53/56、56/56、56/56 heads退化。仍有166/168个head优于global，但第0层head5、16两臂都比global更差；全部保留。FP32合同的pooled误差仅比BF16中心低0.023%–0.038%，逐head正负均有，不叫质量等价，也不能只归因某一种舍入。两合同共同改变了系数计算、Q减法/center舍入和FP32结合次序。

E030总78.28秒、1 DiT加6次额外attention；捕获DiT内另有50次FP4attention/52次BF16 SDPA/200 GEMM。峰值37.60GiB主要包含完整模型转换，模型释放后allocated仅.00891GiB；donor重放输出与旧轨迹一致。[独立迁移汇总](/home/wjq/workspace/svdquant-exp/results/research/E030/independent_summary.json)已复算六输出与basis，当前全部GPU任务已退出。首次CPU检查后发现回调关键字名不匹配，在GPU前修正，旧检查记录保留。

结论收窄为：同样本中心存在压缩信号；单文本固定basis只保留部分收益，缺乏跨样本稳定性；FP32可分解合同本身未造成同等量级的新损失。实际[RoPE/模态源码核查](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/restricted_center_rope_notes.md)也不支持“最低频率自动解释rank16”：32维不旋转，视频帧内时间不变，文本/音频存在直接反例。暂不开发完整consumer；下一步只评估样本自适应中心构造的真实成本和精度，决定是否有超越简单PCA的研究空间，不扩训练/rank/视频网格。Sage/MpFA及公开低秩logit侧码等[近邻边界](/home/wjq/workspace/svdquant-exp/research_state/04_prior_work/bounded_query_centers.md)已记录，低秩代数本身不是贡献。仍无可投稿核心成果。
