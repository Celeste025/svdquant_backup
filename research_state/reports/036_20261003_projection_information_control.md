# 阶段036：低秩侧信息对照与候选收尾

2026-10-03。E040完成。保留E039的通信优化代码和实测收益，停止把这个接口迁移单独扩成论文主线；总体研究目标继续active。

E039显示FP4 packet＋down32的两卡完整投影边界为4.897ms，BF16回传为6.348ms，快22.85%。本轮进一步固定同一个实际packet和native main，仅改变低秩支可用的信息，排除“主支已量化，所以低秩也吃解码输入完全一样”的解释。

| 低秩信息来源 | 有效段最终输出NMSE | 真实53行padding NMSE | 有效段down NMSE |
|---|---:|---:|---:|
| 原full-K down（参考） | 0 | 0 | 0 |
| E039源侧合并的down32 | 5.93241e−8 | 3.06950e−8 | 1.08155e−5 |
| 同packet解码到BF16，再算down | 1.83369e−6 | 1.75478e−6 | .00890076 |

**侧信息确实保留额外信息，但尚不能称质量必需。** 解码输入方案的有效段最终NMSE约为side的30.91倍，绝对RMSE为.02077、参考输出RMS为15.33745；side的RMSE为.003736。相对倍数不能替代绝对误差和真实视频证据。此次没有测解码替代的速度或视频质量。

两个目的段原样保留11264/11328行，含53真实modelpadding。每段只生成一个packet和一个native main；原down和side down直接读取E039的保存状态，第三种down由该packet的独立signed-nibble LUT解码计算。三种均按原BF16 up、1.0乘数及main+branch次序执行。E039原输出与side输出的重放漂移、两个global漂移均为零；未把逐位相等作为科学门槛。

正式GPU0仅2 pack、2 native GEMM、2 decode、2新down、6 up，0通信/attention/DiT。worker6.851秒、launcher7.378秒，独立CPU归约6.672秒complete；没有失败或重跑。worker1369659已退出，GPU0释放。实际packet/main/down/输出均保存于DATA1/E040；独立归约读取这些产物，未用CPU GEMM冒充原GPU计算。

**贡献范围判断：当前不足独立论文。** SVDQuant本就保留量化前输入给低秩支；FlashInfer已提供SM120的外部packet＋down消费ABI。把这套表示搬到A2A边界的实现有用，但本次信息对照没有发现超出该已知设计的新机制。最近文献未明确给出同一接口，不等于新颖性证明。也不能用更多卡数/模型/视频数量填补这一缺口。见[有限查新与API审视](../00_state/E039_next_decision.md)、[C006范围判断](../05_scope/paper_unit_gate_C006.md)。

下一步回到问题筛选，先更新现有证据与最近邻的对应关系，再保留一个具有独立残余的新问题；不默认扩展C006完整SP，也不重启已停止的中心/聚类/普通QAD网格。尚无可投稿核心贡献，工程结果完整保留供后续部署复用。

证据：[E040固定协议](../06_experiments/E040_projection_side_information_plan.md)、[实际执行](../../results/research/E040/run.json)、[独立归约](../../results/research/E040/independent_summary.json)、[E039完整成本](035_20261003_output_projection_boundary.md)。
