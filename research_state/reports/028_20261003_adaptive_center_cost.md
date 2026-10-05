# 028：自适应中心精度可以恢复，构造与消费成本仍是问题

2026-10-03。E031完成三层9次原生attention与全部计时，独立CPU复算通过。**经典随机近似恢复了大部分同样本中心的误差优势，但当前Torch准备实现仍比自由块均值慢；精确GPU SVD则明确过贵。** 尚未实现新的消费者，不是完整路径速度结果。

复用H3 p36/seed59526/step14的三层0/24/48，rank固定16。fullSVD、零次/一次幂迭代的随机range finder均在GPU FP32运行；Ω只用固定seed20261031生成一次，所有层/重复共用同一冻结batch。Q/K均值仍按官方BF16顺序，中心采用E030可分解FP32合同。9个完整输出保存，零DiT、零新视频。

| 构造方法 | μ→basis/A/center | raw Q/K→Qcenter/T17 | 三层保留global→block误差优势 |
|---|---:|---:|---:|
| full GPU SVD | 210.50–212.58 ms | 215.87–217.44 ms | 92.16%–96.65% |
| range0 | 1.839–1.842 ms | 10.340–10.356 ms | 76.71%–91.69% |
| range1 | 3.613–3.614 ms | 12.146–12.159 ms | 90.66%–94.55% |

raw Q/K强预处理baseline：global为5.613–5.621ms，block为8.427–8.437ms。每项3warmup＋10重复，表为CUDA中位范围，原同步wall与全部数组保留。两个范围是分别完整调用，**不能相加**；候选raw准备止于T17，不含完整correction组合、FP4pack或attention；baseline产出自身完整correction，不能按准备差值推断未来消费者速度。这里也不是E027完整QKV到output的31.78ms边界。

| 对BF16输出的NMSE | 层0 | 层24 | 层48 |
|---|---:|---:|---:|
| 自由块均值 | .00241969 | .00685596 | .00641365 |
| fullSVD | .00245793 | .00706813 | .00657264 |
| range0 | .00251454 | .00748673 | .00688275 |
| range1 | .00248255 | .00710886 | .00662317 |

range1三层全部168head优于global；range0有一个head更差。相对单文本固定basis，fullSVD/range0/range1分别改善165/162/163个heads，仍非每head统一改善。三层pooled排序均fullSVD、range1、range0，由好到差。优势保留比例只指张量NMSE差，不能写成视频质量。

候选raw产物397.91MiB，对比block1166.40MiB、global314.59MiB；对应allocated峰值2.437/2.970/2.131GiB，共同resident baseline .61276GiB。峰值包含Torch中间张量，不是只有表大小；reserved也受暖allocator影响。FP32 fullSVD的basis正交最大偏差4.54e−5、QR约4.34e−7均保留，未追加任意门槛。运行97.86秒、进程正常退出。[独立汇总](/home/wjq/workspace/svdquant-exp/results/research/E031/independent_summary.json)已重读9输出、24个计时范围的全部原数组、冻结Ω及小因子；projection只复核已存scalar一致性，没有把未存的本轮GPU均值冒称独立重建。

当前停止把精确SVD当在线方案；不把Torch的负准备成本当专用实现下界，也不为range1直接重写复杂producer。E032随后完成更简单的共享中心表：16个中心、固定6轮聚类，每Qtile仅保存一个id，Q−C[id]与T=C Kcᵀ使用同一BF16 C。三层3次原生attention、零DiT，运行43.69秒；当前精度测试先展开T后调用旧kernel。

| E032共享中心 | 层0 | 层24 | 层48 |
|---|---:|---:|---:|
| 对BF16输出NMSE | .00280712 | .00804822 | .00749995 |
| 保留global→block误差优势 | 66.06% | 55.97% | 71.74% |

全部168个head优于global；相对range1，三层pooled NMSE高约13.1%–13.2%。相对固定迁移basis，104/168个head改善，但三层pooled误差高2.0%–3.0%，反映head能量差异，不能称统一优劣。实际BF16中心/id及全输出已保存，独立CPU汇总重新计算了误差和中心失真；各head都有16个有效中心。

E032中心构造1.768–1.805ms；raw Q/K准备8.486–8.493ms，对比block8.428–8.431ms，重复范围重叠。准备allocated峰值约2.207GiB，对比block2.970GiB，下降约25.7%；仍未含FP4pack及attention。不能用这些数字宣称完整加速。[独立汇总](/home/wjq/workspace/svdquant-exp/results/research/E032/independent_summary.json)已完成；一次人工CPU fixture的精确距离并列差异已保留并修正测试输入，未修改生产算法，也未追加GPU重试。

下一步E033只实现一个共享行consumer，并测完整QKV→output：global、freeblock、精确Qchunk4096/8192、K16共享表，三层各3次预热与10次测量，在线重算中心。先完成10次同pack/table新旧消费者正确性比较，再做468次attention计时；GPU0硬预算900秒含冷构建。若被实际精度/时延/显存前沿支配，就停止该实现路线，不扩参数网格。

已核[近邻](/home/wjq/workspace/svdquant-exp/research_state/04_prior_work/bounded_query_centers.md)：centroid共享、聚类和去均值均有已有构造，不能把kmeans本身包装成新方法。本文记录的是E033执行前的判断；后续consumer与完整路径结果已完成，见[报告029](/home/wjq/workspace/svdquant-exp/research_state/reports/029_20261003_shared_row_consumer.md)。当前仍无视频质量或可投稿核心贡献。
