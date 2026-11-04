# E204 融合概率捕获 profile

同 E203 赛车 480p/124 帧 step0 layer25、全 56 heads，GPU1。此次仅做 profiling，没有修改运行算子或重新生成视频。工具：CUDA events、PyTorch profiler、Nsight Compute（各一个匹配 kernel，17 次 replay）。

## 非 profiler 分段计时

每阶段 3 次预热、15 次计时。

| 阶段 | 平均毫秒 |
|---|---:|
| 融合 dense attention + 局部统计 | 40.50 |
| 概率归一化与 K128 map 归约 | 1.32 |
| 对照：dense attention + QK 最大值（含原接口归约） | 28.21 |

融合主 kernel 占两阶段约 96.8%；归约只占约 3.2%。局部统计形状 [1,56,148,318,128]，FP32 共 1,349,419,008 bytes；最终 map 约 2.45 MB。

## Nsight 硬件指标

| 指标 | QK 最大值 | 融合概率 |
|---|---:|---:|
| 寄存器/线程 | 255 | 255 |
| 实际 occupancy | 16.63% | 16.59% |
| local-memory spilling requests | 0 | 475,465,984 |
| DRAM 吞吐占峰值 | 1.43% | 3.12% |

Spilling 请求数不是字节数。Nsight replay 改变运行环境，报告内 40.37/55.95 ms 不与 CUDA events 混用。GPU 有其他低利用率常驻进程，数字是本次样本的测量，不是独占设备保证。

## 判断和建议

有优化空间，首先处理新增统计引入的寄存器 spill。代码在 update_mdo 前独立重算局部最大值和 exp，随后 online softmax 又计算这些量。优先将统计移到 update_mdo 后、P 转 FP8 前，利用已得到的指数权重与当前每行最大值计算局部 logsumexp；核对 exp offset/缩放以及无效 key 的处理。这样有望减少重复特殊函数运算及临时值存活，但收益需要实测。

次优先缩短统计函数及 QK/PV 寄存器存活区间，观察 spill 是否消失。255 registers/thread 本身并不能解释新增开销，因为两版相同，但 spill 指标有明显差别；shared memory 和 registers 同时限制 resident blocks。

中间缓冲区可进一步减少：不写只用于分母的 prefix 局部统计；必要时合并相邻 K64 后再写出，或压缩有效 query 行。合并会增加跨迭代状态，可能加重寄存器压力。降低存储精度需要额外验证 map 排序。

归约最多约 1.3ms 的量级，不应先投入主要优化精力；当前 DRAM 吞吐未接近峰值，不能把 1.35GB 中间量直接判为 DRAM 带宽瓶颈。也不能仅据此承诺能追平原 QK 最大值捕获。
