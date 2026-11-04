# E209：原版 DraftMap 与概率 map 开销比较

真实赛车 480p、124 帧、step0 layer25、56 heads，和 E208 相同 QKV。10% INT8 + 90% NVFP4，不跳块，maxpool_weight=0.2。full API CUDA event，2 次预热、8 次计时；排除模型加载和首次编译。概率版三行沿用 E208 数字，原版两行本轮补测。

| 方案/粒度 | 首步 ms/层 | 后续 ms/层 | 持久 map 缓存 MB/层 |
|---|---:|---:|---:|
| probability capture/reuse Q128K128 | 35.46 | 28.55 | 2.59 |
| probability capture/reuse Q128K64 | 36.44 | 28.61 | 5.04 |
| probability capture/reuse Q64K64 | 34.31 | 26.26 | 7.75 |
| original DraftMap rebuilt each step Q128K128 | 29.78 | 29.75 | 0.00 |
| original DraftMap rebuilt each step Q64K64 | 26.68 | 27.06 | 0.00 |

原版每次重建 DraftMap，所以后续列是重新评分/排序后的 attention 耗时，不是缓存复用。原版无持久路由缓存，0 MB 不代表没有临时 map/排序缓冲区。原版首步直接执行混合精度 attention；概率版首步执行 dense INT8 并捕获概率 map，首步计算政策不同，不把其耗时差全部视为 map 捕获开销。

这是原版 per-step DraftMap 路径的测试，不是我们早先改版的 QK 最大值首步捕获路径。当前本地 native header 的优化仅在概率捕获编译分支启用，原版每步评分与执行路径未切换为概率捕获。简单检查输出有限，未重跑视频。
