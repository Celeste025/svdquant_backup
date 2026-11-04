# E210：80% 跳块、10% INT8、10% NVFP4 的五项对照

同 E208/E209 真实赛车 QKV：480p、124 帧、step0 layer25、56 heads。接口设置 sparsity_ratio=0.8，保留块中 NVFP4/INT8 各 0.5。前缀仍为 dense INT8。CUDA events 2 次预热、8 次测量；缓存为十进制 MB。原先 90% NVFP4+10% INT8 的数据保持原样。

| 方案 | 首步 ms/层 | 后续 ms/层 | 持久 map 缓存 MB/层 |
|---|---:|---:|---:|
| 概率版 Q128×K128 | 35.32 | 13.19 | 0.62 |
| 概率版 Q128×K64 | 36.45 | 13.20 | 1.11 |
| 概率版 Q64×K64 | 34.29 | 12.48 | 1.73 |
| 原版 DraftMap Q128×K128 | 14.45 | 14.45 | 0.00 |
| 原版 DraftMap Q64×K64 | 13.02 | 13.15 | 0.00 |

概率版首步仍是全量 dense INT8+map 捕获，不受跳块比例直接加速；后续才执行稀疏混合计算。原版首步/后续均执行稀疏混合计算并重建 DraftMap，0 MB 仅代表无持久路由缓存。

## 后续混合主 kernel profile

| 方案 | 寄存器/线程 | spill requests | achieved occupancy |
|---|---:|---:|---:|
| 概率版 Q128×K128 | 255 | 1710912 | 16.59% |
| 概率版 Q128×K64 | 255 | 1710688 | 16.59% |
| 概率版 Q64×K64 | 168 | 0 | 24.86% |
| 原版 DraftMap Q128×K128 | 255 | 1710912 | 16.58% |
| 原版 DraftMap Q64×K64 | 168 | 0 | 24.85% |

Nsight 捕获五项各一个后续 mixed attention 主 kernel，10 次 replay。该表不包含首步 dense 概率捕获，不能据此声称首步也无 spill。Nsight duration 与非 profiler events 不混用。

PyTorch profile 中，概率版 Q64 后续主 kernel 约 5.99ms，Q128 约 7.10ms；准备约 4ms、前缀约 2.2ms 等开销不会随跳块数同比下降。各阶段有并行及父/子事件，不能简单把所有条目相加。原版每步另外有 pooling/评分/路由开销，但生成的 map 不同，混合主 kernel 耗时也会随选择变化。

只测一组真实算子输入、有限性检查；未重跑视频或评价稀疏后的画质。完整五项记录见 result.json，硬件摘要见 hardware_summary.json。
