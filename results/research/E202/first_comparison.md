# E202：新增首步 softmax 概率 map 对照

旧三组的画质来自 E160/E161/E162 原有结果，耗时为各方案此前在小猫麻将 p025 的单次复测；新增方案只生成首步概率 map 复用，采用两遍 INT8 QK 的实验版算子。各视频与既有 BF16 参考对齐。

| 方案 | 四视频平均 LPIPS ↓ | 四视频平均 PSNR ↑ | p025 平均每 step（秒） |
|---|---:|---:|---:|
| 原版首步 QK map 复用 | 0.3020 | 17.78 | 15.46 |
| 原版每 5 步刷新 QK map | 0.2923 | 17.87 | 15.57 |
| Baseline 全 step 全层 DraftMap | 0.3028 | 17.87 | 15.31 |
| 新增首步 softmax 概率 map 复用 | 0.2679 | 18.37 | 16.13 |

新增方案的平均每 step：5 秒视频 15.99 秒；10 秒视频 38.42 秒。旧表的时间仅为 p025，并非四视频均值。

| 案例 | 帧数 | LPIPS | PSNR | DiT 总耗时 | 视频 |
|---:|---:|---:|---:|---:|---|
| 25 | 124 | 0.2626 | 16.41 | 322.68 秒 | /home/wjq/workspace/svdquant-exp/results/local/E202/first/p025/first/p025_bf16_1344x768_124f_20steps.mp4 |
| 48 | 124 | 0.2796 | 18.84 | 317.11 秒 | /home/wjq/workspace/svdquant-exp/results/local/E202/first/p048/first/p048_bf16_1344x768_124f_20steps.mp4 |
| 44 | 243 | 0.2134 | 21.69 | 764.82 秒 | /home/wjq/workspace/svdquant-exp/results/local/E202/first/p044/first/p044_bf16_1344x768_243f_20steps.mp4 |
| 7 | 243 | 0.3160 | 16.54 | 771.98 秒 | /home/wjq/workspace/svdquant-exp/results/local/E202/first/p007/first/p007_bf16_1344x768_243f_20steps.mp4 |

旧数据未覆盖；误启动的 E202 周期方案不用于比较。结果只针对四个视频，不构成统计显著性结论。
