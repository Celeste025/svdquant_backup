# Q64×K64 概率选块跨 step 重叠与 DraftMap 对照

复用前次实验设置：MiniMax-H3，第 25 层，prompt 16、seed 20026、480×832、124 帧，20-step BF16 dense 轨迹，56 heads，step/layer 从 0 计数。同一次重跑中同步采集真实概率图与原版 DraftMap，原版 `maxpool_weight=0.2`。Q64 原生 ragged 布局共有 259×259 个视频 tile；每个 head 全局独立选择 top 10%、20%、30% 的 tile，k 四舍五入。真实分数为 FP32 QK softmax 概率在每个 tile 的有效 query/key 元素上的均值；归一化分母包括 prefix，padding 被排除。

图的彩线是某起始 step 的真实 top-k 与横轴当前 step 的真实 top-k 的交集 / k，黑色虚线是当前 step 的原版 DraftMap top-k 与当前真实 top-k 的交集 / k，均为 56 head 平均。`overlap.json` 包含逐 head 与每一对 step 的完整统计。

| 指标 | top 10% | top 20% | top 30% |
|:---|---:|---:|---:|
| 原版 DraftMap 同 step 重叠，20-step 均值 | 69.69% | 74.68% | 78.34% |
| 真实选块从 step 0 复用到 step 19 | 44.86% | 52.80% | 58.72% |
| 同一设置 Q128，step 0 复用到 step 19 | 48.31% | 55.62% | 61.58% |

使用 step 0 的原始 QKV，独立重新计算第 0 head 的真实概率及 DraftMap 并检查前三档 top 集合：真实概率三个预算全部一致；DraftMap 在 top 10%、30% 完全一致，top 20% 有一个 tile 的差异，可能由不同 head 批次下的近似并列分数引起。步对矩阵通过对称性和对角线为 100% 的检查。结果仅适用于该单 prompt、单 seed、单层 dense 轨迹；top 集合重叠不直接衡量视频质量。

采集脚本 `scripts/research/capture_q64_tile_steps.py`；绘图脚本 `scripts/research/plot_q64_tile_steps.py`；日志 `results/logs/tile_q64_20261103.log`。逐 step top 索引以本地 NPZ 保留；JSON 和图可直接审阅。
