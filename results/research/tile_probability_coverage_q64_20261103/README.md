# Q64 视频 tile 的 softmax 概率覆盖率

## 口径

MiniMax-H3 第 25 层、prompt 16、seed 20026、480×832、124 帧、20-step BF16 dense 轨迹，56 个 head。Q64×K64 原生 ragged 布局为 259×259 视频 tile。选块沿用前次实验：每个 head 在所有视频 tile 中按 tile 内有效 query-key 对的真实 softmax 概率**均值**排序取 top 10%、20%、30%；原版 DraftMap 使用 maxpool_weight 0.2 并在每个 step 重新选块。k 四舍五入。

在当前 step 上，对各个选中视频 tile 的真实 softmax 概率在全部有效视频 query/key 对上求和。主图 `coverage_video` 使用该 head 上全部视频 query→视频 key 的概率总量作分母；`coverage_all_keys` 的分母则包含视频 query→prefix key 的概率。两者分子均只有选中视频 tile，**不含 prefix**；如果 prefix 本身 dense 执行，这不是整条 attention 路径的总保留概率。先计算每个 head 的比值，再对 56 个 head 等权平均。所有 ragged query/key 元素按有效 token 计数，分母未对每个 tile 等权。

彩线选块取自各个历史 step 的真实概率均值图，右侧色标表示起始 step；曲线上每个点使用横轴当前 step 的真实概率评估。黑色虚线代表当前 step 原版 DraftMap。彩线的起点也一般低于 100%，因为选中的只是部分 tile。注意按均值 top-k 选块不一定最大化按总概率质量覆盖，特别是 ragged 边缘块大小不同。

| 视频概率覆盖率 | Top 10% | Top 20% | Top 30% |
|:---|---:|---:|---:|
| 同步真实均值选块，20-step 平均 | 79.50% | 89.80% | 94.33% |
| 原版 DraftMap 同 step，20-step 平均 | 70.79% | 82.77% | 88.93% |
| step 0 真实选块复用到 step 19 | 59.32% | 71.25% | 78.26% |
| step 19 原版 DraftMap | 76.99% | 87.21% | 92.23% |

`coverage.json` 包含两种分母的逐 head、逐 step 对结果以及各 head 的视频概率比例。采集代码 `scripts/research/capture_q64_tile_mass_steps.py`，绘图和汇总代码 `scripts/research/plot_q64_tile_mass_coverage.py`，日志 `results/logs/tile_mass_q64_20261103.log`。每个 step 重算的 top 30% 顺序与先前 Q64 记录完全一致；逐 query tile 检查视频与 prefix 概率之和等于有效 query 数；汇总时检查覆盖率在 0–100%。本实验仅是单 prompt、单层 BF16 dense 轨迹的诊断，不能直接推断量化视频质量。
