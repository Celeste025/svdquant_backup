# E214 — 提高 NVFP4 比例导致 attention 输出误差升高的频率

## 结论

在四条 BF16 轨迹的 224,000 个 prompt×step×layer×head 观测中，13,932 个（6.22%）至少存在一对 NVFP4 占比增加而视频 query 输出 NMSE 严格增加。现象并不均匀：第 49 层最突出，跨 4 个 prompt、20 step、56 head 的反转比例为 47.46%；第 1 层为 31.00%，但大量反转幅度很小。总体第 0 个 step 的标记率为 3.81%，第 19 个 step 为 10.38%；四个 prompt 各自首末 step 均上升。不能从四条相关轨迹推断总体发生概率。

在全 56 个 head 的 video 输出合并后，4,000 个 prompt×step×layer 单元中有 75 个（1.875%）出现至少一组反转；其中 70 个位于 layer 49，5 个位于 layer 45。head 级 6.22% 与整体层级 1.875% 的分母与含义不同，不能相加。

| prompt | 文本主题 | 反转 head 观测占比 |
|---:|:---|---:|
| 25 | 小猫麻将 | 7.99% |
| 7 | 火星宇航员 | 7.22% |
| 48 | 湖上划船 | 4.10% |
| 44 | 游泳 | 5.57% |

第 49 层的 head 10 和 22 在四个 prompt 的全部 20 个 step 均反转；head 11 在 79/80 个 prompt-step 单元反转，且三个比例间最大 NMSE 增幅中位数为 17.24 个百分点。具体例子：麻将、step 14、layer 49、head 11，NVFP4 10%、40%、90% 的 NMSE 分别为 20.29%、31.06%、50.32%。

严格标记包含极小差值：在全部 head 观测中，任意一对增幅大于 0.001、0.01、0.1、1.0 个 NMSE 百分点的比例依次为 1.49%、0.79%、0.39%、0.24%。这些只是描述幅度，主标记未设置阈值。归一化误差在参考输出能量很低的 head 上可能敏感，因此保存逐 head 参考能量和误差能量以供复核。

## 配置

Prompt 来自 `/home/wjq/workspace/178866172854036`，四个 ID 为历史 E202 同一组；统一采用 480×832、124 帧、20 step，而非 E202 不同片长的 768p 配置。种子来自 prompt 表。采集每个真实 BF16 dense 闭环轨迹上所有 50 个主层、所有 20 step 的 post-norm/RoPE QKV，三组比较复用同一 QKV、同一 BF16 dense attention 参考，实验输出不反馈至 BF16 轨迹。第 0 层也测量；refiner 不包含在范围内。

- DraftAttention2 原版 Q64×K64、ragged 布局、maxpool_weight=0.2、官方逐层 NVFP4 Q/K/V scale、INT8 prefix Q/KV。
- 三组全视频 tile 比例：INT8 10% + NVFP4 10% + 跳过 80%；INT8 10% + NVFP4 40% + 跳过 50%；INT8 10% + NVFP4 90% + 跳过 0%。实际每 head 的 INT8 tile 均为 6,708，NVFP4 tile 为 6,708 / 26,833 / 60,373，总 tile 数为 67,081。
- 每个 head 的 NMSE = 全部有效 video query 的输出误差平方和 / BF16 输出平方和；层级 NMSE 为合并 56 head 的误差平方和 / 合并参考能量。
- 严格反转标记：10→40、40→90、10→90 任意一对中，较高 NVFP4 占比的 NMSE 严格更大。逐项标记和所有三组 NMSE 见 `per_head_flags.csv`；逐 layer 的 head 计数、整体 NMSE 见 `step_layer_summary.csv`。

`reversal_by_step_layer.png/.pdf` 展示每个 prompt 的层×step 反转 head 比例；`reversal_by_layer_head.png/.pdf` 展示每个 layer/head 在 4×20 个 prompt-step 中的反转频率。完整数字、幅度统计及各 layer/head 汇总见 `summary.json`，每个 prompt 的 `cells.jsonl` 保存逐 head 误差和参考能量。

## 验证与局限

4×20×50 单元完整，无重复且结果有限；所有 head 参考能量正。基于既有真实 QKV 的四个 head 解码原版路由，验证三组 INT8 tile 完全一致且保留支持集逐级嵌套（`route_nesting_check.json`）；各组比例的整数分配也已核对。该抽查并非所有 4,000 个单元的逐 tile 路由审计。现象是整组策略的配对输出误差比较，不证明特定单块自身的因果作用，不能证明误差原因或视频质量；不同 prompt 的连续 step 相关，不能作为独立样本。第四个 prompt 44 根据 E202 历史四样本组确定。

入口 `scripts/research/run_e214_nvfp4_reversal.py`；启动 `scripts/research/launch_e214_nvfp4_reversal.sh`；统计 `scripts/research/summarize_e214_nvfp4_reversal.py`；计划 `research_state/06_experiments/decision_value_plan_E214.md`。

## 反转幅度图

`reversal_severity.png/.pdf` 统计每个 head 的 `max(0, NMSE40−NMSE10, NMSE90−NMSE40, NMSE90−NMSE10)`，单位为 NMSE 百分点。为突出幅度，采用 >0.1 个百分点的展示阈值：上排每个 layer×step 取 56 head 中的最大值，灰色表示没有 head 超过阈值，颜色采用对数刻度；下排显示超过阈值的 head 百分比（色标 0–25%）。这不是新的严格反转定义，而是展示阈值。

四个 prompt 的 224,000 个 head 观测中 866 个超过 0.1 个百分点，其中 676 个（78.1%）位于 layer 49，60 个在 layer 45，其余分布在 layer 32–47 的少数位置。各 step 的严重观测为 37–52 个，未显示此前严格反转比例那样的明显末期增长；该结论仅限本样本与阈值。逐层逐 step 最大幅度、对应 head/配置对及超阈值 head 数在 `reversal_severity_by_cell.csv`；统计在 `reversal_severity_summary.json`。绘图脚本为 `scripts/research/plot_e214_reversal_severity.py`。

## 整层误差相对变化图

`layer_nmse_relative_change.png/.pdf` 按 prompt 分行、NVFP4 10→40、40→90、10→90 三种比较分列，展示每个层和 step 的相对变化 `100 × (较高 NVFP4 的整层 NMSE / 较低 NVFP4 的整层 NMSE − 1)`；整层 NMSE 对 56 head 的误差能量和参考能量分别求和再相除。红色表示误差增加，蓝色表示下降，白色接近不变；统一 −100% 至 +100% 线性色标，本实验全部数值落在此范围内。横纵坐标均从 0 开始。相对变化比例与 NMSE 增加的绝对百分点不同；低基准误差可以产生较大相对比例。逐 cell 的三组原始 NMSE 和变化率见 `layer_nmse_relative_change.csv`，绘图代码为 `scripts/research/plot_e214_layer_relative_change.py`。

本地 `per_head_flags.csv` 为约 18 MB 的派生逐 head 表，按仓库备份策略不纳入 Git；较小的汇总 CSV、JSON 和图保留在仓库中。
