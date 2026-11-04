# H3 softmax 概率均值 top-tile 重叠

## 设置

复用 E187 的 BF16 dense 闭环轨迹和 `attention_step_similarity` 的连续 step 捕获：prompt 16，seed 20026，480×832、124 帧、20 steps。现有捕获为第 0、25、49 层的 step 0 和 10，另有第 25 层 step 1、2、3。层和 step 均为从 0 开始的索引；这些是方便复用的既有捕获，不是随机选择。每个文件均校验 SHA-256。

视频按项目原有 ragged Q128×K128 布局分成 148×148 = 21,904 个 tile。逐 head 计算真实 FP32 QK 和 softmax：分母包含所有有效 prefix 和视频 key；tile 分数为有效 query-key 对的概率均值，边缘 padding 不参与均值。每个 head 在全部视频 tile 中全局独立取 top-k（10%、20%、30%，k 按四舍五入取整）。相似度定义为相同 head、相同物理 tile 坐标的交集 / k；下表是 56 个 head 的均值。两个独立均匀随机集合的期望重叠分别约为 10%、20%、30%。

## 同层跨 step

| 层 | step 对 | top 10% | top 20% | top 30% |
|---:|:---|---:|---:|---:|
| 0 | 0 → 10 | 84.6% | 88.9% | 90.9% |
| 25 | 0 → 1 | 72.8% | 77.5% | 80.8% |
| 25 | 1 → 2 | 84.7% | 87.2% | 88.9% |
| 25 | 2 → 3 | 87.4% | 89.6% | 90.8% |
| 25 | 0 → 3 | 61.9% | 68.5% | 73.2% |
| 25 | 0 → 10 | 53.5% | 60.7% | 66.4% |
| 25 | 1 → 10 | 60.7% | 66.8% | 71.2% |
| 49 | 0 → 10 | 62.2% | 68.6% | 74.2% |

## 同 step 跨层

| step | 层对 | top 10% | top 20% | top 30% |
|---:|:---|---:|---:|---:|
| 0 | 0 / 25 | 28.0% | 37.0% | 45.5% |
| 0 | 0 / 49 | 33.5% | 42.3% | 50.3% |
| 0 | 25 / 49 | 32.3% | 41.6% | 49.4% |
| 10 | 0 / 25 | 34.6% | 42.8% | 50.8% |
| 10 | 0 / 49 | 38.8% | 46.7% | 54.5% |
| 10 | 25 / 49 | 44.1% | 51.1% | 57.1% |

这是单 prompt、单 seed、三个既有层、BF16 dense 轨迹上的集合重叠诊断，不直接衡量缓存路由后的误差、视频质量或实际量化闭环下的稳定性。第 25 层 step 0→10 的 top 10% 各 head 重叠范围为 4.2%–82.7%，均值掩盖较大差异；逐 head 数据及全部 step 对见 `overlap.json`。跨层同编号 head 不保证语义对齐，跨层比较仅用于测量对应索引上的位置重叠。

复现：`CUDA_VISIBLE_DEVICES=5 /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/analyze_tile_probability_overlap.py`。源捕获为本地非版本化文件，路径和校验值见两个原始 manifest。

## 第 25 层完整 step 曲线

`forward_step_overlap.png/.pdf` 展示同一 BF16 轨迹中每个起始 step 相对后续所有 step 的 top-tile 重叠，颜色表示起始 step；每条线起点为 100%。横轴是从 0 开始的绝对 step，纵轴为 56 head 的交集 / k 均值。`forward_step_overlap.json` 包含全部 20×20 step 对和逐 head 数值。完整轨迹计算时每 step 捕获概率均值图，轨迹设置及分数口径与上文相同；与之前已有第 25 层 step 对的结果逐项核对。图仅反映第 25 层、单 prompt 的 dense 轨迹，不能直接推断其他层或量化闭环的视频质量。

复现：使用项目历史 H3 模型路径、prompt 路径及 GPU 环境运行 `scripts/research/capture_tile_probability_steps.py`，随后用安装 NumPy 和 Matplotlib 的 Python 运行 `scripts/research/plot_tile_probability_steps.py`。采集日志见 `results/logs/tile_steps_20261103.log`。

## 原版 DraftMap 对照

`forward_step_overlap_with_draft.png/.pdf` 在原图上叠加黑色虚线：每一步原版 DraftAttn2 的 DraftMap 与该步真实 softmax 均值 top-k 的重叠率，逐步重建 DraftMap。彩线为以历史 step 的真实 top-k 复用，比较对象同样是横轴当前 step 的真实 top-k。DraftMap 使用本地 native 路径和 `maxpool_weight=0.2`，使用相同 head、Q128×K128 ragged 布局及全局 top-k 预算。两条方法的 QKV 来自相同设定的 BF16 dense 重跑；top 10% 的 step 0、10 分别与 E199 交叉检查。逐 head 结果见 `draft_same_step_overlap.json`；捕获和绘图脚本分别为 `scripts/research/capture_draftmap_steps.py` 和 `scripts/research/plot_tile_draft_comparison.py`。该图比较选块重叠，并不直接衡量量化后的注意力误差或视频质量。
