# DraftAttention2 的同一步选块命中率

## 实验口径

赛车 prompt 16，seed 20026，480×832、124 帧、20 步；使用 BF16 dense 闭环生成轨迹。在每步网络层 0/12/24/36/49（零基）采样 head 0/8/16/24/32/40/48/55。共 20×5×8 个 head 观测，只有一个 prompt 和一个 seed。真实 token 的 Q/K 在同一 DiT forward 中获取，前缀和图像都来自同一实际输入。

原始代理使用官方 SM120 Q128 ragged 分区、native Q/K pooling、0.8×mean-pool softmax + 0.2×max-pool softmax，以及每个 head 所有 video query/key tile 的全局稳定 top-k，不启用 anchors。四个预算独立从相同排序中取前 10/20/30/50%。原 Q/K 为 BF16，原生预处理按实现转换精度。

真实注意力参考：将 BF16 Q/K 按完全相同的空间重排；用 BF16 QK GEMM（累积 FP32、输出 BF16）及 FP32 softmax 计算所有 video query 对全部 prefix+video key 的注意力。每 tile 先累加各 query 对该 video key block 的概率，再对该 tile 的有效 query 求均值。边缘 padding 被排除。各 head 在视频 tile 上独立做全局 top-k。命中率是 |proxy top-k ∩ oracle top-k| / k；两个集合预算相同，因此也是同预算 precision。随机独立选择的期望命中率为预算比例。softmax 参考覆盖全部有效 video token，而非抽取的 query。

另计算同输入 tile 内 QK logit 的最大值 top-k：这是首步 exact-route 指标的 BF16 近似，对应另一个目标，但不是原 CUDA kernel 捕获的逐块 INT8 QK maximum，不能称为 bit-exact INT8 捕获。没有进行跨 timestep map 对比或复用质量实验。

## 汇总

| 视频块预算 | 真实 softmax 块质量 top-k 命中率 | QK tile 最大值 top-k 命中率 | 随机命中期望 |
|---:|---:|---:|---:|
| 10% | 75.70% | 51.18% | 10% |
| 20% | 79.63% | 60.79% | 20% |
| 30% | 81.92% | 66.55% | 30% |
| 50% | 86.68% | 76.35% | 50% |

上述是 20 步×5 层×8 heads 的宏平均，不是独立 prompt 平均。

## 变化与统计

真实 softmax 块质量口径，top-10%：第 1→20 step 的均值 71.4%→77.2%；各层（以 1 为起点）1/13/25/37/50 分别 82.4%/73.8%/64.9%/74.8%/82.6%。中间层在这五个预设点明显较低，表现为非单调形态。

先对层、head 聚合，再将 20 个 step 均值与 step 顺序做 Spearman 相关：10/20/30/50% 的 rho 为 0.958/0.830/0.686/0.346。条件于当前单条轨迹，在 step 标签可交换的假设下，50000 次全局置换并对同一 oracle 的 8 个假设做 Holm 校正，调整 p 分别约 0.00016/0.00016/0.00636/0.675。相邻步存在时间相关，标签可交换性未被证明，故这些 p 值只能作为当前轨迹上的探索性排序检验，不是跨 prompt 的统计显著性证据。

同样对 5 个 layer 均值与深度做 Spearman，10/20/30/50% 的 rho 为 0.3/-0.1/-0.1/-0.1；5! 精确排列且同样 Holm 校正后均为 1.0。不能据此称层位置没有影响：曲线呈非单调 U 形，Spearman 只检测单调趋势，且只有五个层。

QK max 口径的时间趋势更强，top-10% 第 1→20 step 为 43.3%→57.2%；五层均值为 62.7%/52.3%/37.5%/47.0%/56.3%。结果表明不同真实块目标的命中率不能混称。

## 文件

- `summary.json`：全部预算、各 step / layer 数值、排列检验与多重校正。
- `per_cell.jsonl`：100 个 step×layer 单元、8 heads 的独立数据。
- `draftmap_hit_by_step_and_layer.png/.pdf`：两种真实目标对 step 和层深的变化。
- `capture_manifest.json`：模型轨迹与预设采样。
- 源脚本：`scripts/research/analyze_draftmap_hit_rate.py` 和 `scripts/research/summarize_draftmap_hit_rate.py`。
