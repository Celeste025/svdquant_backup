# DraftAttention2：块内最大 softmax 概率的命中率

在 `results/research/draftmap_hit_20261102/README.md` 既定设置上补充第三种真实块排序：对每个 video query，使用 BF16 QK GEMM、FP32 softmax，分母包含全部有效 prefix 和 video keys；按原生 Q128 ragged 物理布局，对 video 区域每个 tile 的所有有效 query/key 对取最大的 softmax 概率。原版 DraftMap 和真实评分分别在各个 head 的所有视频 block 中独立取同预算 top-k。命中率 = 交集 / k。

| 保留 | 真实块概率质量 | 真实块最大 softmax 概率 | 真实块最大 QK logit | 随机期望 |
|---:|---:|---:|---:|---:|
| 10% | 75.70% | 60.68% | 51.18% | 10% |
| 20% | 79.63% | 68.19% | 60.79% | 20% |
| 30% | 81.92% | 72.61% | 66.55% | 30% |
| 50% | 86.68% | 80.15% | 76.35% | 50% |

案例：赛车 p016，seed 20026，480×832，124 帧，20-step BF16 闭环轨迹；固定层 0/12/24/36/49、heads 0/8/16/24/32/40/48/55；原生官方 Q128 MeanMax0.2 代理，无 anchors。所有有效 video tiles 和 query 均纳入，统计是 20×5×8 的宏平均。

最大概率口径 top-10% 预算下，步 1→20 均值 53.11%→64.29%；层 1/13/25/37/50 均值 68.44%/62.66%/50.17%/52.19%/69.96%。另外三个预算的逐步、逐层数值见 summary.json。time Spearman rho 依预算分别为 1.000/0.989/0.994/0.997；依据 timestep 标签可交换假设的 50000 次排列检验，Holm 校正 p 均约 0.00016。层深单调 Spearman rho 分别为 0.1/-0.3/-0.3/-0.3，校正 p 均为 1。邻近 timestep 自相关和单 prompt 限制，p 值仅作为当前轨迹的探索性排序指标，不支持跨 prompt 的统计泛化；5 层存在非单调形态，不能把单调检验不显著解读为层深无影响。

复跑的原先概率质量和 QK 最大值口径，100 个 step×layer×8 head×4 预算×2 指标共 6400 项命中率与上次逐项完全一致。这证明上述输出统计一致，但不是全部底层 QKV 逐字节一致的证明。

图：draftmap_hit_by_step_and_layer.png/pdf；完整数值：summary.json；每 cell head 数据：per_cell.jsonl；生成代码：`scripts/research/analyze_draftmap_maxprob.py`、`scripts/research/summarize_draftmap_maxprob.py`。
