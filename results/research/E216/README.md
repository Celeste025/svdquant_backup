# E216 — 真实概率 top-tile 路由与 Q/K/V 双精度准备

## 路由图与复用需求

`true_top_route_examples.png/.pdf`：真实 FP32 softmax 概率 tile 均值排序，每 head 对全部视频 tile 全局取 top 10%、20%、30% 路由为 INT8，其余分别为 90%、80%、70% NVFP4，不跳块；红色为 INT8、浅蓝色为 NVFP4。图展示 step 0、layer 25 的 head 10、20、30（均从 0 编号），并非随机代表性样本。每一行是 Q block，每一列是 K/V block；格子的颜色是 QK 交互的精度，不是整个 Q 或 K 块的唯一精度。

20 step、56 heads、259×259 个视频 tile 的统计来自先前 BF16 dense 闭环 prompt 16 / seed 20026，480×832、124 帧的真实概率记录。逐 Q 行和 K/V 列检查：`both` 表示同时连接至少一个 INT8 tile 和一个 NVFP4 tile；`only` 表示视频区域只连接该精度。不含 prefix；本实验 prefix INT8，因此即使某个 Q 行的视频交互全是 NVFP4，Q 仍需要 INT8 表示来处理 prefix。

| top→INT8 / 其余 NVFP4 | Q 行两种都需要 | K/V 列两种都需要 | K/V 列仅 NVFP4 | K/V 列仅 INT8 |
|:---|---:|---:|---:|---:|
| 10% / 90% | 99.99% | 92.71% | 7.20% | 0.09% |
| 20% / 80% | 100.00% | 97.07% | 2.63% | 0.30% |
| 30% / 70% | 100.00% | 98.43% | 0.99% | 0.58% |

这些是 20 step 的每 step 全 head 块占比均值；K/V top10 的双精度占比在各 step 为 88.79%–95.66%，具有 head 和时间差异。真实 top30 索引在 step 0、5、10、19 上验证 56 head 均无重复。真实排序 `top_step_*.npz` 和轨迹定义见 `results/research/tile_probability_overlap_q64_20261103/`。

## 单输入准备阶段计时

真实 E187 race prompt 16、step 0、layer 25、480p 124 帧、56 heads、Q64，相同 BF16 QKV；GPU RTX PRO 5000 72GB。调用项目 `_prepare_h3_sm120_inputs` 的孤立 CUDA event：同时生成 INT8 + NVFP4 操作数 **4.532 ms**；只生成 INT8 **3.272 ms**；只生成 NVFP4 **3.363 ms**。每臂两轮正反顺序，每轮 3 次预热和 10 次测量；包括 packing 和 mean/max pooling。INT8 路径含 prefix-query INT8 准备，仅 NVFP4 路径不能单独启用 prefix INT8（原生内核限制）。因此这是不同准备分支的单独测时，不能把差值视作完整 attention 可实现的净节省，也不能简单相加。结果包含原生操作数及 scale 的输出内存：INT8 约 377 MB，NVFP4 约 212 MB（十进制，QKV 共计，含布局和 scale；不等同于峰值显存）。

此外 `result.json` 包含真实同一组输入的原版 pooling DraftMap 路由：E187 的 6 个 layer/step 捕获，及测时各次值。它是对照，非上面真实 softmax 选块图的数据。单输入速度易受内核、内存分配、并行/重叠和硬件影响；未实现按需跳过操作数转换，也未测量端到端优化收益或视频质量。

精度需求统计和图：`true_route_precision_incidence.csv`、`true_route_summary.json`。脚本：`scripts/research/analyze_e216_precision_coverage.py`、`scripts/research/plot_e216_true_routes.py`，日志 `results/logs/E216.log`。
