# E217 — 真实 softmax top-tile 竖条纹初探

## 数据与定义

复用 E187：MiniMax-H3 BF16 dense 轨迹，prompt_id 16，seed 20026，480×832，124 帧，第 25 层、step 0 和 10；所有索引从 0 开始，检查 head 10、20、30。Q64×K64 原生 ragged 布局，video grid 37×15×26，每个 latent 时间位置有 7 个空间 block，合计 259 个 query/key block。真实 FP32 softmax 包含 prefix keys；在 video×video tile 中每 head 以有效 query-key 概率均值全局选 top 10%。

对一个 key block，“列覆盖率”是 259 个 query block 中与它构成的 tile 被选入全局 top 10% 的比例。它是阈值选择统计，不是覆盖概率或因果重要性。`key_column_temporal_spatial.png` 将每个 key block 的列覆盖率按 latent time × 空间 block 画出；亮色对应原图的强竖列。空间 block 是 ragged 2D token 区域，latent time 不是直接的输出视频帧。

## 观察

- head 10 的最强列 step0 是 key block 146，92% query block 选中；step10 最强列是 block 249，95% 选中。head20 最强列两次都是 block 0，分别有 78% 和 86% 的 query block 选中。head30 的分布及跨 step 稳定性不同。
- 块级平均 K 与全视频 token 平均 Q 的点积 / sqrt(128)，与列覆盖率按 259 个 key block 的 Spearman 秩相关，head10 为 0.443（step0）/0.695（step10），head20 为 0.658/0.713，head30 为 0.791/0.620。方向匹配是相关线索，不证明相同 token 对所有 query 都高、也不证明原因；此指标基于 post-RoPE Q/K 均值。
- K block 均值向量的范数与列覆盖率相关性并不稳定，例如 head10 step0 为 −0.015、head20 step0 为 −0.487。最强 token 的 K 范数通常接近块内中位数；当前模型 Q/K 在 RoPE 前经过 norm，不能将强注意力直接归因于 K 范数更大。
- 对每个 head/step 挑列覆盖率最高的 block，以所有有效 video query 的注意力概率在该块所有 key token 上汇总：head10 step0 最高 1 个 token 占该块概率质量 33.3%，前 5 个占 77.7%；step10 前 1/5 个占 45.1%/94.8%。head20 step0 前 1/5 个为 16.0%/54.1%，step10 为 30.3%/73.3%；最强 token 在两步均是 video latent (t=0,y=0,x=0)。head30 的前 5 占 37.5%/81.0%。这些百分比是对**该选中 block、所有 query 聚合**的概率分布，不是每个 query 都关注同一个 token，也不能证明删掉该 token 的输出影响。
- head20 的时间 index mod 5 = 0 的所有空间 block 平均列覆盖率为 28.7%（step0）/30.7%（step10）；其他四个余数分别为 2.3%–6.2% / 3.8%–5.1%。MiniMax-H3 的位置构造明确使用 `_T_GROUP=5` 和 `_FRAME_PER_TOKEN=(1,4,4,4,4)`，这与观察到的周期吻合。仅凭相关不能判断是位置编码、video latent 内容/结构、共同上游处理或交互导致；不应称其已证实是 RoPE 造成。
- step0→10 相同 key block 列覆盖率的 Spearman 为 head10 0.311、head20 0.685、head30 0.092；top13 key block 集合交集为 3/13、9/13、1/13。不可把一个 head 的稳定性推广到其他 head。

对应强块和概率明细见 `summary.json`、`top_tokens.json`。`step0_features.npz` / `step10_features.npz` 为本地中间数据。捕获代码 `scripts/research/analyze_e217_attention_stripes.py`、token 汇总 `scripts/research/analyze_e217_top_tokens.py`、图 `scripts/research/plot_e217_attention_stripes.py`。

## 局限与下一步

当前只有一个 prompt、一层、两个 step、三个非随机挑选 head；只观察相关，不可断言强列是语义物体、背景、全局摘要或 attention sink。step0 在去噪开始时本身很嘈杂，进一步削弱直接语义解释。下一步应在多 prompt、多层按 latent 坐标检查时间组效应；通过调整仅时间位置、内容置换及 causal ablation 区分结构与内容；以 BF16 参考和完整 token 级别测量输出扰动与视频质量。删 token 必须重新归一化并衡量分母/输出变化，不能用 top-10% 命中率代替因果效果。
