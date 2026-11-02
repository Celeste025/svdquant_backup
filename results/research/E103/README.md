# E103 — H3 赛车样本逐 step DiT 输出误差形态

## 口径和数据

赛车 prompt 16、seed 20026；576×1024、124 帧、20 去噪步。复用 E101 保存的同一条 BF16 轨迹调用与标准 DiT 输出，将这些输入逐步重放给同一 rank-32 SVDQuant checkpoint 的**动态 activation 全局 scale**运行时。比较的是每个调用的完整 DiT 输出，`error = SVDQuant - BF16`；不是两条自由生成视频之间的差异。每一步的 video/audio NMSE 都与 E101 配对结果一致。

视频输出每步有 `37×18×32` 个 token，每 token 96 维，按时间、高、宽顺序排列；音频输出为 `414×32`。20 个文件分别保存 BF16、SVDQuant 的 BF16 输出及 FP32 有符号误差，合计 329,544,800 bytes，位于 `/data1/models/svdquant-wjq/research/20261016/E103_h3_race_output_patterns/raw/`。文件路径、SHA256、模型与 BF16 调用缓存的 hash 见 [`manifest.json`](manifest.json)。[`summary.json`](summary.json) 保存逐步统计；[`derived_arrays.npz`](derived_arrays.npz) 保存 channel、潜帧及 token 能量数组，可重画图。

## 初步观察

| 维度 | 赛车这一个样本的观察 |
|---|---|
| 总误差 | 20 步合并能量后的 video NMSE 为 **8.19%**，audio 为 **4.00%**。Video 第 4 步最高 **20.19%**，第 17 步最低 **1.09%**。合并能量 NMSE 不等于逐步 NMSE 的算术平均。 |
| token 集中度 | 每一步误差最大的 1% video token 承担误差能量的 **2.79%–11.72%**，中位数 **5.72%**；同一批 token 的 BF16 输出能量份额中位数只有 **1.25%**。因此不只是这些 token 的标准输出数值更大。 |
| 邻域结构 | 对每潜帧消去整体幅值后，相邻高/宽/时间 token 的 log 误差 RMS 相关系数中位数为 **0.81/0.83/0.66**；BF16 输出幅值对应为 **0.42/0.45/0.37**，帧内随机打乱位置约为 0。有符号 96 维误差向量在相邻高/宽/时间位置的余弦中位数为 **0.49/0.52/0.47**，帧内打乱后约为 0。误差不是独立白噪声。 |
| channel 结构 | 累积误差能量最高的 10/96 个输出 channel 占 **14.97%**，这些 channel 占 BF16 输出能量 **12.06%**，属于中等不均匀，而非极少数 channel 完全主导。channel 80 的单 channel NMSE 为 **12.03%**。相邻去噪步的 channel 误差份额向量相关系数中位数 **0.94**，说明高误差 channel 较稳定。 |
| patch 子位置 | 96 维由 24 个潜变量 channel × 2×2 子位置组成。四个子位置各占约 25% 误差能量，无明显某一个子位置失衡；连续 4 个高误差 channel 更像是同一潜变量 channel 的四个 patch 子位置。 |

误差热点随去噪步变化，但相邻 step 的 token log 误差能量图相关系数中位数约 **0.75**。逐潜帧 NMSE 与空间切片见图；这些数值是单 prompt 的描述统计，不是跨 prompt 显著性检验。

## 图

- [`step_nmse_and_token_concentration.png`](figures/step_nmse_and_token_concentration.png)：逐去噪步误差和 token 集中度。
- [`channel_error_pattern.png`](figures/channel_error_pattern.png)：step × 输出 channel 的 NMSE 与 channel 误差/信号能量份额比。
- [`neighbor_error_structure.png`](figures/neighbor_error_structure.png)：空间和时间邻接结构；虚线是 BF16 输出幅值对照。
- [`step_by_latent_frame_nmse.png`](figures/step_by_latent_frame_nmse.png)：去噪 step × 视频潜帧的 NMSE。
- [`spatial_token_maps.png`](figures/spatial_token_maps.png)：第 0、10、19 步中间潜帧的误差 RMS 与 BF16 输出 RMS。每张面板有独立 p99 色轴，适合看形状，不可仅凭颜色直接比较跨步绝对幅值。
- [`signed_top_channel_spatial_map.png`](figures/signed_top_channel_spatial_map.png)：高误差 channel 80 的有符号误差空间切片；各面板独立对称 p99 色轴。
- [`error_magnitude_distribution.png`](figures/error_magnitude_distribution.png)：标准化绝对误差分布，绘图每个选定 step 抽样 15 万元素；JSON 的分位数取自全量元素。
- [`video_frame_aligned_token_error.png`](figures/video_frame_aligned_token_error.png)：将 step 0/4/10/19 在潜视频时间 8/18/28 的 `18×32` token 误差热图，叠加到约对应的 BF16 解码帧 27/61/95 上。色值为每 token 的 96 维误差 RMS 除以该 step 全部 BF16 视频输出的 RMS；同图使用统一线性色轴，超过 p99 的最高约 1% token 截断显示。这里的 BF16 帧只是空间定位参考；VAE 会混合相邻时空 token，不能将热图理解成逐像素误差或严格的 latent-to-frame 归因。精确口径、原始文件 hash 与逐步 NMSE 见 [`video_frame_aligned_token_error.json`](video_frame_aligned_token_error.json)。

## 解释边界与下一步

现在证实的是**单次完整 DiT 输出的误差有结构**，不是“结构在同 NMSE 下显著改变生成质量”。后一个问题需要在相同 BF16 轨迹或闭环采样中，把误差的每步能量严格匹配后，比较原始结构、空间/时间打乱及 channel 打乱的干预，再用相同 seed 的视频和盲评/质量指标验证。打乱必须保留关键边际分布，避免把总误差幅度或 timestep 能量当作 pattern 效应。

复现入口：`scripts/research/launch_e103_h3_race_output_patterns.sh`、`scripts/research/analyze_h3_output_error_patterns.py`、`scripts/research/render_h3_output_error_patterns.py`。
