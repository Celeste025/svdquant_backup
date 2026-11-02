# E104 — H3 拍手样本逐 step DiT 输出误差形态

## 口径与完整性

沿用 E103 赛车实验的捕获与统计口径：E101 `clap0`，prompt 161（`A person is clapping`）、seed 38001610，576×1024、124 帧、20 步。复用同一条 BF16 自由生成轨迹保存的 20 个 DiT 输入和 BF16 输出，在这些完全相同的输入上运行 rank-32 SVDQuant 动态 activation tensor-wise scale 模型。误差定义为 `SVDQuant - BF16`，不是两条自由生成视频的逐像素差异。每步 video/audio NMSE 与 E101 的配对结果一致。

视频输出形状为 `[37,18,32,96]`，音频输出为 `[414,32]`。20 个原始文件共 329,544,800 bytes，位于 `/data1/models/svdquant-wjq/research/20261016/E104_h3_clap0_output_patterns/raw/`；逐文件路径、SHA256、checkpoint 和调用缓存 hash 在 [`manifest.json`](manifest.json)。[`summary.json`](summary.json) 与 [`derived_arrays.npz`](derived_arrays.npz) 保存全量约简统计和作图数组。20 个文件的 hash/字节数及 7 张图均已检查。GPU 5 上的捕获从 2026-10-16 18:04:51 +08:00 启动；推理环境缺少 matplotlib，数值分析完成后使用已有 PTQ 环境单独补做绘图，未重跑或改变捕获。

## 与赛车 E103 的单案例对照

| 指标 | 拍手 E104 | 赛车 E103 |
|---|---:|---:|
| 合并 20 步的 video NMSE | 2.295% | 8.195% |
| 合并 20 步的 audio NMSE | 1.721% | 4.003% |
| 逐步 video NMSE 峰值 | 7.039%，step 4 | 20.193%，step 4 |
| Top 1% 高误差 token 的误差能量占比，中位数 | 5.80% | 5.72% |
| 同批 token 的 BF16 信号能量占比，中位数 | 1.03% | 1.25% |
| 相邻高/宽/时间 token 的 log 误差 RMS 相关系数，中位数 | 0.846/0.829/0.821 | 0.809/0.828/0.662 |
| 相邻高/宽/时间 token 的中心化有符号误差向量余弦，中位数 | 0.340/0.283/0.346 | 0.485/0.520/0.467 |
| Top 10/96 输出 channel 的误差能量占比 | 15.81% | 14.97% |
| 相邻去噪步 channel 误差份额相关系数，中位数 | 0.941 | 0.942 |

拍手样本逐步 video NMSE 最低为 step 16 的 0.667%。误差最大 1% token 的能量份额逐步在 2.86%–10.44% 之间。空间位置随机打乱后的 log 误差 RMS 邻接相关系数中位数约为 0；有符号误差余弦打乱后也约为 0。因此该样本中误差同样不是空间独立白噪声。拍手与赛车的误差**幅度和有符号邻域对齐强度不同**，但两者都有误差热点和稳定的 channel 结构。只比较两个事先选定的案例，不能据此推断 prompt 类别规律，更不能断言“同 NMSE 下误差形态影响生成质量”。

## 图与复现

- [`step_nmse_and_token_concentration.png`](figures/step_nmse_and_token_concentration.png)：逐 step NMSE 与高误差 token 集中度。
- [`neighbor_error_structure.png`](figures/neighbor_error_structure.png)：误差幅值和有符号误差的邻域结构。
- [`channel_error_pattern.png`](figures/channel_error_pattern.png)：step × 输出 channel 的误差分布。
- [`step_by_latent_frame_nmse.png`](figures/step_by_latent_frame_nmse.png)：step × 潜视频帧 NMSE。
- [`spatial_token_maps.png`](figures/spatial_token_maps.png)：step 0/10/19 中间潜帧的 token RMS 热图；各面板使用独立 p99 色轴，不能用颜色跨步比较绝对幅值。
- [`signed_top_channel_spatial_map.png`](figures/signed_top_channel_spatial_map.png)：高误差 channel 的有符号空间图。
- [`error_magnitude_distribution.png`](figures/error_magnitude_distribution.png)：标准化绝对误差分布。
- [`video_frame_aligned_token_error.png`](figures/video_frame_aligned_token_error.png)：将 step 0/4/10/19 在潜视频时间 8/18/28 的 `18×32` token 误差热图，叠加到约对应的 BF16 解码帧 27/61/95 上。色值为每 token 的 96 维误差 RMS 除以该 step 全部 BF16 视频输出的 RMS；同图使用统一线性色轴，超过 p99 的最高约 1% token 截断显示。BF16 帧只是空间定位参考；VAE 会混合相邻时空 token，不能据此直接推断逐像素误差。精确口径、原始文件 hash 与逐步 NMSE 见 [`video_frame_aligned_token_error.json`](video_frame_aligned_token_error.json)。

入口：`scripts/research/launch_e104_h3_clap_output_patterns.sh`；日志：`results/logs/e104_h3_clap0_output_patterns.log`。分析和绘图复用 E103 的两个脚本。若要判断相同 NMSE 下的形态效应，仍需在闭环生成中进行能量匹配的误差置换干预。
