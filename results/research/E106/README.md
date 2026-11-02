# E106: 打乱真实 SVDQuant 完整 DiT 输出误差的位置

赛车（prompt 16/seed 20026）和拍手（prompt 161/seed 38001610）均完成 20-step、576×1024、124 帧的闭环生成。每步在当前臂自己的输入上配对计算 BF16 与 rank-32 SVDQuant 输出，取真实有符号误差 `e=Q-BF16`，在 video/audio 各自的整个输出张量内随机置换标量位置，再加回 BF16 输出。FP32 误差值多重集在加回前严格不变；最终为与原调度器一致而转回 BF16。`target_nmse` 与 `realized_nmse` 的最大相对 MSE 偏差小于 0.046%，而原始与置换误差同位置余弦均值接近 0。

| Case | 原 SVDQuant 相对 BF16 的解码 RGB NMSE | 真实误差打乱后 RGB NMSE | 变化 |
|---|---:|---:|---:|
| 赛车 | 0.5191 | 0.03737 | -92.8% |
| 拍手 | 0.5720 | 0.2972 | -48.0% |

上表是 H.264 解码后的自由生成轨迹像素距离，**不是**同输入的 DiT 输出 NMSE，也不是感知画质指标。逐步平均的 video 原始/打乱后输出 NMSE，赛车为 7.4343%/7.4345%，拍手为 2.4438%/2.4441%；误差能量基本没变化，却出现明显不同的生成轨迹。这是两个指定 case 上误差位置/结构有作用的证据；不能据此确定究竟是哪一种空间、时间或通道相关性在起作用，也不能外推至更大样本。

## 六臂最终视频距离：RGB NMSE 与 LPIPS-Alex

两项都与**同 case 的 BF16 视频**逐帧对齐比较，越低表示与该参照越近。RGB NMSE 是全部 124 帧、576×1024 原分辨率 RGB 像素的平方误差总和除以 BF16 像素能量总和；LPIPS 使用 TorchMetrics 预训练 AlexNet、FP32、`normalize=True`，对原分辨率 124 帧分别计算后取算术平均。指标计算不裁剪、不缩放、不包含音频。表中图像距离不是每步同输入 DiT 输出 NMSE，也不能单独代表视频主观质量。

| 处理方式 | 赛车 RGB NMSE | 赛车 LPIPS | 拍手 RGB NMSE | 拍手 LPIPS |
|---|---:|---:|---:|---:|
| BF16 | 0.00% | 0.0000 | 0.00% | 0.0000 |
| 原 SVDQuant | 51.91% | 0.5276 | 57.20% | 0.4803 |
| 真实误差换位置 | 3.74% | 0.1537 | 29.72% | 0.2967 |
| 恢复 >4σ 误差 | 44.54% | 0.4940 | 56.21% | 0.4780 |
| 等幅随机正负误差 | 10.19% | 0.2360 | 28.42% | 0.2853 |
| 高斯误差 | 7.99% | 0.2105 | 29.36% | 0.2894 |

六臂带指标图：[赛车](race/six_arm_metrics_poster.jpg)、[拍手](clap0/six_arm_metrics_poster.jpg)。图上每个缩略帧旁重复标注的是对应**整段 124 帧的聚合指标**，并非单帧指标。完整数值、逐帧 LPIPS、原视频 SHA256 和 LPIPS 权重 hash 见 [赛车指标 JSON](race/six_arm_nmse_lpips.json)、[拍手指标 JSON](clap0/six_arm_nmse_lpips.json)；计算脚本为 `scripts/research/evaluate_e106_six_arm_lpips.py`。

可查看 [赛车六臂预览](race/six_arm_poster.jpg)、[拍手六臂预览](clap0/six_arm_poster.jpg)、[赛车数值报告](race/six_arm_summary.json)、[拍手数值报告](clap0/six_arm_summary.json)。含音频的单臂视频与无音频六臂拼接视频路径及 SHA256 都在对应 JSON 中，大文件存放在 `/data1/models/svdquant-wjq/research/20261017/E106_h3_error_permutation/`。实现：`scripts/research/run_e106_h3_error_permutation.py`、`scripts/research/summarize_e106_h3_error_permutation.py`。实验方案见 `research_state/06_experiments/decision_value_plan_E106_h3_error_permutation.md`。
