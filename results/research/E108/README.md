# E108: H3 上一步真实误差的闭环修正（k=0.6）

本实验对赛车（prompt 16，seed 20026）和拍手（prompt 161，seed 38001610）各生成一条 576×1024、124 帧、20-step 视频。量化臂沿用 E101/E105 的 rank-32 SVDQuant checkpoint 和动态 NVFP4 scale。每一步在**修正臂自己的当前输入**上配对运行 BF16 与 SVDQuant DiT，得到原始误差 `e_t = Q_t - BF16_t`；从 step 1 开始返回 `Q_t - 0.6 e_(t-1)`，step 0 不修正。视频与音频分别维护各自的上一轮原始误差，不使用上一轮修正残差。

这是一个 oracle 实验：真实误差需要额外 BF16 DiT 前向才能知道，不能将此结果当作可直接部署的量化推理速度或质量。逐 step 的 DiT 输出 NMSE 衡量同输入的局部误差；最终解码 RGB 与另一次 BF16 生成的差异衡量自由生成轨迹偏离，不能单独代表感知画质。

预注册方案见 [E108 plan](../../../research_state/06_experiments/decision_value_plan_E108_h3_oracle_temporal_correction.md)。实现为 `scripts/research/run_e108_h3_oracle_temporal_correction.py`，可复现启动脚本为 `scripts/research/launch_e108_h3_oracle_temporal_correction.sh`，完整性与拼图脚本为 `scripts/research/summarize_e108_h3_oracle_temporal_correction.py`。大文件保存在 `/data1/models/svdquant-wjq/research/20261017/E108_h3_oracle_temporal_correction/`；本目录留存数值报告和预览图。

## 结果（2026-10-17）

两例均通过首步复现、20 步 video/audio 有限值、BF16 回舍误差、checkpoint SHA256 一致和 124 帧含音频视频的哈希校验。与 E106 使用完全相同的 checkpoint、prompt、seed 和生成配置。

| Case / modality | steps 1–19 原始输出 NMSE | 修正后输出 NMSE | 误差能量下降 | 改善步数 |
|---|---:|---:|---:|---:|
| 赛车 video | 9.252% | 5.483% | 40.7% | 19/19 |
| 赛车 audio | 4.528% | 1.867% | 58.8% | 19/19 |
| 拍手 video | 2.990% | 2.216% | 25.9% | 18/19 |
| 拍手 audio | 2.925% | 1.506% | 48.5% | 19/19 |

这些数字在**修正臂自身轨迹的相同 DiT 输入**上比较原始 `Q_t` 与 `Q_t-0.6e_(t-1)`。step 0 完全不变；拍手 video 第 2 步变差。不能把这里的原始 NMSE 曲线视为未修正自由生成臂的逐步误差曲线，因为两臂从 step 1 起输入轨迹不同。

| Case | 原 SVDQuant 相对 BF16 的解码 RGB NMSE | oracle 修正后 RGB NMSE | 下降 |
|---|---:|---:|---:|
| 赛车 | 0.5191 | 0.3125 | 39.8% |
| 拍手 | 0.5720 | 0.5256 | 8.1% |

赛车三联图中，修正版的道路/拱门时间进程比原量化版更接近 BF16，但不是逐帧复刻；拍手的修正版仍明显改变了镜头视角和手部位置，肉眼看不出与 BF16 同等幅度的改善。故结论仅为：真值 oracle 的时序修正在这两个例子中稳定降低局部输出误差，并使像素轨迹距离有所下降；它**没有证明**生成质量普遍提升，且真实 BF16 误差在部署时不可免费获得。

查看 [逐 step NMSE 曲线](oracle_prev_error_k06_step_nmse.png)、[赛车三联预览](race/three_arm_poster.jpg)、[拍手三联预览](clap0/three_arm_poster.jpg)；可机读的细节分别在 [赛车报告](race/three_arm_summary.json) 和 [拍手报告](clap0/three_arm_summary.json)。报告包含完整 MP4 绝对路径及 SHA256；三联视频无音频，单臂视频包含音频。
