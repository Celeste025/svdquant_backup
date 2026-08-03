# SVDQuant 扩展至 Wan2.1-1.3B：完整实验汇总

本文汇总本项目中已经实际执行的主要实验。除特别注明外，Wan 实验统一使用：

- 模型：Wan2.1-T2V-1.3B-Diffusers
- prompt：`An astronaut feeding ducks on a sunny afternoon, reflection from the water.`
- seed：44
- 832×480，81 帧，50 个去噪步，flow shift=3，16 FPS
- 正常质量基准 CFG=6
- W4A4：group=64、rank=32，300 个 Linear fake quant
- 激活：动态 group-64 INT4；FFN-down 在 shift 后使用 unsigned INT4

NMSE 均定义为 `mean((quant-bf16)^2) / mean(bf16^2)`。

---

## 实验 1：FLUX.1-schnell BF16 与官方真 W4A4 的质量、速度和显存

### 做了什么

使用同一 prompt、相同 prompt embeddings、seed 42/43/44、1024×1024、4 steps，
比较 BF16 FLUX.1-schnell 与官方 `mit-han-lab/svdq-int4-flux.1-schnell`。
W4A4 路径使用 Nunchaku 高效 kernel，不是 fake quant。

### 观察

| 指标 | BF16 | W4A4 |
|---|---:|---:|
| 4-step transformer 平均耗时 | 2.487 s | 0.839 s |
| 每步 | 621.7 ms | 209.7 ms |
| 加速比 | 1× | 2.97× |
| resident GPU memory | 23,858 MiB | 7,368 MiB |

三组图像主内容和视觉质量接近，W4A4 没有出现 Wan 那种明显噪声和边缘闪动。

### 数据和样例

- BF16 benchmark：[benchmark_bf16.json](flux_comparison/benchmark_bf16.json)
- W4A4 benchmark：[benchmark_w4a4.json](flux_comparison/benchmark_w4a4.json)
- BF16 图像：[seed42](flux_comparison/bf16_seed42.png)、
  [seed43](flux_comparison/bf16_seed43.png)、[seed44](flux_comparison/bf16_seed44.png)
- W4A4 图像：[seed42](flux_comparison/w4a4_seed42.png)、
  [seed43](flux_comparison/w4a4_seed43.png)、[seed44](flux_comparison/w4a4_seed44.png)
- 完整配置：[BF16 metadata](flux_comparison/bf16_metadata.json)、
  [W4A4 metadata](flux_comparison/w4a4_metadata.json)

### 结论

SVDQuant 在 FLUX-schnell 上确实是可运行的真 W4A4，24 GB 4090 可运行，且 transformer
部分约 3 倍加速、resident 显存显著下降。最初的 FLUX 结果不是“只做了权重量化”
或 fake quant。

---

## 实验 2：Wan BF16 对齐与初始 fake W4A4

### 做了什么

打印并检查 Wan 模型结构，参考 DVDQuant 的加载、scheduler、negative prompt 和随机数
设置。最初 seed42 基准与 DVDQuant 不一致后，修正为：

- 完整 negative prompt
- UniPC scheduler，flow shift=3
- CUDA generator 和全局随机种子对齐
- 后续统一 seed44

然后将原版 SVDQuant 思路移植到 Wan：SmoothQuant scale、activation shift、
W4 residual、rank-32 low-rank compensation、动态 A4。

### 观察

- BF16 对齐修正后，生成内容与 DVDQuant 基准一致。
- 初始 W4A4 可以完整生成 81 帧视频，但每帧纹理、局部结构和边缘都有明显下降，
  同时存在时序闪动。
- 问题不只是帧间一致性：单帧本身也明显受损。

### 数据和视频

- 对齐 BF16：[bf16_seed44.mp4](wan_svdquant_fake/bf16_seed44.mp4)
- 初始 W4A4：[quant_seed44.mp4](wan_svdquant_fake/quant_seed44.mp4)
- BF16 配置：[bf16_metadata.json](wan_svdquant_fake/bf16_metadata.json)
- W4A4 配置：[quant_metadata.json](wan_svdquant_fake/quant_metadata.json)
- 初始校准 checkpoint：
  [svdquant_seed44_calibrated.pt](wan_svdquant_fake/svdquant_seed44_calibrated.pt)

### 结论

实现和生成链路已经跑通，但 Wan 对 W4A4 的容忍度远低于 FLUX。后续实验必须区分
“局部量化误差较大”和“模型会放大量化扰动”。

---

## 实验 3：补齐原版 SVDQuant 质量技术与扩大校准集

### 做了什么

把原版方法中与质量相关的步骤加入 Wan：

1. activation-aware smooth scale；
2. FFN-down activation shift；
3. 在 smooth 后权重域做 low-rank decomposition；
4. 对 residual weight 做 group-wise W4；
5. rank-32 up/down BF16 补偿；
6. 交替/网格搜索 smooth 与量化参数。

之后将校准数据扩大为：

- 8 个 prompt；
- 每个 prompt 10 个 timestep；
- 每个量化组至少 2048 tokens；
- 覆盖全部 210 个 projection group、300 个 Linear；
- 分 8 个 shard 并行收集和校准。

### 观察

- 大校准集版本主观质量比早期小校准集略有改善。
- 边缘异常闪动和少量噪声纹理仍存在，并没有因增加普通校准量而消失。
- 大校准 checkpoint 的联合线性输出平均 NMSE 为 0.911%，说明单层拟合已经不差。
- 大校准和小校准的平均 NMSE不能直接视作同一测试集上的优劣：两者采样分布不同。

### 数据和视频

- 校准 manifest：[manifest.json](wan_svdquant_calib_large/manifest.json)
- 最终 checkpoint：
  [svdquant_large_calibrated.pt](wan_svdquant_calib_large/svdquant_large_calibrated.pt)
- checkpoint 摘要：
  [svdquant_large_calibrated.json](wan_svdquant_calib_large/svdquant_large_calibrated.json)
- 8 个校准 shard：[calibrated_shards](wan_svdquant_calib_large/calibrated_shards)
- 大校准生成视频：
  [quant_seed44.mp4](wan_svdquant_fake_large_calib/quant_seed44.mp4)
- 生成配置：
  [quant_metadata.json](wan_svdquant_fake_large_calib/quant_metadata.json)

### 结论

校准数据不足确实是质量因素之一，但不是主要瓶颈。普通局部 MSE 校准继续扩量的边际
收益有限，因为它没有约束后续 block、CFG 差分或最终 latent。

---

## 实验 4：CFG、W4A16 与敏感层 BF16 消融

### 做了什么

在相同 seed44 下测试：

- W4A4，CFG=1/3/6；
- BF16，CFG=1/3；
- W4A16：保留 smooth scale，权重仍 W4+rank32，activation 不取整；
- 重新针对 W4A16 校准；
- 将一批经验敏感层恢复 BF16。

### 观察

对 BF16 CFG=6 视频的像素指标如下：

| 配置 | RGB PSNR | Temporal residual MAE | Edge temporal residual MAE |
|---|---:|---:|---:|
| W4A4 CFG6 | 13.87 dB | 11.36 | 30.39 |
| W4A16 CFG6 | 13.29 dB | 10.55 | 30.31 |
| W4A16 重新校准 | 13.42 dB | 11.09 | 31.18 |
| 敏感层 BF16 | 13.07 dB | 11.89 | 31.66 |

CFG=1/3 的量化结果在像素 MSE 上更接近对应数据，但 BF16 本身在 CFG=1/3 时基础
生成质量很差。因此这些较小 MSE不能用于证明低 CFG 修复了量化。

W4A16 没有稳定优于 W4A4，说明问题不能简单归因于 A4；经验式恢复一批层 BF16
也没有改善，说明当时选择的敏感层集合不正确或误差由系统交互决定。

### 数据和视频

- 汇总指标：[wan_ablation_metrics.json](wan_ablation_metrics.json)
- W4A16 重校准指标：
  [wan_ablation_w4a16_recalibrated_metrics.json](wan_ablation_w4a16_recalibrated_metrics.json)
- W4A16 视频：[quant_seed44.mp4](wan_ablation_w4a16/quant_seed44.mp4)
- W4A16 重校准视频：
  [quant_seed44.mp4](wan_ablation_w4a16_recalibrated/quant_seed44.mp4)
- 敏感层 BF16：
  [quant_seed44.mp4](wan_ablation_sensitive_bf16/quant_seed44.mp4)
- CFG1/3 BF16 与量化样例：
  [BF16 CFG1](wan_ablation_bf16_cfg1/bf16_seed44.mp4)、
  [W4A4 CFG1](wan_ablation_cfg1/quant_seed44.mp4)、
  [BF16 CFG3](wan_ablation_bf16_cfg3/bf16_seed44.mp4)、
  [W4A4 CFG3](wan_ablation_cfg3/quant_seed44.mp4)

### 结论

这组实验排除了“只要保留 activation BF16 就能修复”的简单解释。CFG=1/3 实验由于
基准质量不足，只能用于诊断数值误差，不能作为有效画质方案。

---

## 实验 5：W8A4 与 W4A4 rank64

### 做了什么

保持 group=64 和大校准集，分别测试：

- W8A4 rank32；
- W4A4 rank64。

### 观察

| 配置 | 校准线性输出平均 NMSE | RGB PSNR | Edge temporal MAE |
|---|---:|---:|---:|
| W4A4 rank32 | 0.911% | 13.87 dB | 30.39 |
| W8A4 rank32 | 0.481% | 15.70 dB | 28.80 |
| W4A4 rank64 | 0.765% | 11.84 dB | 34.84 |

W8A4 明显改善空间质量，说明 W4 权重误差仍是重要因素。rank 从 32 提升到 64 虽降低
局部校准 NMSE，却使最终视频更差，表明低秩拟合指标与端到端生成质量并不单调对应。

### 数据和视频

- 指标：[wan_ablation_w8a4_r32_w4a4_r64_metrics.json](wan_ablation_w8a4_r32_w4a4_r64_metrics.json)
- W8A4：[quant_seed44.mp4](wan_ablation_w8a4_r32/quant_seed44.mp4)
- W4A4 rank64：[quant_seed44.mp4](wan_ablation_w4a4_r64/quant_seed44.mp4)
- 对应 metadata：
  [W8A4](wan_ablation_w8a4_r32/quant_metadata.json)、
  [rank64](wan_ablation_w4a4_r64/quant_metadata.json)

### 结论

权重 4 bit 是 Wan 质量下降的一部分来源，但单纯增加 SVD rank 不是可靠修复方案。
校准目标需要考虑网络传播和 CFG，而不能只优化单层输出。

---

## 实验 6：官方 FLUX checkpoint 与复现链路复核

### 做了什么

检查官方 FLUX-schnell 模型结构和 W4A4 checkpoint：

- 约 11.89B transformer 参数；
- 19 个 dual-stream block、38 个 single-stream block；
- hidden size 3072；
- checkpoint 含 qweight、group scale、smooth/smooth_orig 和 rank-32 up/down。

尝试通过 DeepCompressor 从 BF16 完整复刻官方 PTQ。单卡/batch1 仍无法在 24 GB 内完成
完整校准；8 卡 prompt shard 收集产生约 2.7 GB 中间数据，但完整 PTQ 没有完成。因此
该路径不能声称已经从 BF16 完整复现 checkpoint。

随后直接加载官方 FLUX.1-dev W4A4 checkpoint，完成 1024×1024、50-step、CFG3.5、
seed42/43/44 真量化生成。

### 观察

- FLUX.1-dev W4A4 三张图均成功生成。
- seed42/43/44 分别约 12.02/10.79/10.86 秒。
- 峰值 allocated 11.78 GiB，reserved 13.56 GiB，4090 24 GB 可运行。
- checkpoint unpack/repack 校验完全一致，确认分析的是官方真实布局和 W4A4 数据。

### 数据和图像

- 模型结构审计：[model_audit.json](flux_official_reproduction/model_audit.json)
- FLUX-dev metadata：
  [official_w4a4_metadata.json](flux_dev_official_w4a4/official_w4a4_metadata.json)
- FLUX-dev 图像：[seed42](flux_dev_official_w4a4/official_w4a4_seed42.png)、
  [seed43](flux_dev_official_w4a4/official_w4a4_seed43.png)、
  [seed44](flux_dev_official_w4a4/official_w4a4_seed44.png)
- DeepCompressor 尝试日志：
  [flux_deepcompressor_reproduction](flux_deepcompressor_reproduction)

### 结论

官方 FLUX checkpoint 和 Nunchaku W4A4 路径可信；但“从 BF16 在当前 24 GB 单卡上完整
重新校准出官方 checkpoint”没有完成。后续 FLUX/Wan 数值对比使用官方 FLUX 真量化，
Wan 则是算法等价 fake quant，这一点必须保留为实验边界。

---

## 实验 7：原始及 smooth 后 activation outlier/A4 误差

### 做了什么

第一轮比较实际 smooth 后的激活：

`x_scaled = (x + optional_shift) / smooth`

FLUX 覆盖 4 个 block、8 类 projection、50 steps；Wan 使用大校准集中全部 210 个
projection group。之后做严格配对实验：两者均选浅/中/深三个 block、四类 Linear，
在 step 1/6/…/50 每次采样 2048 tokens。

### 观察

第一轮广覆盖结果：

| 模型 | A4 NMSE 中位数 | max/RMS 中位数 |
|---|---:|---:|
| FLUX-dev | 2.635% | 22.77 |
| Wan | 1.150% | 10.33 |

严格配对结果：

| 模型 | A4 NMSE 中位数 | max/RMS 中位数 |
|---|---:|---:|
| FLUX-dev | 1.428% | 11.30 |
| Wan | 1.278% | 7.47 |

FLUX 深层 FFN-down 个别测点 max/RMS 可超过 200，但 A4 NMSE仍只有约 0.7%–1%。
因此极少数 absmax outlier 不足以预测整体 A4 或最终画质。

### 数据和图表

- 第一轮汇总：[summary.json](activation_error_comparison/summary.json)
- 第一轮图：[scaled_activation_error_comparison.png](activation_error_comparison/scaled_activation_error_comparison.png)
- 严格配对原始数据：
  [FLUX](quant_error_diagnosis/postsmooth_activation/flux_metrics.json)、
  [Wan](quant_error_diagnosis/postsmooth_activation/wan_metrics.json)
- 严格配对图：
  [postsmooth_activation_a4_error.png](quant_error_diagnosis/postsmooth_activation_a4_error.png)

### 结论

“Wan 因 smooth 后 activation outlier 更严重而难量化”不成立。Wan 的实际 A4 局部
误差没有比 FLUX 大，真正差异应在误差方向、网络敏感性、CFG 或多步传播。

---

## 实验 8：timestep-dependent activation outlier

### 做了什么

参考 SVQ-GPTQ Fig.1：

1. 对每个 timestep、每个 channel，计算跨全部 token 的 activation absmax；
2. 取 `log2(absmax)` 形成 channel×timestep surface；
3. 先比较约 2/3 深度的 FFN-down；
4. 再扩展到浅/中/深 × attention Q/out、FFN up/down 共 12 类层；
5. 额外计算相邻 timestep top-outlier channel 集合 Jaccard，相比单纯 p99 上包络更适合
   衡量 outlier channel 身份是否随时间变化。

### 观察

单层结果：

| 指标 | FLUX | Wan |
|---|---:|---:|
| 每 channel timestep range 中位数（log2） | 1.286 | 0.758 |
| step p99 range（log2） | 1.137 | 0.523 |
| 相邻 top1% channel Jaccard | 0.752 | 0.903 |

在该层中 Wan 的 outlier channel 反而更稳定。多层结果不是完全一致：Wan 某些
attention-out 层变化较大，FLUX 某些层也同样明显；没有观察到“Wan 所有层都具有更强
timestep outlier”的系统性证据。

### 数据和图表

- 单层数据：[timestep_outlier_metrics.json](timestep_outliers/timestep_outlier_metrics.json)
- 单层 Fig.1 风格图：
  [wan_flux_timestep_outlier_fig1.png](timestep_outliers/wan_flux_timestep_outlier_fig1.png)
- 多层数值：[multilayer_metrics.json](timestep_outliers_multilayer/multilayer_metrics.json)
- 全部 12 组 Fig.1 风格 PNG：
  [fig1_style_similarity](timestep_outliers_multilayer/fig1_style_similarity)
- 汇总 surface：
  [FLUX](timestep_outliers_multilayer/flux_multilayer_surfaces.png)、
  [Wan](timestep_outliers_multilayer/wan_multilayer_surfaces.png)
- 原始 surface tensor：
  [FLUX](timestep_outliers_multilayer/flux_multilayer_surfaces.pt)、
  [Wan](timestep_outliers_multilayer/wan_multilayer_surfaces.pt)

### 结论

当前数据不支持用“Wan timestep outlier 更严重”单独解释质量下降。p99 curve 只表示
每步跨 channel 的第 99 百分位上包络，不追踪同一个 channel，解释力有限；相邻
timestep channel 相似度更合理，但也没有显示 Wan 一致更差。

---

## 实验 9：每 5 步捕获 transformer output 与 scheduler 后 latent

### 做了什么

在 step 1/6/11/…/46/50 捕获：

- teacher-forced model output：BF16 和量化使用同一 BF16 latent 输入；
- rollout model output：各自使用自己的历史 latent；
- scheduler 更新后的 latent。

FLUX 使用官方真 W4A4；Wan 使用大校准 fake W4A4。

### 观察

关键 NMSE（%）：

| 模型/step | Teacher-forced output | Rollout output | Post-step latent |
|---|---:|---:|---:|
| FLUX step1 | 0.097 | 0.134 | 0.00012 |
| Wan step1 | 5.071 | 5.071 | 0.00108 |
| FLUX step26 | 0.357 | 5.079 | 0.315 |
| Wan step26 | 6.825 | 19.824 | 2.823 |
| FLUX step50 | 4.171 | 29.435 | 7.196 |
| Wan step50 | 146.302 | 136.532 | 26.929 |

第一步 scheduler update 很小，所以 latent NMSE 看起来都很低；这不能否认 Wan 的
model output 已经在第一步出现 5.07% NMSE。随着 rollout 进行，误差累积并进入 latent。

### 数据和图表

- 汇总数据：[metrics.json](rollout_latent_error/metrics.json)
- 图：[wan_flux_output_latent_error.png](rollout_latent_error/wan_flux_output_latent_error.png)
- BF16/quant/teacher-forced 原始 tensor：
  [rollout_latent_error](rollout_latent_error)

### 结论

Wan 的问题从第一步模型输出就存在，不是纯粹由长时序累积产生；多步传播会进一步
放大。逐帧质量下降因此可以由同一个首步空间预测误差解释，视频时序只是继续传播并
把结构化误差表现为闪动。

---

## 实验 10：权重重构、逐 block 传播、输出头与 CFG 的最终定位

### 10.1 W4 + rank32 权重重构

#### 做了什么

选择浅/中/深三个 block × attention Q/QKV、attention out、FFN up/down，共 12 个层。
严格比较 smooth 后权重：

`W_s = W · diag(s)`

与：

`Q4(W_s - U D) + U D`

FLUX 官方 checkpoint 的 packed qweight、wscale、smooth 和 low-rank 参数都按
Nunchaku layout 逆变换；repack 后逐元素一致。

#### 观察

| 模型 | 12 层 NMSE 中位数 | 最大值 |
|---|---:|---:|
| FLUX | 1.770% | 2.927% |
| Wan | 0.834% | 1.013% |

Wan 的 W4+rank32 权重重构反而更准确。

#### 数据和图表

- [FLUX 原始数据](quant_error_diagnosis/flux_weight_reconstruction.json)
- [Wan 原始数据](quant_error_diagnosis/wan_weight_reconstruction.json)
- [weight_reconstruction_error.png](quant_error_diagnosis/weight_reconstruction_error.png)

#### 结论

Wan 质量差不能归因于静态权重重构 NMSE 更大。

### 10.2 首步逐 block 累积误差

#### 做了什么

Wan 捕获两个 CFG 分支的 block 输出，先每三层采样，再加密 block 18–24。FLUX 使用
真实 Nunchaku `forward_layer` / `forward_single_layer` 的 block 输出。

#### 观察

Wan 两分支平均 NMSE：

| block | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 27 | 29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NMSE | 5.69% | 1.24% | 1.73% | 16.50% | 16.57% | 10.83% | 6.36% | 0.94% | 0.716% |

FLUX residual stream 峰值为 2.646%，最后测点 2.012%。Wan 在 block 21–22 发生强烈
局部放大，之后又被后续 block 收缩，误差并非单调累积。

#### 数据和图表

- Wan 稀疏测点：[wan_firststep_metrics.json](quant_error_diagnosis/block_propagation/wan_firststep_metrics.json)
- Wan 加密测点：
  [wan_firststep_dense_head_metrics.json](quant_error_diagnosis/block_propagation/wan_firststep_dense_head_metrics.json)
- FLUX：[flux_firststep_metrics.json](quant_error_diagnosis/block_propagation/flux_firststep_metrics.json)
- 图：
  [firststep_block_error_propagation.png](quant_error_diagnosis/firststep_block_error_propagation.png)

#### 结论

Wan 的关键差异是网络对扰动的放大方式，而非输入局部量化误差本身。block 21–23 是
当前最合理的混合精度候选，优先级高于凭静态 layer NMSE 选层。

### 10.3 输出头与 CFG 放大

#### 做了什么

分解 block29 residual → norm_out → timestep modulation → proj_out，并分别记录
unconditional 和 conditional 分支，最后按 CFG=6 合成。

#### 观察

| 测点 | 两分支平均 NMSE |
|---|---:|
| block29 residual | 0.716% |
| norm_out | 0.712% |
| timestep modulation 后 | 2.991% |
| proj_out | 0.400% |

proj_out 单分支：

- unconditional：0.349%
- conditional：0.451%
- `uncond + 6 × (cond-uncond)`：**5.411%**

该 5.411% 与独立 rollout 实验首步 transformer output 的 5.071% 基本闭合。

#### 数据和图表

- 原始数据：
  [wan_firststep_dense_head_metrics.json](quant_error_diagnosis/block_propagation/wan_firststep_dense_head_metrics.json)
- 图：
  [wan_firststep_output_head_error.png](quant_error_diagnosis/wan_firststep_output_head_error.png)
- 最终诊断摘要：[summary.json](quant_error_diagnosis/summary.json)
- 本组详细报告：[REPORT_zh.md](quant_error_diagnosis/REPORT_zh.md)

#### 结论

首步最终误差大的直接原因不是 proj_out 本身很差，而是两个 CFG 分支存在不同的量化
误差，差分后被 CFG=6 放大。FLUX 当前比较路径没有 Wan 这种双分支 CFG=6，所以相同
量级的单层 A4/W4 误差可以产生完全不同的最终效果。

---

## 总体结论

所有实验合起来支持如下因果链：

1. Wan 的 smooth 后 activation outlier/A4 NMSE 不比 FLUX 大；
2. Wan 的 W4+rank32 静态权重重构 NMSE 也不比 FLUX 大；
3. 但 Wan block 21–22 会把量化扰动放大到约 16.5%；
4. 后续 block 会收缩部分误差，但 conditional/unconditional 分支误差并不完全相同；
5. CFG=6 将两分支误差差分放大，使首步 guided output 达到约 5.1%–5.4% NMSE；
6. 误差在 50-step rollout 中进入 latent 并持续累积，最终成为单帧纹理损伤、边缘异常
   和跨帧闪动。

因此下一步优先级应当是：

1. block 21–23 精确混合精度消融；
2. 直接校准 `cond-uncond` 或 CFG-guided output；
3. 以 block 输出、scheduler 后 latent 或短 rollout loss 作为校准目标；
4. 再测试 timestep modulation 邻近路径 A8/BF16；
5. 普通单层 MSE 校准继续扩量或盲目增加 rank 的优先级较低。

