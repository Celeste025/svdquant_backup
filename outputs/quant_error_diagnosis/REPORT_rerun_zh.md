# 实验 7–10 重跑报告（MSE + NMSE 双指标）

重跑时间：2026-07-30。旧诊断产出已删除后完整重捕获。  
指标定义：

- `MSE = mean((quant − bf16)²)`
- `NMSE = MSE / mean(bf16²)`（分母 `clamp_min(1e-20)`）
- 表格中的 `%` 为 `NMSE × 100`

跨模型比较优先看 **NMSE**；同模型沿深度/测点追踪绝对误差看 **MSE**。  
FLUX residual 幅度远大于 Wan，原始 MSE 不可跨模型直接对比。

## 代码修复点

1. **Exp7 第一轮 FLUX**：`collect_flux_dev_scaled_activation_metrics.py` 现对 packed smooth 做 `unpack_scale`，FFN-down 使用 `shift=0.171875` + unsigned INT4；prompt/seed 与诊断脚本对齐。
2. **共享指标**：新增 [`scripts/quant_error_metrics.py`](../../scripts/quant_error_metrics.py)（`mse_nmse` / `quant_metrics`）。
3. **Exp8 multilayer**：除 absmax surface 外，同步采集 postsmooth A4 的逐步 `mse`/`nmse`。
4. **画图**：Exp7/9/10 图均同时展示 MSE 与 NMSE；Exp10.2 FLUX 汇总固定 `stream=="image"`。
5. **OOM**：A4 指标计算强制在 CPU，避免与 DiT 抢显存。
6. **环境**：Wan 用 `wan-qvdit`，FLUX/Nunchaku 用 `svdquant` + `LD_LIBRARY_PATH`。

重跑脚本：[`scripts/rerun_exp7_to_10.sh`](../../scripts/rerun_exp7_to_10.sh)。日志：`logs/exp7_10_rerun/`。

---

## 实验 7：postsmooth A4

### 第一轮（广覆盖，采样不严格配对）

| 模型 | A4 MSE 中位数 | A4 NMSE 中位数 | max/RMS 中位数 |
|---|---:|---:|---:|
| FLUX-dev（修后） | 7.70e-4 | **1.372%** | 12.33 |
| Wan | 3.13e-4 | **1.151%** | 10.28 |

相对旧结果：旧 FLUX NMSE 中位数约 **2.635%**（smooth 未 unpack、FFN-down 未 shift）。修后 FLUX 局部 A4 明显下降，与 Wan 更接近；结论「Wan 并不因 A4 更差」仍然成立，且 FLUX 侧不再被错误放大。

图：`outputs/activation_error_comparison/scaled_activation_error_comparison.png`

### 第二轮（严格配对层）

| 模型 | A4 MSE 中位数 | A4 NMSE 中位数 | max/RMS 中位数 |
|---|---:|---:|---:|
| FLUX-dev | 7.91e-4 | **1.430%** | 11.10 |
| Wan | 7.97e-4 | **1.278%** | 7.47 |

与旧严格配对结果几乎一致（旧 NMSE：FLUX 1.428% / Wan 1.278%）。

图：`outputs/quant_error_diagnosis/postsmooth_activation_a4_error_mse_nmse.png`

---

## 实验 8：timestep outlier + A4

### 单层（~2/3 深度 FFN-down）

| 指标 | FLUX | Wan |
|---|---:|---:|
| channel timestep range 中位数（log2） | 1.286 | 0.758 |
| step p99 range（log2） | 1.137 | 0.523 |
| 相邻 top1% channel Jaccard | 0.752 | 0.903 |

与旧结果一致：该层上 Wan outlier channel 更稳定。

### Multilayer postsmooth A4（新增双指标）

| 模型 | median A4 MSE | median A4 NMSE |
|---|---:|---:|
| FLUX | 7.89e-4 | 1.419% |
| Wan | 7.72e-4 | 1.278% |

图：`outputs/timestep_outliers_multilayer/multilayer_a4_mse_nmse.png`  
surface / Jaccard：`outputs/timestep_outliers*/`

---

## 实验 9：rollout / teacher-forced / latent

关键步（NMSE% / MSE）：

| 模型/step | Teacher-forced | Rollout output | Post-step latent |
|---|---|---|---|
| FLUX step1 | 0.100% / 1.92e-3 | 0.137% / 2.63e-3 | 0.00012% / 1.20e-6 |
| Wan step1 | 5.071% / 0.136 | 5.071% / 0.136 | 0.001% / 1.07e-5 |
| FLUX step26 | 0.353% / 7.59e-3 | 4.998% / 0.107 | 0.324% / 2.08e-3 |
| Wan step26 | 6.825% / 0.151 | 19.824% / 0.438 | 2.823% / 1.78e-2 |
| FLUX step50 | 4.184% / 7.71e-2 | 31.018% / 0.571 | 7.616% / 9.63e-2 |
| Wan step50 | 146.302% / 2.587 | 136.532% / 2.415 | 26.929% / 0.345 |

与旧表几乎逐格一致。Wan 首步输出已有 ~5% NMSE；rollout 继续放大。

图：`outputs/rollout_latent_error/wan_flux_output_latent_error_mse_nmse.png`

---

## 实验 10：权重 / block / CFG

### 10.1 权重重构（12 层）

| 模型 | median MSE | max MSE | median NMSE | max NMSE |
|---|---:|---:|---:|---:|
| FLUX | 5.94e-5 | 2.05e-4 | 1.770% | 2.927% |
| Wan | 2.00e-5 | 2.34e-4 | 0.834% | 1.013% |

结论不变：Wan 静态 W4+rank32 重构不差。

### 10.2 首步 block 传播（Wan CFG 分支平均）

| 指标 | 峰值位置 | 峰值 |
|---|---|---:|
| MSE | **block 21** | 0.151 |
| NMSE | **block 22** | 16.573% |
| FLUX image peak NMSE | — | 2.163% |
| FLUX image peak MSE | — | 182.1（尺度不可比） |

block 21–23 仍是高误差区；峰值按指标在 21/22 之间略有偏移。

### 10.3 输出头 + CFG

| 测点 | MSE | NMSE |
|---|---:|---:|
| block29 residual | 2.09e-2 | 0.716% |
| norm_out | 7.12e-3 | 0.712% |
| timestep modulation | 5.54e-3 | 2.991% |
| proj_out | 4.64e-3 | 0.400% |
| CFG=6 guided | **0.150** | **5.411%** |

CFG 相对分支均值放大：**MSE ×32.3**，**NMSE ×13.5**。  
modulation 的 NMSE「尖峰」对应 `ref_power` 从 ~1.0 降到 ~0.19；绝对 MSE 实际下降。

图：

- `outputs/quant_error_diagnosis/weight_reconstruction_error_mse_nmse.png`
- `outputs/quant_error_diagnosis/firststep_block_error_mse_nmse.png`
- `outputs/quant_error_diagnosis/wan_firststep_output_head_error_mse_nmse.png`
- `outputs/quant_error_diagnosis/summary.json`

---

## 与旧结论对比

| 主张 | 重跑后 |
|---|---|
| Wan postsmooth A4 不比 FLUX 差 | **仍成立**；第一轮 FLUX 因 bug 曾被高估，修后更支持该结论 |
| Wan timestep outlier 系统性更差 | **仍不成立** |
| Wan 首步输出已 ~5% NMSE，rollout 放大 | **数值几乎复现** |
| 权重重构 Wan 更准 | **成立（MSE/NMSE 均支持）** |
| block 21–22 放大 | **成立**；MSE 峰在 21，NMSE 峰在 22 |
| CFG=6 放大分支误差 | **成立且 MSE 放大倍数更大** |
| modulation 放大误差 | **仅 NMSE 表象**；MSE 下降 |

因果链不变：局部 A4/W4 不是主因 → 中部 block 放大 → CFG 差分放大 → 多步 rollout 进入 latent。
