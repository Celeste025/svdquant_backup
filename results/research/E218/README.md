# E218 — 强 K 列的位置复现及 token 删除

## 实验

两个额外 prompt：25（小猫麻将，seed 2141）、48（划船，seed 9939）；每个 prompt 重跑 BF16 dense 闭环轨迹，480×832、124 帧、20-step 调度，在 step 0 和 10、layer 25 的 head 10/20/30 测试。Q64×K64 真实 FP32 softmax，分母包含全部有效视频及 prefix key；对每 head 的 video×video tile 按有效 query-key 概率均值全局取 top 10%。列覆盖率是 259 个视频 Q block 中与某 K block 对应的 tile 入选的比例。

对每个 prompt/step/head，选择列覆盖率最高的 K block，按所有 video queries 对它们的实际注意力概率总和选该 block 内 top 5 token。删去这 5 个 video key token，在该 head 的**所有视频 query**上重新归一化 softmax 并算注意力输出；分别比较同块随机选择 5 个 token 的 5 组固定种子对照、随机另外一个块的 top5，以及删整个强块。仅评估一次真实 QKV 条件下的 attention 输出，不影响后续 denoising。NMSE 的分母为被测 head 未删除时的全部 video-query attention 输出能量，属于 head-local 干预，不是整层误差或视频质量。

使用 `O_new = (O - sum_{j∈D} p_j v_j)/(1-sum_{j∈D} p_j)`，其中所有 p 基于完整 softmax（含 prefix）；对每个 case/head 的第一批 query，将公式与将目标 logit 设为 `-inf` 后重新 softmax 的直接结果核对，最大绝对差约 3.1e−5，符合 FP32 重排误差。每个 head、step 单独选择 top 5，该选择与评估数据相同；与随机对照的差异是诊断而非无偏泛化估计。

## 结果

head20 的时间分组偏好在新增两个 prompt 和两个 step 中均出现。按所有空间 K block 平均，latent 时间索引 `%5==0` 的列覆盖率为 21.7%–29.3%，其余四类的平均为 4.6%–6.8%；赛车 E217 的同 head 同层两个 step 也有类似偏好。`(latent t=0, spatial block=0)` 在新样本中列覆盖率为 48.6%–94.2%，其 259 个视频 K block 中排名分别为 7、3、2、2；但最强 block 并不总是它，最强位置分别为 block 5、105、175、3，因此并非唯一固定位置。五步时间结构与模型 `_T_GROUP=5`、`_FRAME_PER_TOKEN=(1,4,4,4,4)` 相吻合；位置编码、latent 内容或其他结构因素的因果来源尚未区分。其他两个 head 在这两个 prompt 的周期偏好不一致或较弱。

| Prompt / step | head | 强 block | 删除 top5 后 NMSE | 同块随机 5 token 的平均 NMSE | 删除整个强块的 NMSE |
|:---|---:|---:|---:|---:|---:|
| 麻将 / 0 | 10 | 63 | 0.0372% | 0.00109% | 0.0484% |
| 麻将 / 0 | 20 | 5 | 0.02485% | 0.00057% | 0.0370% |
| 麻将 / 10 | 20 | 105 | 0.13217% | 0.02659% | 0.1533% |
| 划船 / 0 | 20 | 175 | 0.03680% | 0.00131% | 0.0448% |
| 划船 / 0 | 30 | 162 | 2.17107% | 0.00035% | 2.1762% |
| 划船 / 10 | 20 | 3 | 0.08186% | 0.01375% | 0.1039% |

12/12 个 prompt-step-head 案例删除 top5 的 NMSE 都大于 5 次同块随机删除的均值；幅度差别很大。划船 step0 head30 的 2.17% 是最大的受测例子，不能当作典型值；因选的是所有 query 聚合注意力最高的 token，存在选择偏差，随机样本仅 5 次。删除整个块的误差也不保证大于删除其中 5 个 token：softmax 归一化与输出方向会变化，更多删除可能相互抵消。

前三块强 token 的注意力质量多数明显偏集中，但按注意力高低选择不证明其是人类可解释的语义实体，也不保证它们在生成闭环、其他 head 或 layer 的功能。更不能把 token 删除误差与 NVFP4 分配误差等同起来。要验证位置还是内容驱动，需多 prompt 和仅改变时间编码、置换内容的受控干预；要研究视频质量，需在闭环中测试，并注意干预可能超出模型分布。

逐 head 完整数值、5 组随机样本、token 坐标和列覆盖率在 `p025_step0.json`、`p025_step10.json`、`p048_step0.json`、`p048_step10.json`。入口 `scripts/research/run_e218_stripe_intervention.py`、启动 `scripts/research/launch_e218_stripe_intervention.sh`，日志 `results/logs/E218_p25.log` 和 `E218_p48.log`。
