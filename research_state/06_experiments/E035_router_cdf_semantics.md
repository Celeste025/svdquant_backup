# E035 路由 CDF 边界说明（正式 router run 前）

2026-10-03。冻结 capture/plan 不改；仅消除 plan 中文“超过”与其指定原 BSA 规则之间的边界歧义。原作者本地 [bsa_attn.py:165](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/bsa_attn.py:165) 先保留首项，再令 `keep_sorted[...,1:] = cumsum[...,:-1] < cumulative_threshold`，随后 union `min_kv_blocks`。

因此本诊断遵循**最短累计概率达到或超过 p 的前缀（首个 CDF ≥ p），至少 4 块**。它等价于本 CPU 实现的 `crossing = (cdf < p).sum(-1) + 1`；不是严格大于 p。例：概率 `[0.5,0.25,0.125,0.125]`、p=0.5、最小数=1，应保留 1 项。真实实验仍固定 p=0.9/min4，不改几何、有效长度、其他参数或样本。

此前严格 `>` 的 v2 CPUcheck 未运行真实 capture 路由，原结果保留为 `results/research/E035/router_check_strict_gt_superseded.json`。修正版重新 CPUcheck；正式 run 使用本说明绑定的 source/check。验证依据选择器自己的 FP32 CDF；另一归约顺序计算的 masked-sum probability 仅报告，不施加逐位相等或科学误差门槛。并列排序顺序不作为研究结论。
