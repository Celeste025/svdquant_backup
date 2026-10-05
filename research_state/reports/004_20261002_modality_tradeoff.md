# 阶段报告 004：H3 局部优化与最终视频响应发生取舍

日期：2026-10-02。E003完成；这是小规模机制诊断，尚无可投稿的新方法或视频质量提升证据。

## 做了什么

在MiniMax-H3的同一个校准prompt、step0/19，把block0的plain W4A4或修正后SVDQuant误差注入BF16模型，观察剩余49层后的video/audio denoiser输出。另分别只注入text行或video行的真实误差。所有目标线性层以外的计算固定，实际attention为torch BF16 SDPA；两次零误差回放均严格复现teacher。

## 结果

以下变化均是SVD相对plain，NMSE越低越好。

| step | 局部全部token误差变化 | plain最终video NMSE | SVD最终video NMSE | video变化 | 最终audio NMSE：plain → SVD |
|---|---:|---:|---:|---:|---:|
| 0 | -10.98% | 0.00117165 | 0.00133185 | +13.67% | 0.00180213 → 0.00083862 |
| 19 | -0.99% | 0.00022152 | 0.00024038 | +8.52% | 0.00039432 → 0.00032644 |

两个时间步都出现：局部总误差下降，最终video误差增大，audio误差下降。text注入误差能量分别是video的34.15倍、3.88倍，但仅video误差产生的最终video偏差却分别是仅text误差的3.28倍、1.65倍。总能量在这两个case中不能可靠决定哪个校准目标更好。

这比较的是实际误差，不是等能量扰动，不能据此说video模态具有更高的内在敏感度。单独pulse来自全block量化输出，也不等于只量化某一种token。非线性传播使各pulse不可直接相加。

## 判断与下一步

这是一条有用的故障线索，但MBQ、MixDQ、RGSQ及PulseQuant已经覆盖模态加权、局部与下游误差不一致等方向。**普通模态重加权本身不构成新颖性。**

下一步只安排一个强简单对照E004：固定其余参数，等容量重拟合最后一个MLP投影的rank32分支，比较pooled MSE、模态NMSE等权和视频加权。用不同prompt隔离本轮拟合与评价；它们仍属于旧PTQ校准池，不能称作全流程heldout。若简单目标解决问题，就归档为工程改进，换研究问题。并静态核查真正heldout及不同block的最短验证路径，不先扩成全模型调参。

## 边界与产物

没有自由生成、感知质量指标或端到端原生FP4性能结果；旧量化state尚未重新校准。本实验峰值显存36.23 GiB，运行GPU已释放。首轮检测到默认Sage后在forward前停止，其日志保留；正式结果全部在显式torch后端重算。

- [完整实验契约和数据解释](../06_experiments/E003_modality_pulse.md)
- [汇总与原始结果哈希](../06_experiments/results/E003_summary.json)
- [原始结果](../../results/research/E003_h3_modality_pulse.json)
- [当前研究状态](../00_state/current_state.md)
