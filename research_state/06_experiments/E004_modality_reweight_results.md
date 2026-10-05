# E004 结果：普通重加权改善局部误差，未达到完整输出门槛

2026-10-02。**实验完成并按预注册止损：不继续扫权重或ridge，也不把本次阴性当作新机制证据。**

固定block0 rank32的A、W4残差权重和smooth，只重拟合fc2的B。训练p20/p25×step0/19，评估p11/p46×step0/19。所有样本都属于原PTQ校准集；只与本轮refit隔离。完整packed序列，torch BF16 attention，block0之后49个block均BF16。

## 主结果

以下aggregate NMSE均为所有案例误差能量之和/参考能量之和，未平均不同分母的NMSE。

| 方案 | 评估block视频误差相对旧corrected | 完整denoiser视频NMSE | 相对pooled视频改变 | 完整denoiser音频NMSE |
|---|---:|---:|---:|---:|
| plain W4A4 | — | 0.00141451 | +0.61% | 0.00075844 |
| corrected旧state | 基准 | 0.00134376 | −4.42% | 0.00084373 |
| pooled refit | −39.02% | 0.00140595 | 基准 | 0.00066244 |
| 各模态NMSE等权 | −45.40% | 0.00145264 | **+3.32%** | 0.00068172 |
| video:audio:text=4:1:1 | −46.61% | 0.00131120 | **−6.74%** | 0.00078461 |

- 三个refit在四例的block视频误差都改善；真实训练目标分别下降19.57%、29.82%、41.69%，与FP64闭式预测接近。因此不是“拟合未起效”的阴性。
- video4完整视频输出在3/4例优于pooled、aggregate优于plain，但其**6.74%改善未达到预先固定的10%门槛**。相对旧corrected只改善2.42%；没有稳定强收益。
- normalized完整视频误差相对pooled反而增加3.32%，只2/4例改善。
- video4音频相对旧corrected改善7.01%，但相对pooled恶化18.44%。不能隐藏比较对象而称同时优化。

## 逐例完整视频输出 NMSE（×10⁻³）

| case | plain | corrected旧state | pooled | normalized | video4 |
|---|---:|---:|---:|---:|---:|
| p11 step0 | 1.14887 | 0.90333 | 1.06222 | 1.10219 | 1.10636 |
| p11 step19 | 0.73002 | 0.71200 | 0.63529 | 0.61434 | 0.61925 |
| p46 step0 | 2.38532 | 2.47457 | 2.50161 | 2.60018 | 2.19355 |
| p46 step19 | 0.17548 | 0.17842 | 0.14780 | 0.13142 | 0.12810 |

E003中p1的“corrected比plain视频更差、音频更好”不是一致跨prompt现象：本次四例aggregate corrected视频比plain**好5.00%**、音频却**差11.24%**，方向相反。p11两个step的corrected视频均优于plain；p46两个step略差。不能把E003个例取舍当成已经成立的模型级规律。

## 决策

1. 当前证据不足以把模态误差取舍或普通加权校准立为新贡献；MBQ/MixDQ/PulseQuant等现有工作撞题风险不变。
2. 该固定A/fc2-B简单对照确实能显著改善本轮隔离的块级视频误差，但大部分改善未传到完整denoiser。继续以局部NMSE作为优化收益代理的风险仍在。
3. 不继续本轮扫参，不增加层数/rank追结果。负结果只约束这个有限简单对照，不能证明一般模态重加权无效，也不能据此声称新敏感性机制。
4. 这些都是teacher输入下单次denoiser输出；没有自由生成、解码、VBench或用户可感知质量收益证据。

## 正确性与运行记录

- 4例BF16 zero pulse：video/audio逐元素一致，误差0。
- 每例teacher与所有arms真实torch SDPA调用数相同：p11为52、p46为102（packed分段不同）；Q/K/V全为BF16。
- 所有W4残差权重、smooth、A运行前后SHA256一致；只B变化。
- 使用全部有效模态token累计正常方程，padding不作为拟合目标；完整模型保留padding。
- 三臂同参数量、同数据、同ridge规则1e-3，不搜索。BF16实际重放确认训练收益，不用线性近似替代真实结果。
- 时间325.4秒，GPU1峰值33.21GiB，EXIT_CODE=0，已释放GPU。时间包括模型加载/磁盘offload，**不是性能基准**。
- 预注册：[E004_modality_reweight_plan.md](E004_modality_reweight_plan.md)。CPU求解自检通过，语法检查通过。

原始完整结果：`results/research/E004_h3_modality_reweight.json`。机器摘要：`research_state/06_experiments/results/E004_summary.json`。日志：`results/logs/E004_h3_modality_reweight.log`。脚本：`scripts/research/probe_h3_modality_reweight.py`。三个B产物在 `results/research/E004_*_fc2_B.pt`；必须与**旧state重构出的固定qweight**配合，不能直接覆盖旧state的B后又重算残差qweight，那会改变本实验定义。旧state未改动。
