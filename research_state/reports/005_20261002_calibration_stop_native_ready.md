# 阶段报告 005：模态校准未过门槛，原生算子对照就绪

2026-10-02。E004/E005均已完成；E006的原生attention与固定scale编码器前置验证通过。还没有可发表的新方法或最终视频质量收益。

## 尝试与结论

**普通校准目标对照（E004）。** 固定旧SVDQuant的量化权重、smooth、低秩A，只重拟合block0最后投影的B，三个方案参数量相同。4个训练样本，另4个样本评估，均来自旧PTQ校准池、仅本轮拟合隔离。实际计算保持torch BF16 attention。

| 拟合目标 | 局部video误差变化（对旧corrected） | 完整denoiser video误差变化（对旧corrected） |
|---|---:|---:|
| 普通pooled MSE | −39.0% | +4.6% |
| 各模态相对MSE等权 | −45.4% | +8.1% |
| 视频权重4、音频/文本各1 | −46.6% | −2.4% |

视频加权对普通MSE的最终video收益为6.74%，未过预设10%门槛，且音频误差比普通MSE高18.44%。p11/p46上原corrected相对plain的总体排序与E003的p1相反；之前的模态取舍不是已成立的跨提示词规律。**停止该路线的进一步扫参**；不能用显著局部改善代替完整输出收益，也不把阴性结果当新机制。

**模拟与原生计算核对（E005）。** 在真实H3 qkv/fc2输入、两时间步、plain/SVD两recipe上，独立pack/unpack复现旧QDQ，原生SM120 FP4 kernel由profiler确认。RNE中点舍入与原生累加次序使对BF16的线性层误差最多相对改变0.175%，未改变plain/SVD视频误差排序。这个有限核查没有发现足以解释此前失败的格式假象；尚未验证整个原生量化模型。

**下一问题的必要条件（E006）。** 隔离环境安装FlashInfer0.7.0.post1，在SM120上跑通原生FP4 attention；非整块长度、短零输入、独立batch检查通过。独立固定scale重编码器通过7类原生逐字节对照，包括精确中点、负零和scale下溢。为后续因果实验保留当前输入的smoothing/correction，仅冻结QKV scales；内部P量化仍动态变化。

## 接下来

固定H3的3个block、2个prompt、2个时间步，检验“native W4A4 qkv projection × native FP4 attention”的2×2交互，并加入固定QKV scale的对照。先看实际张量交互是否有量级、是否伤害结果以及固定scale是否有解释力。必须逐段保持原attention mask，且每个真实输入继续验证编码一致性。这个组合本身已有先例，只有得到可重复且超出现有解释的机制，才推进完整生成和更大评测。

## 可复核产物

- [E004逐例结果、预设门槛与止损](../06_experiments/E004_modality_reweight_results.md)
- [E005量化格式与原生GEMM报告](../06_experiments/E005_h3_native_contract.md)
- [E006固定实验入口](../06_experiments/E006_attention_interface_plan.md)
- [当前状态](../00_state/current_state.md)
