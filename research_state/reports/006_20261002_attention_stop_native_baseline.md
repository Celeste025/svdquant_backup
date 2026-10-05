# 阶段报告 006：attention 接口假设止损，转向真实运行基线

2026-10-02。当前没有已成立的论文贡献，也没有经过视频质量验证的优化收益。

**E006 的结果不支持“投影量化触发 attention scale 切换，造成显著额外损伤”这条路线。** 12 个预指定案例全部完成；遵守事先门槛，停止该路线，不扩大 continuation 或参数搜索。

## 尝试与结果

在 H3 的 2 个 prompt × 2 个时间步 × 3 个 block 上，比较 BF16、仅原生 NVFP4 QKV 投影、仅原生 FP4 attention、两者同时开启，以及固定为 BF16 输入对应 QKV scales 的反事实。真实分段 attention、QK norm/RoPE 和 padding 语义保持一致；其他 linear 仍为 BF16。

| 视频 attention 输出上的预注册指标 | 结果 | 判断 |
|---|---:|---|
| 非加性交互能量 / 组合误差，中位数 | 31.71% | 有交互，但这本身不是损伤 |
| 组合误差 / 两个单独误差向量相加后的能量，中位数 | 0.9608；0/12 大于 1 | 每个案例都是净抵消，未出现假设中的放大 |
| 固定 QKV scales 后误差 / 原组合误差，中位数 | 0.9354；12/12 改善 | 改善 6.46%，未达到事先设定的 10% 继续门槛 |

这说明不能仅凭“大交互项”立题：交互项与原误差的交叉项为负，整体反而稍小。固定尺度有一致的小收益，但它使用 BF16 oracle，且当前只是局部误差；不能当作可部署方法或视频质量提升。样本来自旧校准数据，门槛用于分配研究资源，不是显著性检验，也不否定其他量化交互问题。

完整实验耗时 312 秒、峰值 allocated 显存 38.17 GiB，退出码 0。12 次 BF16 回放误差为 0；24 个 attention segment 的 T0/T1 共 48 次独立重编码全部逐字节相同，固定尺度分支保留当前输入 correction；profiler 确认原生 FP4 投影与 attention kernel。GPU 已释放。

## 下一阶段

建立 Wan 的完整原生 NVFP4 DiT 基线，先回答实际运行问题，再选方法。历史多数 W4A4 实验仍是 QDQ 后 BF16 GEMM，缺乏可靠的端到端性能证据。

1. 验证现有 checkpoint 的 300 个 residual weight 能原码导出，保留 smoothing 和高精度低秩分支；先闭合全部 30 blocks 的数值验证。
2. 使用可计时的 activation packer，测完整 linear、完整 DiT 的延迟、峰值显存和耗时分布。报告包含 pack、动态 scales、低秩分支等实际成本。
3. 根据瓶颈决定是否值得研究算子、稀疏或并行化交互；已有的 QKV/down 共享和常规融合只计作工程基线。通过数值与性能验证后，再投入视频生成和质量评估。

该阶段是研究基础建设，不包装为新贡献。当前并行工作维持三项：原生 adapter、独立运行/验证、有限系统文献核查。

证据：[E006 计划](../06_experiments/E006_attention_interface_plan.md)、[可复算摘要](../06_experiments/results/E006_summary.json)、[完整原始结果](../../results/research/E006_h3_attention_interface.json)、[原生 Wan 可行性](../00_state/native_wan_feasibility.md)。
