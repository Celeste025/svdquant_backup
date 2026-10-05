# 053 — 激活来源对照停在数值控制，尚无机制结论

2026-10-04。MiniMax-H3为主；普通native适配、伪量化或kernel修复不计创新。目前仍无可投稿核心贡献或经过独立人评的等质量优化。

## 动机 / 观察

E056–E058在H3两个短窗口发现量化误差有持续的方向相关，但交互项也会抵消误差，不能直接把相关性或交互范数当优化目标。下一决策问题是：**保留一份固定W4权重时，激活表示残差是否提供值得干预的跨步误差来源？** 旧software QDQ与native本身存在不可忽视的差异，不能直接切换实现来归因W/A。

## 实验

E059预先规定保留原native FP4主GEMM、bias及BF16低秩支路，只额外加回实际packed激活的表示残差经过同一packed权重后的FP32投影。它是高成本来源对照，不是精确W4A16、唯一W/A方差分解或部署方法。

固定p30 s5→6、p36 s14→15各三个已有输入，共六输入。先执行六次零补偿，要求raw/velocity逐元素复现历史；再允许六次真实补偿。每个真实补偿调用在block0四个linear固定前2行×32输出核对CPU FP64公式，且加回BF16的舍入RMS须小于补偿RMS的10%。这项局部数值门槛不是所有层/模态的代表性抽样，也不是科学不可行性定理。计划上限12次完整前向、单空闲GPU、1800秒。

## 结果 / 结论

- CPU检查：六份actual input全部匹配历史，1.160秒、0 CUDA/0模型前向。
- GPU执行：六次零补偿的raw/velocity全部精确复现。首个真实补偿在block0 FFN-down数值控制处停止；共7次attempt、6次完整前向、**0次完整oracle前向**，耗时82.839秒。
- 四个小样本的FP32补偿都通过FP64公式检查；失败位置的相对RMS计算误差为1.87e−6。但该位置BF16加回舍入RMS为1.94466，补偿参考RMS为18.80998，比值**10.3384%**，略超预设10%门槛。其余三个样本为3.9229%、2.0436%、2.3987%。
- **没有完整oracle输出，因此没有运行主效果汇总，也没有得到激活来源的重要性、跨步点积下降或质量改善结论。** 这是对照分辨率门槛未通过，不是激活机制阴性，不证明权重误差主导。不能把10.34%与10%的差距解释成普遍不可辨识。

执行源、失败记录及真实codes/scales样本均保留；不放宽阈值、重跑精度网格或用部分层读出替代预定主终点。原worker已退出，GPU0已释放。本次工程与数值检查不作为新研究贡献。

## 后续决定

按原投入规则，停止继续推进当前activation时序方法候选；来源问题保留未判定状态。没有证据支持立即训练、增加权重副本、缓存历史activation或开展视频质量评测。下一轮需要先提出新的、可区分竞争解释的研究预测；不能仅因这次控制未完成而继续调数值接口。研究目标保持active，但此候选没有晋级为方法。

## 可追溯材料

- [预写计划](../06_experiments/E059_h3_activation_residual_oracle_plan.md)
- [执行器](../../scripts/research/probe_h3_activation_residual_oracle.py)，SHA `7d91212dc09af739738d9d2cf1ff332007350c91f9ee4a9c969b81d7c721354b`
- [CPU检查](../../results/research/E059/check.json)，SHA `0421077ef18cfe9134464e45818f8e6ad0684f9e8f72e8808d89b80d4a09b395`
- [执行与停止记录](../../results/research/E059/evaluate.json)，SHA `cfdc5751e8c5078b22112432b77eb65fd8aa1b5333b581773c4b3b03a74da561`
- 失败样本：`/data1/models/svdquant-wjq/research/20261004/E059/failed_samples/oracle/e010_p030_s05_source_teacher.pt`，SHA `9cef3c7138e0d59b31ca576cf87ae7e73ec949f1794e98409d3b06f51449aa9a`

独立复核已完成：[independent_check.json](../../results/research/E059/independent_check.json)，SHA `84c1597ba9029aeeaec635df496ef98c36221dfd45c4f7fedf4abfb1344cbb64`。CPU 0.420秒、0GPU；六份actual输入和24份raw/velocity数组与历史逐byte一致。独立NumPy从真实codes/scales/global重算四个FP64补偿，参考张量最大差为0，标量统计最大差2.22e−16，确认同一数值门槛失败。
