# 054 — 校正H3低秩基线口径，准备有界初始化对照

2026-10-04。本轮没有新增方法或视频质量结论。上一轮E059已实际执行、独立复核并改变下一行动，归类为progress；本轮E060完成新的基线来源核验。

## 动机 / 观察

检查“低秩与低位支路是否大幅相消”时发现，理想权重截断SVD的正交性不能直接套到当前H3配方。现有生成脚本先对量化残差做随机SVD，候选之间重置该残差；官方校准器也支持残差初始化，但会携带并更新上一轮量化权重。**初始化本身不是错误，当前实现与完整官方校准的等价性却没有建立。**

这影响研究判断：既有量化残余不能直接作为“标准SVDQuant已经解决不了”的证据。普通校准修正也不应被包装成新方法。

## 核验 / 结果

E060只读CPU审计完成，0.725秒、0CUDA/0模型前向：

| 核验对象 | 结果 |
|---|---|
| 301.6MB原quant_state重新SHA256 | 与E009导出记录一致 |
| 200层导出的smooth、A、B，共600字段 | 全部与state逐byte一致 |
| 每层保存的实际候选数 | 2/3/4/5/6次分别为102/71/23/3/1层；配置上限50不等于执行50次 |
| 当前候选脚本与vendored校准器 | 前者重新初始化Q0并换seed，后者跨轮更新self.qw |
| 历史生产源码身份 | 原导出没有绑定生成脚本和SVD实现的SHA，不能由今天的源码和候选数反推历史执行过程 |

本次只核因子及元数据，未重新hash全部11.14GB packed导出；没有重新量化、修改checkpoint或测试质量。已有同一配方的对照数值保留，后续明确称**legacy smooth＋低秩残差基线**；不得单凭名称把它视为完整官方强基线。

另两项有限筛选没有留下新方法：理想SVD支路整体相消有正交反例，换因子表示不能自动消除实际误差；NVFP4跨token尺度耦合、scale下溢与常规分离解法已有直接覆盖。详[支路筛选](../02_problems/h3_branch_representation_screen_20261004.md)及[尺度筛选](../02_problems/h3_scale_support_screen_20261004.md)，不据此启动新kernel/scale网格。

## 后续实验 / 决策

E061已预写[计划](../06_experiments/E061_h3_lowrank_initialization_plan.md)，runner准备中，尚未GPU执行。固定现有smooth/rank32/native执行，以同seed比较“分解量化残差”和“分解平滑权重”两个初始化，并与现有selected基线比较完整200层H3输出。四个固定teacher输入的video误差须逐点同时优于两参照至少20%，才进一步考虑正式matched PTQ；否则停止该固定smooth对照。

这只判别初始化目标，**不检验官方carry-Q交替优化，也不等于完整官方SVDQuant复现**。只用既有两个提示的数值输出，不能证明泛化、质量或部署收益。不自动启动整模校准；不把补强基线计为创新。

可追溯：[E060计划](../06_experiments/E060_h3_lowrank_provenance_plan.md)、[CPU审计](../../results/research/E060/audit.json)，SHA `683ec67dfc1ab8876290977aa20a08923a6d101dd3bd258b07fbf04fcae38052`；[当前官方校准器源码](https://raw.githubusercontent.com/mit-han-lab/deepcompressor/main/deepcompressor/calib/lowrank.py)仅用于当前语义比较，不能替代历史生产记录。
