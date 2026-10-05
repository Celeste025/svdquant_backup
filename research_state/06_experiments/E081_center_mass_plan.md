# E081：V中心化收益的来源

2026-10-05。Exploration / mechanism discrimination，非新方法创新声明。用户已授权继续实验。沿用decision-value-experiment-planner工作流。

问题与动机：E080 center32局部PV SSE下降40%–42%，含QK参考下降15%–16%，是应验证的阳性信号。但其同时改变V量化表示和均值项所使用的概率质量，不能将收益全归因动态范围。

假设：大V共同分量会把P量化概率质量误差转成共同输出误差；只修正此项可能解释中心化的部分或大部分收益。竞争解释是V残差量化本身改善。数学性质已知，不认领创新。

最小有效实验：完全复用E080四份clap r0/r1、s14/b0,b24捕获，原72query/56heads/完整keys和native_qk、exact_qk两条件。只保留g32、128两个已测设置，不扫描。mu与E080相同为BF16块均值；仅全video块处理。P含原online缩放/未量化normalizer，hatP指同一轮P量化重构，所有路径共享P。

四臂（每个g分别）：
- base = hatP Q(V)
- residual_quantmass = hatP Q(V−mu) + (hatP1)mu
- base_massfix = hatP Q(V) + ((P−hatP)1)mu
- center_exactmass = hatP Q(V−mu) + (P1)mu

residual_quantmass vs base隔离改变V表示的有限效果；center_exactmass vs residual_quantmass隔离概率质量补偿，base_massfix单独测试该补偿。两种改动的向量加和恒等式检查，但SSE收益有交叉项，不当作可加百分比贡献。residual减法先BF16，误差计入V表示，不能称无代价纯量化噪声修复。

必要检查：QKV官方byte/scale/dequant parity；base和center_exactmass输出对E080逐tensor精确复现；FP64 online/dense恒等式；修正向量identity归一能量≤1e−8；独立CPU复算SSE/contrast及向量关系。不得以错误实现作阴性。

读出：每case保留PV及full-attention SSE/差分SSE，输出全部向量；固定native_qk是主要、exact_qk验证来源方向。报告四臂实际改善、补偿量及其与原误差的交叉项。只有两个seed，不把head/query当独立样本。

决策：若massfix在两b24 seed均达到center整体SSE改善量的≥80%，记为质量补偿可解释大部局部收益；否则若移除补偿损失≥20%整体改善，记为两者均值得关注；其余报告V表示占主导。此分类只是决定下一干预，不作为创新或视频质量证据。无论哪种解释，只要复现center阳性，保留有限视频验证优先级。视频阶段需要完整变长/QK/P合同与可接受工程成本，单独冻结方案，不用不等价替代冒充原中心化。

资源/停止：单GPU0命名tmux、1200秒/60GiB；0DiT/训练/新模型/视频/校准。复用原数据不重捕获。数值失败保存记录后修新版本；不扩group/层/步。E080原源和结果不改。视频实现只做必要接入，不将常规kernel开发计为创新。
