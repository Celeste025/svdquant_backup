# 025：省掉块均值校正矩阵的便宜路径会损失什么

2026-10-03。**E026用两次原生attention、零DiT，确认了固定原均值而复用FP4 K会引入可见精度损失；同时仍保留部分对全局均值方案的优势。它是一次有方向的算子诊断，尚非新方法或加速结果。**

复用E018已保存的真实H3 p36/seed59526/step14/block0输入，56个head、22,539有效tokens。保持Q/K/V codes及scale六个packet不变，仅替换第七个块均值校正项；相同原生FP4 softmax/P/PV实现，实际P随score改变。官方当前校正为μKcᵀ，其中μ和Kc来自BF16数据、乘法及输出FP32；干预为μKhat_cᵀ，μ不再额外量化，Khat_c由实际FP4 packet按真实scale/行布局还原。

| 全有效token、全head合并 | 相对同一BF16输出NMSE | 余弦 |
|---|---:|---:|
| 原block mean | .00241969 | .998995 |
| 固定原μ × FP4 K还原值 | .00331458 | .998538 |
| 已有global mean参照 | .00356130 | .998426 |

干预比原block mean的NMSE提高36.98%，56个head中52个变差；仍比global mean低6.93%，56个head中53个更好。它消耗本例block相对global误差优势的78.39%，并未将所有优势完全抹掉。去掉key方向行常数后、再按softmax scale缩放的校正差RMS=.13118。原packet重放与已保存block输出实际相同，但未把逐位相同设成研究前置门槛。

机制边界很具体：即使保持原μ，仍新增μ(Khat_c−Kc)ᵀ，因此只把μ编码得更精细不直接消除这一项。不过本实验没有重新优化μ或K，所谓oracle只是“不再增加μ编码误差”的固定μ对照，**不是所有融合或训练方案的数学下界**。单层NMSE也不代表最终视频质量；此前E017已有反例。

原校正矩阵形状[1,56,177,22656]，其生成/存储是拟优化对象。把均值当附加query行在SM120可表达，但现成M128已经占满，朴素额外M16 stripe并不免费；复用FP4 K的精度代价现已测得。若用原BF16 K保证输入精度，需要计入其额外读取、BF16 MMA低行利用率及广播/占用。已有工作已覆盖校正的融合消费和Q-chunking；本轮未发现的生产侧实现差异不等于新颖性证明。[近邻与代数核查](/home/wjq/workspace/svdquant-exp/research_state/02_problems/query_mean_augmented_rows_check.md)。

实际两次attention共18.53秒（含读盘与CPU统计），峰值allocated4.05GiB；这不是内核时延。没有新kernel、完整生成或加速声称。[运行记录](/home/wjq/workspace/svdquant-exp/results/research/E026/run.json)、[固定实验计划](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E026_qmean_k4_oracle_plan.md)。

下一步只评估一个具体成本问题：保留BF16 K输入的在线校正，是否能超过已有query-chunking基线。优先只改现有correction生产/消费接口；若需要重写整个调度，先用真实chunk基线与独立M16 stripe测量决定开发预算。独立stripe时间不能冒充融合时延或下界。当前仍没有足以投稿的核心贡献。

独立CPU复核已完成（6.12秒、CUDA未初始化）：直接从保存输出重算pooled/per-head指标，独立还原K4和实际μ并核对原六packet；[independent_summary.json](/home/wjq/workspace/svdquant-exp/results/research/E026/independent_summary.json)。correction仅保存hash而非完整tensor，未冒称其FP32 matmul已独立逐位重算。
