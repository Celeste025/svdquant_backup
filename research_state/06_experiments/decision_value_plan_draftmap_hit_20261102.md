# DraftAttention2 选块命中率：单 prompt 机制检查

- 模式：探索。问题：原始 DraftMap 的高分块与真实注意力高分块有多一致，以及这一致性是否随去噪步或层深变化。
- 假设：命中率高于相同预算的随机选择；随 timestep、层深可能变化。它只研究同一步选块，不直接验证跨步路由复用。
- 数据：赛车 p016、seed 20026、480×832、124 帧、20 步 BF16 teacher 闭环轨迹；层 0/12/24/36/49，预先固定 head 0/8/16/24/32/40/48/55。Q128、MeanMax=0.2、空间 ragged 重排、无 anchors，与项目此前路由设置一致。
- 原版代理：调用 SM120 原生 preparation 与 DraftMap 概率算子；每 head 在所有 video Q×K 块上按官方全局预算稳定降序排序。分别保留 10/20/30/50%。
- 真实参考一：同输入 BF16 Q/K，对每个 video query token 在 prefix+video 全序列上 softmax，按 Q128×K128 video tile 汇总各 query 的 key 概率和并对有效 query 求均值；对每个 head 的全局 tile 排序。真实参考二：tile 内有效 token 的 QK logit 最大值，作为先前首步精确路由的近似对照；与 INT8 capture 数值并不完全相同。
- 指标：同预算下 |proxy top-k ∩ oracle top-k| / k；随机匹配预算的期望命中率等于保留比例。每个完整 tile、选定 heads 均计算；边缘有效 token 正确遮罩，prefix 参与真实 softmax 但不参与视频 block 预算。固定 20×5 网格，不按看到的结果挑选。
- 统计：按预设 20 step×5 layer×8 head，先聚合 head/另一轴，对 timestep 和 layer 均计算 Spearman 相关和排列检验；标明排列假设、多个预算多重检验和单轨迹局限，不将 tile/head 伪装成独立 prompt。探索结果不声称总体显著或因果性。
- 决策：若高于随机且预算/深度变化明确，可优先研究路由估计；若明显失配，考虑精确路由与替代评分。不能仅凭本实验证明跨时间复用可行。数据不完整或非有限值时停止并报告。
