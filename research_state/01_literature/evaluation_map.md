# Evaluation Map

## 2026-10-04：MJ-VIDEO 的使用边界（D076）

用户指出此前汇报频繁依赖 MJ 总分。本轮核对论文、实际 evaluator 和已保存结果；只读 CPU 核查，没有重新评分或启动 GPU 实验。MJ 可保留为辅助偏好读出，不能单独裁决量化方法有效性、细节修复或等质量加速。此前数值保留，但“分数上升”只表示该模型输出上升。

**论文支持的范围。** [MJ-VIDEO 原文 v3](https://arxiv.org/html/2502.01719v3#S4.T2) 表2报告在 MJ-Bench-Video / Safesora-test / GenAI-Bench 的整体成对偏好准确率为 68.75% / 64.16% / 70.28%。表1的细节、连贯性成对偏好 strict 准确率为 58.82% / 58.46%；不能把另一项连贯性分类准确率95.36%当成偏好准确率。这些结果不是单例置信度，也未验证 Wan NVFP4 细微伪影的检测能力。论文2.2.2还过滤过于相似的视频对，其验证结论不能直接外推到差异很小的量化配对。

**本地实际实现。** E051 调用 [eval_mjvideo_e021.py](../../scripts/research/eval_mjvideo_e021.py)，加载 `/data1/models/svdquant-wjq/models/MJ-VIDEO-2B`，直接保存 `out.score`、五方面、28细项。

- 总分由官方 `scripts/model/moe_reward.py:253–277` 两级 learned gating 计算，含 alignment、safety、fineness、coherence_consistency、bias_fairness 全五方面。旧 scope 中“safety/bias remain raw only”仅指未单列为汇报指标/门槛，**不代表它们从总分中排除**。没有另造质量复合分数；现存 JSON 未保存 gating，不能精确归因它们对分数变化的贡献。
- E049/E051 实际帧索引 `[0,10,20,30,40,50,60,70]`，81帧中71–80未抽到。官方 `scripts/data_processor/data.py` 使用 `endpoint=False`；`max_num=1` 将832×480整帧非等比 resize 到448×448，单patch/帧。相同预处理保证协议一致，但不能保证短暂接触、局部闪烁、细小伪影和末尾动作被看见。
- 输出不是概率或有固定零点/比例意义的画质量表；`.851667→.755266` 是原始 reward 变化，不能翻译成画质降低多少百分比，也没有已验证的通用显著差异阈值。
- E051 喷水r0：MJ `.997390151→1.002087474`，而 RAFT dynamic `true→false`。这只说明代理读出有分歧，既不证明 MJ 错，也不证明视频静止。AMT、RAFT、DINO 同样不能充当动作/物理真值。

**后续原则。** 新尝试先写具体待验证损伤和可证伪判断，再选择对应证据；不按 MJ 总分的小幅涨跌单独接受/淘汰方向。当前四提示×两seed是反复查看的诊断集。关键质量结论需要预先固定的独立提示/seed、随机左右顺序且隐藏方法身份的完整视频成对人工评价，允许平局/无法判断；细节损伤与动作完成分别记录。当前九帧非盲预览不构成人工研究，独立人评和新增评估尚未执行。自动指标提供补充，必要时检查抽帧稳定性；报告逐提示效应和以提示为单位的不确定性，不把帧数当独立样本量。MJ在本地量化样本上的排序可靠性尚待验证。不通过事后改权重、挑样本或新造总分修饰结果。

证据：[E051 MJ原始输出](../../results/research/E051/mjvideo_scores.json)、[E049 MJ原始输出](../../results/research/E049/mjvideo_scores.json)、[E051摘要](../../results/research/E051/evaluation_summary.json)。原始实验代码、协议和结果未修改。

## 此前 E048 阶段记录（历史）

