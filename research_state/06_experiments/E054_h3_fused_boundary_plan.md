# E054：H3 通信表示收益能否经受融合强对照

> 执行状态补充：本计划停在准备状态，未执行CPU结构检查或GPU运行。用户强调普通工程优化不能作为创新后，root优先执行E055机制筛查；本计划不自动启动。

2026-10-04；exploration / decision-value。用户恢复研究并指定MiniMax-H3为主实验、Wan1.3B仅快速验证；Wan14B完整PTQ不启动。本计划在新GPU结果出现前固定。

**动机与问题：** E039同一真实H3边界的FP4 packet＋down32比BF16回传后投影快22.85%，但共用consumer未融合、FP8编码为普通PyTorch。最关键竞争解释是收益来自实现成熟度差异。此轮只判断工程信号是否还值得继续研究，不将旧C006恢复为已成立论文主线。

**假设：** 在三臂共用成熟FP4 main＋LR-up融合算子，且普通FP8对照也融合编解码后，传递consumer所需FP4表示与down32仍有稳定完整边界延迟优势。反假设：强对照消除或反转收益。

**最小有效设置：** 沿用E039已有H3 block0真实attention输出与原导出权重，0DiT/0attention/0视频。22592真实行含53原modelpadding；56heads×128、输入7168、输出5376、rank32；物理GPU0/1。沿用目的token分片/原NVFP4两级scale域，不重新量化W；模型padding保留。旧输入 `/data1/models/svdquant-wjq/research/20261003/E039/attention_output.pt`、权重 `.../20261002/E009/legacy_export/layers/blocks.0.attn.out_proj.pt`。

**三臂：**

1. BF16-return：BF16回传，原smooth/legacy pack和原BF16 down；成熟FlashInfer `mm_nvfp4_svdquant`融合main/up。
2. FP8-return-compiled：保持E039每source/目的message单E4M3 scale；用torch.compile融合amax/encode及接收端decode/拼接/smooth，不更换为另一MXFP8量化方案；随后与第一臂相同pack/down/fused main-up。
3. FP4-side-fused：按E039共同目的global生成原packet与BF16 down部分积，A2A后重组，目的端同一fused main-up。

动态alpha=gA*gW以及BF16(B/alpha)重算全部计时；承认其LR舍入/epilogue顺序与旧torch consumer不同，报告同轮各臂输出差及相对旧E039数值，不要求跨实现逐byte一致。尽量复用已有编码/布局，不为候选新写kernel。编译必须fullgraph且无静默eager fallback；保存编译/实际GPU kernel证据。若接口无法正常执行，记录实现失败，不当作科学阴性。

**计时与预算：** 3次warmup＋10轮三臂轮换重复；每次取两rank较慢同步wall，保存全部数组与中位/范围及配对差。计时始于resident head-owned O，终于token-owned BF16 projected output，含smooth、scale归约、编码、拼接、通信、down、alpha/rescale及fused GEMM；检查/写盘在计时外。首次JIT只在预热/准备阶段、单独记录，但计入整个作业1200秒总上限。记录resident/peak allocated与通信实际应用载荷，不将payload直接当线缆流量或整模速度。named tmux、GPU即时空闲检查、只终止自身进程组；不自动重试，不延长原deadline。

**决策规则：** 若相对BF16或融合FP8不再有稳定优势，停止将E039局部22.85%作为系统路线依据。若优势保留，只允许下一步评估完整DiT关键路径价值和合理行并行强对照；本轮三臂不是穷尽最优系统，不宣称超越所有成熟实现，更无等质量结论。不改rank/shape/样例网格来挽救阴性。科学质量、更多卡数和生成视频暂不属于必要证据。

结果放 `results/research/E054`，大张量放 `/data1/models/svdquant-wjq/research/20261004/E054`，原E039/E040结果及代码不覆盖。
