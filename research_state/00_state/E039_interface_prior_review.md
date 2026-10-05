# E039：输出回传—SVD投影接口的有限审视

2026-10-03；只读本地实现，定向重读两篇原始论文；无GPU、模型加载或实验代码。

**建议做既定一次成本实验，但第四臂先称“消费端所需表示的通信实现”，不称新方法。当前没有核实到近邻完整实现同一合同，也没有足够残余建立新颖性。**

- **CompactFusion**对激活或跨步残差做量化/自适应低秩近似，配合误差反馈；它传的是可重建激活的压缩表示。E039的32维量来自既有模型的固定LR-down权重，是后续算子所需量，而非本次拟合的激活低秩基；不需要跨步缓存。这区分了具体合同，却不使“低位＋小矩阵通信”变新。[§3.1–3.4、Appendix C](https://arxiv.org/html/2507.17511v1)
- **LongLive/Ulysses**已覆盖head/token ownership交换及NVFP4通信。所读LongLive Appendix D式(11)明确是预attention的Q/K/V，未给出输出回传后同时消费原BF16输入LR支的实现；不能从其宽泛“低位通信”直接推定完整覆盖，也不能把换到输出端当贡献。[Appendix C/D](https://arxiv.org/html/2605.18739v2#A4)
- **Nunchaku/FlashInfer**已使低秩与量化/GEMM融合成为强工程先验。本机FlashInfer的`svdquant_linear`实际为smooth+pack、独立BF16 down、融合main/up三个环节，并预折叠smooth到down权重；这不自动保持本机`bf16(O/smooth)`再down的舍入合同。若E039共享未融合consumer，结论只限同实现间的分布式路径，不能归因于新融合或称超越成熟实现。[本地API](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/gemm/gemm_svdquant.py:986)、[当前实际线性合同](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_nvfp4.py:185)

**最重要的强基线细节：让第三臂也在token-owner才做一次LR-up。** 源侧得到native主支部分积与32维down部分积后归约，接收端合并down再乘B，bias只加一次；不要每rank先把down升到完整输出宽度再归约，给第四臂留下人为重复计算优势。本轮采用BF16主支/down部分积拼成一次BF16 ReduceScatter，避免默认FP32传输人为抬高成本；候选采用BF16 down部分积、FP32求和再BF16，二者归约舍入差异如实报告。固定权重列分片直接来自同一旧packet；不能为行并行重新量化W或改变原global。

四臂足够作首次成本决策；**尚不足以单独证明32维侧信息必要。** 最贴近的遗漏数值对照是同一packed主支、LR改吃独立decode的X：它放弃原输入LR合同，却是普通FP4通信的廉价替代。可以沿已有CPU局部LR对照说明其误差，不必现在增加第五计时臂；若侧信息并未换来有意义的既定输出保真，不能声称其必要性。

共同必须说清：outer-global沿目的token分片的全部channels统计；按源head分片直接各取一个amax不是同配方。16-channel groups、smooth的BF16舍入、padding、LR归约dtype/顺序、主支部分积及bias的舍入位置应固定或明确报告差异，**不设逐位科学门槛**。主比较仅完整`head-owned O → token-owned projected output`的同步wall与峰值，含统计/pack/重排/通信/投影；不能用payload比例、分阶段CUDA时间相加预测收益。若普通FP8或合理行并行已占优，停止第四臂；若第四臂更优，留下的是值得解释的具体表示/所有权接口收益，仍非论文贡献或视频质量证据。
