# E038 之后：只保留一个待判别的部署问题

2026-10-03；未改E038、未运行GPU或新实验。重读既有失败/近邻地图，新增仅定向核查LongLive-2.0的SP段落；不凑第二项。

**问题：attention输出回传能否直接进入下一层的原生SVD投影，而无需传完整BF16输出、也不把原高精度低秩分支改成量化输入？** 这是与中心数无关的真实接口：E036/E037共同的逆A2A为154.875MiB，分别占T16/global总A2A应用载荷约48%/54%；前向QKV压缩后它仍保留。仅这些载荷比例不预测时延。[E036](../reports/032_20261003_sequence_parallel_correction.md)、[E037](../../results/research/E037/independent_summary.json)。

机制来自当前代码而非假设业务需求。[NativeH3Linear.forward](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_nvfp4.py:185)先产生BF16 `X=O/smooth`，主支消费`Q4(X)`，rank32分支却消费原`X`：`Linear(Linear(X,A),B)`。仅量化O回传再解码，会同时改变主支编码和原LR输入；不能称“反正下一层也W4A4，所以通信量化免费”。按heads分片时，smooth与16-channel分组可以本地处理，LR-down则满足实数恒等式`XAᵀ=Σp Xp Apᵀ`。因此一个待验证接口是传下一GEMM的真实packed输入，以及原输入产生的32维部分积，再在token-owner合并。具体合同包括：native activation outer-global原本沿目的token分片的全部channels统计，不能直接换成源head分片的全tokens amax；smooth必须保留逐元素BF16舍入，再进入编码/down；分片LR-down归约又改变舍入位置。另须保持物理SF重组和原模型padding，**不承诺原输出逐位相同**，也不将舍入重新包装成native/QDQ新机制。

最强替代排在构想前面：**①源侧完成整个out_proj的行并行部分，再ReduceScatter恢复token ownership：总FLOPs相同而矩阵维度分配不同，可能根本无需回传O。** 它同样须交代outer-global计算域和部分积/epilogue舍入，不能因实数等价就假定native逐位等价。②普通FP8逆A2A，接收后走原完整投影；③已有Nunchaku/FlashInfer的pack＋LR-down融合，避免用未融合实现虚构候选收益。CompactFusion已覆盖低位/低秩通信与误差反馈，低维side information和分布式matmul本身不新。[既有系统近邻](../01_literature/native_systems_frontier.md)。新核的[LongLive-2.0 Appendix D，式(11)](https://arxiv.org/html/2605.18739v2#A4)明确限定预attention交换的集合为Q/K/V；该段没有直接给出上述输出回传与原输入LR分支合同。这只是此次来源未直接覆盖，绝非新颖性证明；不能把“已知融合＋低位通信”直接报为方法。

**最便宜的判别证据**先取一份已保存E034/E036实际attention输出及E009 block0 out_proj权重，在CPU比较原LR分支、改吃解码FP4输入的LR分支、按head分片后合并原输入down部分积；同时只列实际格式所需载荷及必须新增的统计/重组。它回答能否保留原高精度信息、简单量化回传改变多少既定投影输出，不回答视频质量，也不以任意百分比或byte一致设科研门槛。只有留下可用接口，后续才值得一次完整两卡`head-owned O → projected token-owned output`测量，对照上述BF16/FP8/常规行并行强路径，计入所有pack、down、通信、合并和投影。若常规替代已更优或新增同步/重组吃掉收益，停止此候选，不扩大rank/格式网格。

它不依赖coarse16赢得E038：目标是减少既定部署配方的通信代价，质量评价不能成为所有同合同工程工作的通用前置门槛。但E038仍决定当前H3配方是否值得作为实际产品参照；若完整生成并不可靠，就优先处理这个事实，不靠另一个单层加速掩盖它。当前建议仅是一个有实测成本来源、尚未验证的备选问题，没有第二个同等证据强度的架构/serving候选。
