# 阶段035：输出通信与原生SVD投影

2026-10-03。E039完成，有一个值得继续解释的工程信号，尚无可投稿核心贡献。

把attention的head分片输出送回token分片后，原SVD投影主支需要NVFP4输入，低秩支需要量化前输入。我们比较四种完成整个边界的方式；候选直接发送原主支packet和既有32维LR-down部分积。没有新拟合低秩基，也没有新融合kernel。

| 完整边界实现 | 两卡较慢wall中位数／范围 ms | 相对BF16-return的有效段NMSE |
|---|---:|---:|
| BF16回传＋原native投影 | 6.348／6.318–6.453 | 0（参考） |
| FP8回传、还原后原native投影 | 11.090／11.072–12.748 | 2.33519e−3 |
| 源行并行main/down，一次BF16 RS，目的端up | 8.762／8.583–10.956 | 7.55112e−6 |
| FP4 packet＋down32回传，目的端完整主支/up | **4.897／4.879–5.088** | **5.93241e−8** |

候选比BF16回传快22.85%，十个配对轮次均更快；比该行并行实现快44.11%。原生投影本身已经量化，因此参考不代表全BF16模型。候选输出不是逐位相等：有效段RMSE .003736、最大绝对差2.0。真实53行modelpadding另计，候选NMSE3.06950e−8。

row与候选的down实际张量一致；相对原full-K down的NMSE约1.08e−5。两者最终输出差异与row主支先舍入部分积再归约的路径一致，不能把本例归纳为普遍定理。两种源编码的目的段global均与BF16回传参考相同。

双向远端A2A应用载荷：BF16 154.438、FP8 77.219、候选44.828 MiB；row的逻辑RS载荷235.016 MiB。row/候选另各有逻辑MAX AllReduce共16 bytes；这些不是实测线缆流量。allocated峰值BF16/FP8/row/候选为.87775/.87775/.82168/.64503 GiB，候选比BF16低26.51%；共同驻留了full和half权重，不能当各自最小部署显存。共享allocator的reserved峰值不作节省结论。

实验使用同一真实H3 block0输入，22592行含53行原模型padding；原BF16 attention准备仅2 SDPA。正式双GPU0/1、每臂3 warm＋10轮换重复，共104边界、130 native GEMM、0 DiT。launcher9.23秒，CPU独立归约4.25秒完成；全部worker退出、两卡释放。实际NCCL通道P2P/CUMEM、PCIe NODE，无NVLink。没有视频生成或质量结论。

计时包含当前统计、编码、复制、通信、主投影及低秩计算。FP8是普通PyTorch实现，FP4编码已用Triton，故不能声称优于成熟融合FP8/Nunchaku。普通矩阵重排和低秩侧信息本身亦不是新颖性；近邻审视见[E039接口审视](../00_state/E039_interface_prior_review.md)。

下一步只做同packet、同native main下的LR信息来源控制，检验廉价FP4解码是否已足够，同时核查成熟实现。若侧信息没有实质保真价值，不据此包装新方法；不立即扩卡数、层数或视频网格。

证据：[固定协议](../06_experiments/E039_output_projection_parallel_plan.md)、[两卡原始结果](../../results/research/E039/run_rank0.json)、[rank1](../../results/research/E039/run_rank1.json)、[独立归约](../../results/research/E039/independent_summary.json)。代码与数据路径均保留于结果；大tensor位于/data1/models/svdquant-wjq/research/20261003/E039。
