# E039：输出通信与原生SVD投影的完整边界

2026-10-03，exploration；执行前固定，尚不构成新方法。问题是前向QKV压缩后仍需回传的BF16 attention输出，能否直接以消费端所需表示传递，并优于合理的常规行并行/FP8路径。

## 假设与决策

同一head-owned O与固定native SVD out_proj下，传下一主GEMM的NVFP4 packet及既有LR-down的32维部分积，可能比回传完整O后投影更便宜。最强替代是源侧主支行并行＋小down部分积归约，目的端才一次LR-up；不能让对照重复up或默认采用昂贵FP32大通信。本次实测所有必要操作，而非用载荷比例或拼接kernel耗时预测。

若合理row-parallel或FP8已取得更好的实际成本/精度取舍，停止候选第四臂，不扩rank/格式/卡数网格。若第四臂取得明确成本收益且保真有价值，继续解释其表示与所有权接口，并补最便宜的侧信息必要性对照；仍须审视成熟融合实现和整模价值。输出误差据实报告，不设逐位一致或任意百分比科学门槛。单层测量不能证明完整模型或视频质量。

## 真实输入

采用E035已保存block0完整native QKV[22592,56,128]，对应同一真实H3 teacher状态的本地native投影；由原BF16 varlen attention助手计算有效段22539和模型padding段53，仅2次SDPA、0DiT，保存完整O。不能把既有E034仅有效O补零当原模型padding：真实53行会参与下一native activation global。

两个head-owner各28heads（3584通道），token-owner实际行数11264/11328，后者含原53行模型padding。W沿输入通道切半但保持原codes、E4M3 SF、global，不重新量化权重。out_proj为5376×7168、rank32、bias=None。BF16 smooth先逐元素舍入，主支仍是原H3 legacy NVFP4 encoder＋native scaled_mm。

源侧编码的global按目的token段×全部7168通道的amax域；两源对两个目的段分别计算partialamax，一次FP32[2] MAX AllReduce。真实modelpadding纳入；只排除额外传输/存储零padding。主GEMM保持actual M=11264/11328，SF物理RP=11264/11392；无效SF为0。RS要求等长槽，两个槽各11392，因此分别另付128/64零行载荷，明确计入。

## 四臂

1. bf16_return：BF16 O split-A2A回到token-owner；原NativeH3Linear完成smooth、pack、主GEMM、BF16 down/up和相加。
2. fp8_return：每source/目的段独立动态E4M3 FP8编码、FP32 decode scale；codes/scale一次coalesced uint8 A2A；目的端解码BF16后走同一原Native投影。它是不同精度选择。
3. row_parallel：源BF16 smooth＋共同目的global编码，每目的段一次half-K native main及BF16 down部分积；main5376与down32拼成一次BF16 ReduceScatter。目的端一次BF16 up和main相加，不重复LR-up。部分主支和down提前舍入、BF16归约与原full-K存在差异，报告它。
4. fp4_side：源同共同global编码；真实主支packet和原BF16输入产生的BF16 down32部分积合并一次A2A，目的端重组物理SF；down两部分以FP32求和再BF16，原完整main_from_packet与一次BF16 up/add。禁止只为第四臂改变down精度以制造保真优势。

前三臂是必要强控制。当前无pack/LR新融合kernel，也不把共同未融合consumer的成本说成优于Nunchaku/FlashInfer。第四臂的侧信息来自既有模型矩阵，而非新低秩拟合；普通分布式恒等式本身不是贡献。

## 执行与证据

GPU0/1，3轮warm＋10轮四臂轮换，共52次完整边界/rank、104次总边界；row每次2主GEMM，因此总130次真实scaled_mm，0DiT/0新增视频。准备另2次BF16 SDPA。计时起于resident head-owned O，终止于token-owned projected BF16 output，包含smooth、统计、pack、wire整理、所有collective、主支与LR计算；排除加载/JIT/保存和计时前barrier。取每次两rank较慢wall，再汇总每臂中位/范围，保留全部原数组；显存峰值及应用层remote字节分开报告。

保存四臂最终输出，独立CPU比较同baseline的有效段与真实modelpadding误差，并核实际collective/call账本。不同算法的float归约不要求byte相等。小CPU布局检查仅保障数据传输，不替代实际native算子或性能。

代码、报告在仓库；大artifact在/data1/models/svdquant-wjq/research/20261003/E039。prepare300秒；主两卡900秒（含必要JIT）；具名tmux、日志、进程组deadline、每启动即时查GPU空闲，失败只清理自身子进程。原E035–38源与结果不改，协议变更另记。参考E039_interface_prior_review.md。
