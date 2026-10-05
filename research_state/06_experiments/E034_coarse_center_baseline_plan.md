# E034：先补同容量的连续粗粒度中心强基线

2026-10-03，GPU前固定。E033完整consumer已经完成110.37秒/478native/0DiT；5组新旧控制输出一致。K16约31.5ms/2.755GiB，对比freeblock31.7ms/3.518GiB；精度损失与E032一致。该点未被已有四臂直接支配，但不足以证明聚类必要性。下一不扩视频、prompt或K网格，先补最直接的成熟粒度基线。

固定同一E018 p36/s14三层0/24/48，N22539、Np22656、G177、H56、D128。coarse16将原连续Qtile均匀分16组，边界b_j=floor(j*177/16)，j=0..16。第j个中心直接对BF16 padded Q[:,128*b_j:128*b_{j+1}]求mean，结果BF16；不是先均值再量化的近似，也不是从结果选择边界。末组包含原协议的padding，正如官方block/global中心定义；K仍先对valid tokens减BF16 globalmean再pad。每Qtile固定id=j，Q−C[id]以BF16计算，T=C.float@Kc.floatᵀ，使用已验证E033共享行consumer。16个table rows与codebook相同，但不计算unused全局/自由块Q均值，不跑聚类。

四臂global/freeblock/codebook/coarse16，同进程、同resident HND BF16 QKV，各3warm＋10repeat，三层共156native、0DiT。全部Q/K/V各一次pack，含中心/在线6轮聚类（只codebook）、必要pad/复制、FP32 K转置及T计算、native attention、valid NHD输出拼装。CPU上传、哈希/文件、输出误差在计时外。每arm独立清cache，记录resident与warm baseline、逐repeat event/synchronizedwall/allocated及reserved peak增量、全实际计数。已有private源/旧实验不改。

固定geometry边界/id预先在CPU构造并上传，约39.6KB，与聚类weights一样计入所有arm的resident baseline；所有中心仍每repeat重新计算，最后测量返回的C/id在计时外保存。不要用floor(g*16/177)替代按上述边界填id。

不重跑E033的10个消费者控制（无新kernel/packet ABI）；保存所有12个最终输出，既有三臂与其E033结果作数值报告，coarse16对BF16及block/global/codebook独立比较，不对科学误差设任意门槛。CPU只核边界覆盖/顺序/ID与16行shape、源输入、CUDA未初始化。小fixture不要求跨归约顺序bit相等。

GPU0单一root命名tmux，900秒硬预算，重新检查空闲；数据/输出/cache在DATA1。结果回答聚类是否超过同容量连续粗中心的实际精度/时延/显存取舍。经典分组/聚类不是新算法；若coarse16支配codebook，停止聚类路线。如果互不支配，也不靠一个额外点立论文claim，之后再决定是否值得最小跨状态检验。当前只准备，尚未GPU。
