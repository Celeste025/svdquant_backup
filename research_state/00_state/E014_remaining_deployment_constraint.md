# E014 后的一个部署约束：主支压缩后，长序列 attention 仍占主要时间

2026-10-02，systems 独立只读核对。没有启动 GPU、改冻结源或生成新方法结论。

**先确认对照有效。** plain 导出 200/200 原 W 的 SHA 与 E009 独立保存的原 W SHA 逐层一致；全部 packet decode 与 common `nvfp4_qdq(original W)` 数值 exact，仅允许零符号差。首次 plain evaluate 的三个固定输入均为 200 FP4 GEMM、102 BF16 SDPA、零 disk load；200 次 fastpack checks 均 invalid=0，SF0 受影响调用分别 1/2/5 次并有索引。非目标 tensor identity 保持。SVD 与 plain 的常驻存储差恰为 301,414,400 bytes，与删除 200 层 smooth 和 rank32 A/B 的逐 shape 推导相等，没有残存原 BF16 主层权重的迹象。

同 GPU5、独立进程、每臂 1 warmup + 3 repeats 的完整 resident DiT host 中位数（含 flag 检查）为 BF16 **8.291513 s**、SVD **6.804377 s**、plain **5.823077 s**。plain 三次范围 5.809770–5.833834 s；常驻 11.665507 GiB、steady peak allocated 15.059898 GiB。该结果只是固定状态的速度与内存；三个状态的 video/audio 误差有取舍，不能宣布 plain 质量更优。

**唯一建议继续定位的约束，是低精度主投影与长序列 attention 之间的成本失衡。** 独立读取 plain Chrome trace、核 SHA、只计 `cat=kernel` 后：

|实际 kernel 类别|调用数|累计时间|占 kernel work|
|---|---:|---:|---:|
|PyTorch BF16 FlashAttention|102|3.415213 s|58.697%|
|SM120 原生 FP4 GEMM|200|1.007787 s|17.321%|
|已有在线 H3 pack 三个 kernel|600|0.202202 s|3.475%|
|其余，包括原模型 norm、RoPE、调制、拷贝及未量化算子|—|1.193169 s|20.507%|

这些互斥 work 总和为 5.818371 s，符合原 trace 总和；它们不是严格 critical-path 比例，不能机械转成预测加速。没有把 CPU/GPU annotations、copy API 的阻塞等待再加一遍。本轮未加 E009 semantic labels，因此 raw `categories=other` 不表示全部未识别计算；上表依据实际 kernel 名单独归类。packing 已不是主要成本，继续只优化主 GEMM 或低秩分支不会回答这个主要剩余问题。

**这还不是研究机会成立的证据。** 当前 attention 已是实际 Flash kernel，并非 Python attention 逐元素模拟；但这不证明 BF16 是必要精度，也不证明现有 FlashInfer/Sage 类成熟方案无效。E006 仅做局部组合误差筛查，不能替代当前 legacy projection 配方下的完整速度/内存/输出对照。原 `_sdpa_varlen_attention` 的 `cu_seqlens.tolist()` 和 RoPE/调制产生的中间张量属于可常规优化的实现选择，不能宣称模型或 SM120 固有限制。

最有决策价值的下一项成熟对照，是在同一 E014 状态、同一 plain 或 SVD 投影臂中，仅替换为**已经安装且实际运行过的 SM120 NVFP4 attention**，把该完整路径的延迟、峰值内存和两模态输出误差并列。其默认 `per_block_mean=True` 的 FP32 Q-mean correction 会在此 22400-token 形状物化约 0.818 GiB，长度翻倍时该项按源码变为约 3.271 GiB；这是源码推导，尚未测长序列峰值。官方已有 `per_block_mean=False` 线性空间选项，必须列为成熟 baseline，而不是为默认二次缓冲区重新提出方法。先查这两种已存在的路径能否在完整模型给出有价值取舍；若能覆盖问题，就只是部署基线补全。若在相同自然显存预算内出现经过完整对照仍无法消除的精度/容量取舍，才有理由重新界定研究对象。目前无新 claim。

来源绑定：`results/research/E014/export.json` SHA `a9af02fe2c23d1639aa79971c295da4fce04baabb2f5ff0ca3e79c2a4a9c7719`；plain evaluate SHA `7c16721f5c411c45196ecfee652bf0735588c00f85e4d2797bb156cf16a26c3a`；plain bench SHA `3c87faae24f22ebaa8d617a392583c97b5b1fed9c839cac6626bdfbced007570`；plain trace `/data1/models/svdquant-wjq/research/20261002/E014/bench/plain.trace.json` SHA `93c32fd05512a6f288b7812b08fbf9130e161091142e31e0bcb24a7628ae904b`。FlashInfer 源码与容量推导见 `phase4_systems_signals.md`，没有新增未验证接口假设。
