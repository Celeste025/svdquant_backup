# Native NVFP4 视频 DiT：系统方向的有限先验核查

核查时间：2026-10-02。范围：官方论文、作者仓库和官方框架源码；本轮未运行 GPU，未修改实验代码。使用 claim-prior-work-triangulation 的碰撞原则。网页 main 分支只代表访问时状态，未视为本机已经安装或可在 SM120 运行。

**结论：目前没有足够证据提出新的系统方法。** NVFP4 视频部署、低秩分支融合、量化通信、稀疏 attention 与并行调度均已有强覆盖。只留下两个需要真实 profile 决定是否值得继续的问题；它们不是论文 claim，也不是推荐立即写新 kernel。E006 的停止决定保持：dense attention 中较大的 interaction norm 并未构成有害耦合，不能换一个系统术语重新包装。

## 已覆盖的路线与最近实现

| 已核实 primary source | 实际已做什么 | 对我们的约束 |
|---|---|---|
| [SVDQuant / Nunchaku，ICLR 2025](https://arxiv.org/abs/2411.05007)；[作者实现](https://github.com/nunchux-ai/nunchaku) | 将低秩 down 与 activation quantization、up 与低位 GEMM 融合，专门解决低秩分支额外数据搬运 | “rank 很小但开销很大”、共享 QKV down、融合低秩分支均非新问题 |
| [LongLive-2.0，v1 2026-05-18 / v2 05-19](https://arxiv.org/html/2605.18739v2)；[作者代码](https://github.com/NVlabs/LongLive) | 视频 W4A4 NVFP4、低秩/LoRA 路径、chunkwise NVFP4 KV、异步 VAE；Appendix D 将 runtime Q 与已有压缩 KV 用于低精度 All-to-All | “FP4 计算 + FP4 通信 + 视频”已做。AR 缓存和本项目双向 rCM 的区别只是 setting，不能独立撑贡献 |
| [CompactFusion，2025-07，正文 §3 / Appendix C](https://arxiv.org/html/2507.17511v1) | 对跨步通信 residual 做压缩与 error feedback，支持低位、稀疏、低秩；量化低秩因子换取更多子空间覆盖，并隐藏压缩成本 | 通信 delta、error feedback、低秩再 INT4 都高度重叠；其 28/50 步设置不能直接证明 4 步收益 |
| [StreamFusion，2026-01](https://arxiv.org/html/2601.20273v1) | 拓扑感知 SP、Torus Attention、单边通信与计算重叠 | “NVFP4 变快后通信暴露”本身只是已有系统问题的新速度区间 |
| [SparSP，2026-09-26](https://arxiv.org/html/2609.32197v1) | 面向 PCIe 视频稀疏 SP，按时空依赖放置 blocks、按需求直送 KV、将传输执行解耦；其排列保留 native KV block 成员 | 稀疏率不等于通信量、按稀疏 mask 路由、重排改善局部性已做 |
| [FVAttn，2026-07-17](https://arxiv.org/html/2607.16190v1) | Top-p 稀疏造成 head/rank straggler；按实际 mask 做 P2P head 迁移，再用非关键 rank 的 slack 增加有效 blocks；测试含 step-distilled Wan2.2 | 泛泛稀疏负载平衡及“短步下动态 workload”不足以区分 |
| [Sol-Attn，2026-07-27](https://arxiv.org/html/2607.24027v1)；[SLA2，2026-02-13](https://arxiv.org/html/2602.12675v1) | 前者在线路由、动态预算与遗漏块近似纠正；后者可学习路由 + sparse/linear 分解 + 低比特 attention QAT | 路由开销、稀疏误差补偿、量化与稀疏联合均拥挤；不能重新命名为新组合 |
| [xDiT 当前代码/发布记录](https://github.com/xdit-project/xDiT/releases)；[通信参数源码](https://github.com/xdit-project/xDiT/blob/main/xfuser/config/args.py) | 已集成 NVFP4 GEMM、Blackwell FP4 attention、FP8 Ulysses All-to-All；有通信 scale 与 safety factor、混合精度时步调度 | 先与现成路径比。这里核实的是代码/发布记录，未验证其 SM120 兼容、质量与实际性能 |
| [FlashInfer SVDQuant 源码](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_svdquant.py)；[SM120 attention 源码](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py) | 已有 SM120 native low-rank/FP4 linear 与外部 packed QKV 接口；本机 0.7.0.post1 数值契约见此前专门审计 | baseline 中未融合的额外开销必须标为实现差距，不能归因于 NVFP4 的固有限制 |

另有两项直接工程碰撞：[PyTorch/TorchAO 官方 2026-04-08 报告](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)已明确识别小矩阵 activation packing 不划算与 NVFP4 下 CPU launch 开销，并使用 selective quantization/CUDA Graphs；[vLLM-Omni 官方 H3 2026-09-01 报告](https://vllm-project.github.io/2026/09/01/minimax-h3-production-serving.html)已区分 SVDQuant loader 正确性和 native fused 性能，报告 encoder/DiT/VAE 分离及量化/稀疏 attention 路径。普通完整部署和公平 benchmark 是必做基础，不自动产生研究贡献。

## 问题 N1：跨分片边界能否继续使用同一份原生低位表示？

**状态：事实部分已核实；瓶颈与剩余贡献 unverified，已有工作覆盖很重。**

具体疑问不是“能否传 FP4”，而是：在完整 video DiT 的算子边界上，分片前的 packed codes/scales 是否能直接成为分片后的消费格式；若不能，保持既定量化配方到底需要几次全局统计、重新编码、转置/重排和高精度纠正？NVFP4 算术变快后，这些依赖是否成为关键路径，而非数据 payload 本身？

**事实边界。** 本项目 native linear 使用 tensor global + 16 元素局部 E4M3 scales；本机 FlashInfer attention 使用另一套契约，包括序列上的 K centering、Q block centering 与 correction，不能把 linear packed activation 直接当 attention packed QKV。对 token 分片，16-channel 分组并未天然跨 rank；真正可能跨分片的是 tensor global、序列统计、边界 block 与布局。K 的统一平移在精确 softmax 下不改变输出，但各 shard 独立减不同均值不具备同样的不变性，不能仅凭此省略处理。[官方 attention 源码](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py)

**致命 prior work。** LongLive 已在低位空间做 SP 交换；Transformer Engine 当前 [NVFP4Tensor/Quantizer 源码](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/nvfp4_tensor.py)已有 amax reduction group、row-scaled NVFP4、FSDP gathering 时的 metadata/scale 处理，且明确某些 GEMM-swizzled scale 路径不支持 FSDP2。该 FSDP 权重路径不等于视频动态 activation SP，但已覆盖“尺度和布局需要参与分布式契约”的基本思想。

**最便宜的测量。** 在整模 baseline 通过后，只取真实 q/fc2 activation 与 norm/RoPE 后 QKV 的少量层/时步。先在单卡做虚拟 P=1/2/4 的 shard→pack→重组，对照 full-pack→分片；固定逻辑 token 顺序、有效长度、分组成员和 correction 语义。记录码值/重建差异及真正新增的统计/转置/encode 次数。随后仅对实际存在的非局部依赖做两卡一次完整边界计时，包含 pre/post pack、metadata、通信与纠正，不先搭整套分布式 serving。报告关键路径，不能把有重叠的 kernel 时间相加。

**Kill 条件。** 若保持 block/scale ownership 即可直接交换，额外预处理在可复现 trace 中不足总 DiT 时间的 5%，或用现有 TE/xDiT/LongLive 的常规布局即可消除，就停止。码值变化本身不是贡献；只有它同时强迫有意义的质量/关键路径代价，才值得进一步查新。5% 是本轮投入门槛，不是统计显著性标准。若多卡部署并不需要此边界，不做这个实验。

**当前缺证据。** 没有两卡 trace；没有证据证明原生 NVFP4 存在无法被已有 per-row/per-shard scale 解决的障碍；没有证据说明这种处理能带来方法创新。不能声称所有 NVFP4 都需要 per-forward amax AllReduce，更不能重提已经被弱化的“全 tensor scale 导致 batch 灾难”。

## 问题 N2：量化是否改变稀疏路由的工作图，导致低位数值误差变成系统成本？

**状态：仅是机制假设；不是 E006 正结果的延伸。**

已有 sparse-SP 调度通常以已生成的 mask 为输入。窄问题是：真实 W4A4 projection 的误差，会不会通过 top-p/阈值这种离散决策，改变每个 rank 的 active blocks、remote-KV demand union 或重排工作，从而使平均稀疏率看似相同，却出现关键路径/通信开销不同？这里比较的是工作图与实际时间，不是简单声称量化和稀疏误差相乘。

**致命 prior work。** FVAttn 已按实际 mask 迁移重 head；SparSP 已优化 remote demand；Sol-Attn 已控制动态路由预算；SLA2 已联合 learned routing 和低比特 QAT。因此“加安全 top-k”“mask 不稳定”“量化后重新调阈值”都很薄。剩余问题只可能是：投影量化产生了可重复、不能被这些既有机制消除的成本变化，并能解释其来源。本轮未核实这种剩余事实。

**进入门槛与一个实验。** 首先完成 native Wan trace。只有 self-attention 仍占显著比例、且现成 sparse backend 可在预算内调用，才做小规模重放；不为这一假设新写 sparse kernel。取同一 teacher 输入的 BF16/W4A4 QKV，用同一个官方 router 得到 M0/M1。固定 W4A4 QKV 与相同 sparse kernel，分别消费 M0 和 M1，以此只改变路由；另留 BF16+M0 为参考。先统计 active-block 数、head/rank 最大负载、remote-KV union、路由时间和完整 attention 时间，报告总预算与分布，不能只报 mask Jaccard。若 block 数不同，先显式解释数量效应，再做一次匹配预算重放；不反复搜索阈值。E006 表明 dense attention 的量化交互不足立题，这个实验只有离散工作图本身出现成本效应才有独立信息。

**Kill 条件。** 在预先指定的少量层/时步中，工作图变化不能导致稳定的至少 10% attention 完整路径成本差异，或只是 active-block 数变化、现成 FVAttn 调度/标准阈值校准可恢复，就停止系统方向；不能改报输出误差来回避失败。若 attention 占比为 f，则候选整模收益上限约受 f 约束；即便 attention 变化 10%，整模只有极小收益也不扩样。10% 同样是筛选门槛，未代表已经测得效果。

**当前缺证据。** 还没有在真实 native projection 后运行 sparse router，没有 mask-level 成本数据。固定 top-k 可能根本不改变 workload；top-p 的细微选择差异也可能被大量 heads 平均掉。即使测到成本改变，也仍需说明为何比现有稀疏调度/预算控制更有辨识力。此处不作 novelty 断言。

## 当前执行顺序与停止纪律

1. 先交付同环境、同 attention 的 BF16/QDQ/native Wan 完整 DiT profile，计入 smoothing、动态 scale、pack、原生 GEMM、BF16 LR、norm/attention 与临时显存。
2. LR/down 重复、缺失 fusion、Python dispatch 或 packer 很慢时，优先认定为 baseline 工程缺口；与现成 fused 路径建立下界后再解释。测得“rank32 占很多时间”不足以创建第三个题目。
3. N1 仅在确有多卡部署边界时触发；N2 仅在 attention 确实主导且已有 backend 时触发。两者都不触发，则本轮结论是暂无可信系统研究题目，保留完整部署成果，不凑第三项。

## E007 补充：同码值 fake-quant / native 算术分歧的最近先验

2026-10-02，限时查新；只补解释边界，不建立新 claim。输入事实来自主 agent 的 E007 汇总：300 份 weight roundtrip exact、完整 bypass exact、真实 activation packer byte-exact；已测单层 native GEMM 对 BF16-QDQ 的 NMSE 约 5e-6～1.3e-5，30-block DiT endpoint native/QDQ 为 4.28%，二者各自对 BF16 为 16.40% / 16.19%。本节没有复跑或独立重算这些数字。

**核查结果：在本次有限搜索内，没有核实到专门隔离“相同 NVFP4 codes/scales、BF16 反量化算术与原生 tensor-core 算术，并追踪视频扩散完整网络分歧”的直接研究。不能据此声称没人做过。** 下面四份 primary sources 最接近；其中两份是通用工程契约，不是扩散传播论文。

| 来源与已读范围 | 直接支持的事实 | 与 E007 的距离 |
|---|---|---|
| [NVIDIA TensorRT：Working with Quantized Types，explicit Q/DQ 段落](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) | 从 fake-quant 导出的 Q/DQ 保持量化语义，但优化可能改变浮点运算顺序，结果不保证逐位相同 | 最直接的官方先验。“码值相同仍可能有不同输出”本身不是新发现；文档没有给 NVFP4 扩散整模分歧的大小或方向 |
| [Asaria 等：Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs，2026-06-12，§3/§5/§6](https://arxiv.org/html/2606.14598v1) | 在 Ideogram 4.0 中将 dequant-to-BF16 路径替换成原生 INT8；分别报告整数累加正确性、GEMM 和端到端指标 | 最接近的 diffusion 部署对照。INT8→INT32 不是 NVFP4 分组浮点累加；kernel cosine=1 也不等于 fake/native 完整网络逐位相同。质量证据仅 4 prompts 点估计，作者未重测 OCR，不能拿来保证本项目质量 |
| [Yang 等：An Empirical Study of Microscaling Formats for Low-Precision LLM Training，ARITH 2025，§II-A，PDF p2](https://aisystemcodesign.github.io/papers/FP4.pdf) | 使用 BF16 fake-quant 模拟 MX，并报告获得 B200 后，所测模拟 GEMM 与真实 MX 结果相同 | 是重要的相反例子：fake/native 差异不是所有 4-bit/microscaling 的必然属性。该工作采用 MX 的 E8M0 二次幂 scales，不能外推成 NVFP4 两级 scale 的严格等价保证 |
| [PyTorch 官方 Numerical accuracy，BF16 GEMM reduced-precision reduction 段落](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) | GEMM 的输入/输出 dtype 不完整规定内部累加；部分实现会截断中间 reduction，BF16 也有控制开关 | 支持记录 backend、reduction 配置和舍入位置。它没有证明 E007 使用了某种截断，更不能据此宣称 native kernel 错误 |

**可保留的解释价值是执行契约审计，而非质量或新量化机制结论。**

- 三种一致性应分开：packed representation 一致；给定同一输入的局部 GEMM 近似一致；完整网络自由前向的 endpoint 一致。前两项不能自动推出第三项。相同 activation recipe 也不等于两个自由前向在后续层仍有相同 activation codes，因为上游小差异会改变下游输入。
- 一个具体、尚未归因的差别是 scale 与 BF16 舍入的位置。软件路径先构造 `bf16(gA*sA*qA)` 和 `bf16(gW*sW*qW)` 再做 BF16 GEMM；原生路径消费 codes 与 block scales，并按其 kernel 契约应用 global alpha。即便保存 weight 的 BF16 roundtrip 完全一致，这两条有限精度算式也不因实数代数等价而自动相同。E007 当前尚不能把全部 4.28% 分歧唯一归因于 global scale、累加顺序或某个 epilogue；这些是候选来源。
- 此处的 4.28% 是 **30 个 blocks 的单次 DiT 输出分歧**，不是已经验证的多 denoising 步误差累积，更不是视频质量下降。16.40% 与 16.19% 接近，只表明这次 endpoint 上二者对 BF16 的距离接近；不能推导 native 更差/更好或同等生成质量。
- 工程上的结论已经成立：今后比较校准方案，不能把 BF16-QDQ 的输出当作 native NVFP4 的逐位 oracle；应分别交代表示验证、局部数值容差与真实部署后的整模/生成评估。研究上的剩余证据还不够：没有跨输入/层位的归因，没有原生数学参考分解，没有因算术差异而产生的可重复质量排序变化。**不新增论文候选。**

后续定向补查见 [native_surrogate_collision.md](native_surrogate_collision.md)：QUADS 已直接讨论跨引擎 BF16 漂移与 activation QDQ 对齐不足；NVIDIA QAD 和官方 fused SVDQuant 实现进一步约束了剩余解释。宽泛的 native-surrogate mismatch 不能作为新 claim；无 global 的 E2M1×E4M3 可以精确表示于 BF16，不应把输入额外舍入误归因于该乘积本身。

## E009后补充：RoPE公共相位候选暂不投入实验

2026-10-02，root基于attention瓶颈检查了共同Q/K相位旋转：实数点积不变，但低位舍入改变。该宽泛对称性已被[GaugeQuant §3.1](https://arxiv.org/html/2607.20757v1)明确写成与RoPE交换的逐pair平面旋转；作者未利用这个较小自由度。其未实现不等于我们的候选已具有贡献。一般post-RoPE等价变换和量化友好变换也已有QuaRot/SpinQuant、[ReQAT Q-FIT](https://aiha-lab.github.io/ReQAT/)等近邻。没有证据证明该自由度在NVFP4视频中形成稳定、有实用价值的特殊问题；只换成video或搜索phase不足以立题。park、不新增claim、不跑GPU。
