# 第一轮文献索引：视频生成 NVFP4 W4A4

更新：2026-10-02。目的：建立能否决重复 idea 的工作地图，非穷尽综述。下列速度、质量均是作者报告，未在本机复现。只使用论文原文、作者项目页或官方代码/发布；搜索聚合页仅用于发现。

核实等级：A = 读到方法与实验协议的相关原文；B = 官方摘要/项目页，未完成全文核实。预印本不等于已被同行评审。按技术碰撞优先级排列，非质量排名。

| ID | 论文、日期、直接来源 | 已核实技术动作与边界 | 对本研究的作用 / 核实程度 |
|---|---|---|---|
| P01 | [SVDQuant](https://arxiv.org/html/2411.05007v4)，2024-11 初稿，ICLR 2025；读取 v4 2025-11-08；[Nunchaku](https://github.com/mit-han-lab/nunchaku) | activation smoothing 后以高精度低秩权重分支吸收困难结构，4-bit residual；Nunchaku 融合分支以消除额外访存。主质量验证为图像模型。 | 基础算法和真实速度基线；“低秩补偿+量化”已占据。A：§3–5。 |
| P02 | [ViDiT-Q](https://arxiv.org/html/2406.02540v3)，2024-06 初稿，ICLR 2025 | DiT 的 channel/token/timestep 异质性；静态/动态组合与混合精度。成功区主要 W8A8/W4A8，并报告 GPU 实速。 | 非平稳 activation 不是新发现；应保留高质量 INT8 对照。A：摘要与全文结构，细节尚需按候选 claim 补读。 |
| P03 | [DeltaQuant](https://hanlab.mit.edu/projects/deltaquant)，CVPR 2026，2026-06 | 局部 3D cube 均值 core token 保留 FP8，delta token FP4，权重沿用 SVDQuant 低秩。验证 Wan2.2 / LTX-Video。 | 最危险的时空均值/残差碰撞。111.8× 含其他加速，量化增量为作者报告 3.0×。B：官方项目页、CVF 摘要；PDF访问失败，不能声称已审计 kernel。 |
| P04 | [6Bit-Diffusion](https://arxiv.org/html/2603.18742v1)，2026-03-19 | 用 block 输入输出变化预测 NVFP4/INT8 精度；Temporal Delta Cache 和 outlier cache purification；实验证据为 CogVideoX-2B/5B、50步 DDIM、RTX5090。 | “随步混合精度+缓存”已有；不能将结果直接外推到 Wan/H3/4步。A：§4–5。 |
| P05 | [PulseQuant](https://arxiv.org/html/2609.33384v1)，2026-09-27；官方 abs 显示 v2 2026-09-29 | isolated block-step 扰动后 dense continuation 测传播风险；风险加权校准和响应子空间 code edits。涵盖 Wan/Self Forcing/H3。质量表在 H200 Q/DQ；另在 RTX5080 测 packed NVFP4。 | “局部重建不等于最终损伤”及 pulse 实验已直接做过；必须核对 codebook-Q/DQ 到 native NVFP4 的质量保真。A：§4、5.1、附录目录；未审代码。 |
| P06 | [AccuQuant](https://arxiv.org/html/2510.20348v1)，2025-10-23；[NeurIPS 2025原文](https://papers.nips.cc/paper_files/paper/2025/file/9f8265edac41c4f8b4445b0d0cb83b80-Paper-Conference.pdf) | 校准显式模拟多个 denoising steps，以多步输出差约束累积误差；内存复杂度由步数线性降为常数。 | trajectory-aware objective 已有；主要不是原生 NVFP4 视频部署。A：摘要、方法和实验覆盖。 |
| P07 | [TCEC / Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models](https://arxiv.org/html/2508.12094v1)，2025-08 | 建模量化扰动随 solver 传播，做 timestep-aware 累积误差补偿；不是仅层内校准。 | 误差传播/在线 correction 直接邻居；理论近似与当前 flow-matching/少步适用性应单独核实。A：§3及附录传播公式。 |
| P08 | [OrbitQuant](https://arxiv.org/html/2607.02461v1)，2026-07-02 | normalized rotated basis、RPBH、共享 Lloyd–Max codebook，不依赖 calibration 数据；权重离线吸收旋转。 | “免校准/旋转/跨步统一分布”已占据；任意4-bit codebook不能直接当作 NVFP4 E2M1。A：摘要、表征公式。 |
| P09 | [QVGen](https://arxiv.org/html/2505.11497v1)，2025-05-16 | QAT 辅助补偿模块改善训练，再用 rank-decay 移除推理辅助分支；1.3B–14B 视频模型。 | “先加低秩再去分支”已有；QAT 是强方法族，不应先验排除。A：摘要、方法。 |
| P10 | [SageAttention3](https://arxiv.org/html/2505.11594v1)，2025-05-16，NeurIPS 2025 | QK/PV 使用 NVFP4；QK smoothing、P 两级缩放避免 E4M3 scale 范围利用不足；RTX5090 实测。 | FP4 attention 和 P 的 scale 特殊性均已知；attention kernel速度≠模型总速度。A：§3。 |
| P11 | [QuantSparse](https://arxiv.org/html/2509.23681v4)，2025-09 初稿，读取 v4 2026-03 | 量化与稀疏共同放大 attention shift；多尺度 salient attention distillation、时序稳定二阶残差与 SVD projection；主设置 W4A8、15% attention density。 | 不能只声称“量化和稀疏有交互”；更细机制须独立证据。A：§1–3和实验设置。 |
| P12 | [SLA2](https://arxiv.org/html/2602.12675v1)，2026-02-13 | learnable router 在 sparse/linear attention 间分配计算，learnable 混合比例，attention QAT。 | “稀疏+线性补偿+低比特”已有完整方法；97% sparsity/18.6×为 attention指标。A：摘要与方法入口。 |
| P13 | [SharQ](https://arxiv.org/html/2606.26587v1)，2026-06-25 | input-adaptive N:M sparse FP4 主干 + 相对量化主干定义的 dense FP4 residual；共享权重与融合 preparation；含 Wan2.2 实验。 | “动态稀疏 outlier + FP4 residual”已有，并非只做 LLM。A：§3。 |
| P14 | [VC-Attention](https://arxiv.org/html/2609.15810v1)，2026-09-14 | V token 在线聚类、block mean residual量化；FP8 ExpCast去掉显式 exp/cast。H3 用第三方8步LoRA。workstation用4-bit V-Smooth，datacenter主推8-bit。 | V聚类/去均值与软最大算子都是强碰撞；明确区分GPU类别。A：§3.4、4及表2。 |
| P15 | [GCBT / Joint Branch-Space Transform Coding for Diffusion Activation Quantization with CFG](https://arxiv.org/abs/2610.00930)，2026-10-01 | 条件/无条件 activation 作2D相关源，离线2×2正交变换同时考虑 guidance方向和二阶统计。 | “利用 CFG 两分支相关性作 joint coding”刚被占据。B：官方摘要，尚未核实全部模型与原生kernel。 |

## 补充基线与检索入口（未计入15篇核心精读）

- [DVD-Quant](https://arxiv.org/abs/2505.18663)，2025-05-24；[ICLR2026正式PDF](https://openreview.net/pdf?id=3AnRMvlVDw)：data-free、rotation、delta-guided bit switching。已核实摘要，须补原生格式细节。
- [SpargeAttn](https://arxiv.org/abs/2502.18137)，2025-02-25：在线两级稀疏过滤与量化。已核实摘要。
- [VSA](https://arxiv.org/abs/2505.13389)，2025-05-19：可训练 coarse-to-fine tile 稀疏attention；含小模型 scaling 实验。已核实摘要。
- [Sparse VideoGen2作者项目](https://svg-project.github.io/v2/) / [官方代码](https://github.com/svg-project/Sparse-VideoGen)：语义置换使稀疏块适配硬件；NeurIPS2025。已核实作者项目，未读全文。
- [VideoMLA](https://arxiv.org/abs/2605.30351)，2026-05-28：共享 latent KV 缓存；明确报告预训练 KV 并非天然低秩，训练适应瓶颈。已核实摘要，不能和 pre-softmax QK低秩混为一谈。
- [FastWan-QAD作者发布](https://haoailab.com/blogs/fastwan-qad/)，2026-06-15：Wan2.1-1.3B NVFP4、3步蒸馏及低位attention完整系统；不是以博客代替已审稿论文，作为工程强基线。
- [PyTorch/TorchAO Blackwell diffusion官方实测](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)，2026-04-08：基础native实现参考。
- [GAMP / Closing the Null Space](https://arxiv.org/abs/2607.08241)，2026-07-09：仅保guidance gap存在branch drift盲区；已核实摘要。

## 覆盖边界

已覆盖：SVDQuant与native low-bit工程、视频PTQ/QAT、跨步累积误差、low-bit/sparse/linear attention、最新H3碰撞。尚未系统覆盖：多GPU通信量化、serving batching语义、跨请求global scale、2026年每一篇NVFP4 scale优化、OpenReview评审意见。后续只围绕拟保留claim补检索，不能把本轮未检出写成“无人研究”。
