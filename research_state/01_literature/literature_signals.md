# 文献信号：尚非候选论文结论

更新：2026-10-02。所有“可能/未知”均为本轮推断，不是文献已证明事实。来源编号见 [paper_index.md](paper_index.md)。

## S1：多步量化的误差方向与跨步相关，可能比单步大小更关键

- 来源：[PulseQuant](https://arxiv.org/html/2609.33384v1)、[AccuQuant](https://arxiv.org/html/2510.20348v1)、[TCEC](https://arxiv.org/html/2508.12094v1)。这些工作已否决“先量local MSE就能知道最终好坏”的充分性。
- 尚不清楚：native NVFP4在重复weight与动态activation上是否产生可重复的同方向误差；final error中跨步cross terms到底占多少。若只复现local/global不一致，将直接撞P05。
- 最小有效检验：同一初始latent完整轨迹，分离weight-only、activation-only和两者；采集每步quant-minus-dense velocity及solver加权方向。先在相同dense输入上看相关，再用closed-loop看传播，避免两种误差混用。
- 反证：相关在prompt/seed上符号不稳、被简单bias correction解释，或互补噪声降低相关却不改善终点/视频质量。
- 候选后续动作：先查antithetic/complementary rounding及solver error-feedback；无查新结果前不实现复杂方法。

## S2：模拟表征的优势未必能保留到真实NVFP4

- 来源：[OrbitQuant](https://arxiv.org/html/2607.02461v1) 的球面codebook、[PulseQuant §5.1](https://arxiv.org/html/2609.33384v1)、[SageAttention3 §3](https://arxiv.org/html/2505.11594v1) 的scale敏感性。
- 未决问题：方法改善的误差方向或codebook选择，在E2M1、16-element E4M3 scale及有限global scale约束下是否仍然有效。表征切换可能比校准改进更大。
- 最小有效检验：固定层覆盖/输入/权重，比较native pack/dequant与宣称量化契约的reference模拟输出；如果不一致，定位为format mismatch，先修实验基础。不能拿实现bug充当论文insight。
- 反证：bit-exact native一致，且各方法名次/收益不受表征影响。若剩余只是工程纠错，按基线工作收尾。
- 可能的研究价值：只有发现跨模型可解释的结构性失配、给出契约内方法并兑现端到端收益时才继续。

## S3：短轨迹可能改变“保护哪种误差”的性价比

- 来源：[6Bit-Diffusion](https://arxiv.org/html/2603.18742v1) 的50步与缓存、[VC-Attention](https://arxiv.org/html/2609.15810v1) 的前25% grouping与复用、[FastWan-QAD](https://haoailab.com/blogs/fastwan-qad/) 的3步模型。
- 未决问题：少步不是简单减少次数；solver跨度、误差方向与preprocess摊销同时改变。静态“早步更重要”未必能跨checkpoint/solver成立。
- 最小有效检验：同模型家族常规/蒸馏两种checkpoint，先测precision×step sensitivity与真实preprocess占比，而非扫几十组hybrid bit schedule。
- 反证：每种设置都只需独立调阈值就解决，没有可泛化机制，或者仅得到一个稍快实现。
- 注意：不能把rCM4步结果直接当作原始Wan50步证据；QAD/缓存/混合精度已有强先例。

## S4：batch内global scale可能让独立请求数值耦合

- 来源：[SageAttention3](https://arxiv.org/html/2505.11594v1)证明二级scale本身影响误差；[PyTorch native diffusion实现](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)可作为实现核查入口。跨请求问题本身尚无本轮证据。
- 明确假设：只有runtime把多个请求共同归入global amax/scale统计域，才存在该机制。per-request或固定scale实现不受此假设影响。
- 最小有效检验：同请求同输入独立执行/与不同幅度、不同timestep请求拼batch，固定所有随机量，比较native输出及scale；用per-request scale/固定scale作因果对照；测真实pack/GEMM成本。
- 反证：实际kernel已隔离scale、变化仅来自GEMM非确定性，或层误差不传导到视频质量。
- 价值门槛：发现实用serving设置中的可复现语义耦合，并提出batch不变且保留低位吞吐的方法；只报batch不完全bit-exact不够。

## 暂不推进的表面idea

- 时空均值+FP4 delta：DeltaQuant。
- block变化决定动态bit，再叠cache：6Bit-Diffusion / DVD-Quant。
- pulse测最终损伤再权重校准：PulseQuant。
- 稀疏+量化产生attention shift再蒸馏修正：QuantSparse；sparse+linear+QAT：SLA2。
- 动态N:M稀疏outlier再FP4 residual：SharQ。
- V按相似性分组、减均值补偿：VC-Attention。
- CFG两分支共同旋转编码：GCBT。

后续判据：先用小而保真、能推翻候选的实验收集证据；任一signal被否决就明确记录，不扩scope挽救。


## S5 定向补查：总体重建改善、视频 token 误差却恶化

更新：2026-10-02。本节仅追踪三条最近先例，不扩展通用地图。输入证据来自主 agent 报告的 E001：H3 block 0、真实 packed、全长度、2 prompts、早晚步；text 约占 3% tokens，却占参考输出能量 88%–99.7%；旧 state 的 SVDQuant 相对 plain W4A4 在首步使全 block 误差改善约 14%，video-token 误差反而增加 46%–88%；hook 修复后现象保留。**本子 agent 未独立审计这些数值；这里仍是局部 signal，不能外推端到端质量。**

| 最近先例与原始来源 | 本轮核实内容 | 对 S5 的约束 |
|---|---|---|
| **MBQ: Modality-Balanced Quantization for Large Vision-Language Models**；2024-12 初稿，CVPR 2025。[原文 §3.1–3.2](https://arxiv.org/html/2412.19509v2)。 | 已读公式 6–17：视觉/语言 token 的敏感度差异导致普通重建目标配置不当；先展示按模态加权的 MSE，再用最终 SFT loss 梯度导出模态加权 MAE，搜索 equalization factors。 | **普通 modality-aware loss、把不同模态平均后加权、按梯度分配模态权重，都不是新点。** 任务是 VLM 理解，不是联合音视频生成；但迁移重加权本身仍是弱贡献。 |
| **MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization**；2024-05-28，ECCV 2024。[原文 §3.1–3.2](https://arxiv.org/html/2405.17873v2)。 | 已读：SDXL CLIP 的 BOS token 幅值显著高于其余 token，影响 cross-attention K/V 量化；其 prompt 不变特性允许预计算并保护该 token。另指出 SQNR 驱动的 bit allocation 可使生成质量恶化，采用内容/视觉质量分离的指标。 | 已覆盖“条件 token 极值支配量化”和“数值 proxy 改善但生成质量更差”的上位叙事。它不是 H3 共享 joint block 的模态能量主导校准；S5 若只改指标/保护 text 仍容易被吸收。 |
| **RGSQ: Riemannian Geometry-Sensitive Quantization for Large Vision-Language Models**；2026-09-21。[原文 §IV-A–IV-B](https://arxiv.org/html/2609.25492v1)。 | 已读：按模态收集 activation/gradient 二阶因子，以敏感度融合 Kronecker–Fisher metric，进行 geometry-aligned rotation 与 whitening，在非各向同性的误差度量下校准。 | 仅把模态权重升级为 Fisher/几何度量也已有近邻。**没有核实它处理 LayerNorm/RMSNorm gauge equivalence**；不能把一般信息几何直接等同于 normalization-gauge invariant calibration。其评测为 VLM，不是联合 A/V/T denoising。 |

本次有限检索**未核实直接针对联合 audio/video/text diffusion PTQ、且以 normalization gauge 为核心的对应方法**。这是核实缺口，不是空白证明。下一步重点应从“模态重加权”转向可区分的机制证据：

- 排除 proxy 的分母效应：分别报告各模态 absolute SSE、每 token MSE、relative error 和参考能量，不能只看拼接后的一个比值；text 参考能量占比高并不单独证明 text 主导了实际训练/搜索 loss。
- 因果检验是选择性替换/传播：仅替换 text、video 或 audio 的误差，让后续 dense blocks/solver 继续，观察终点视频与音频损伤。少量高能条件 token 也可能对视频极重要，不能仅因它们不是最终生成 token 就降低权重。
- 若提出 normalization gauge，先给出**精确不变变换及适用计算路径**。LayerNorm/RMSNorm 局部消去的方向，未必对带 residual、AdaLN、attention 的整个 block 不可见；简单除以 token norm 不等于 gauge-invariant。可先测试 normalization 前后误差名次是否反转，再验证传播敏感度。
- 停止条件：若普通 MBQ 式 reweighting 或各模态相对 MSE 就稳定消除现象，且没有更深的预测/方法收益，按有用工程修复记录，不包装为新论文主张。
