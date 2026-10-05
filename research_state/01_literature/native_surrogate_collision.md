# Native / QDQ surrogate 的有限碰撞核查

2026-10-02；本轮只读论文和官方实现，没有运行实验。沿用 claim-prior-work-triangulation 的碰撞原则。前轮的范围与 E007 数字见 [native_systems_frontier.md 的 E007 补充](native_systems_frontier.md#e007-补充同码值-fake-quant--native-算术分歧的最近先验)。

**判定：泛泛的“量化模拟与真实执行不一致、上游差异改变后续 activation quantization、需要训练/部署数值对齐”不能立新 claim。** 本轮指定论文没有直接隔离 E007 的 same-packed-operands / global 放置位置，但这只留下执行合同的归因工作，不足以成立新方法。fused BF16 LR 的舍入差异也已有官方实现结构可直接解释，不能把普通 epilogue 舍入包装为新机制。

## 最接近的证据

| Primary source 与核实范围 | 已覆盖 | 没有直接核实到的部分 |
|---|---|---|
| [QUADS，arXiv v1 2026-07-17](https://arxiv.org/html/2607.15810v1)，§2.1、§3.1–3.2、§4.5 | 显式区分 weight quantization、activation quantization 与 numerical-path 差异；指出两引擎的 BF16 activations 在量化前已经不同，activation QDQ 不能简单消除这种差异。其 BF16 对照涉及 kernel、累加顺序和舍入。 | 未展示将 A/W codes 与全部 scales 同时固定后，只替换 BF16-QDQ/native GEMM 的隔离；未分析 FP32 global 乘法前移/后移；未研究 SVDQuant BF16 LR epilogue。§3 的实数代数不完整指定这些有限精度位置。 |
| [NVIDIA QAD 报告，arXiv v1 2026-01-27](https://arxiv.org/html/2601.20088v1)，§2、§3.1、Appendix D；[NVIDIA 官方 PDF](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)为同题报告 | NVFP4 双层 scale、QAD 对齐 teacher 分布、QAT/QAD 与 native quantized training 的目标和 forward/backward 范围区别。 | Appendix D 区分哪些 GEMM 被量化，并不是 samecodes BF16-QDQ/native 的逐算术合同对比；所读正文未给 global 位置或 fused LR 舍入消融。不能把“QAD”直接视为已经解决任意 native-surrogate mismatch。 |
| [Transformer Engine 官方 PR #2644](https://github.com/NVIDIA/TransformerEngine/pull/2644)，作者描述与已合并状态；对应 [humans& 原始说明，2026-07-10](https://humansand.ai/blog/nvfp4-rl) | backward 可以采用 forward 实际量化值的反量化副本；4/6 训练与推理实现要求 codes/scales 及 error-reduction 决策的数值合同一致。 | backward 输入与 forward quantization decisions 对齐，不等于证明 BF16 反量化 GEMM 和 native forward 逐位相等。4/6 的 bit-exact 合同主要是**编码决策**，不是固定 codes 后所有 GEMM/epilogue 的等价定理。 |
| [FlashInfer 官方 `gemm_svdquant.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_svdquant.py)，`mm_nvfp4_svdquant` 文档、fused/unfused runner；访问时 main 分支 | API 明确 `alpha * (a @ b.T + d @ l1.T) + bias`；SM120 fused 路径在 FP32 accumulator epilogue 加 LR/bias。unfused 路径先输出 BF16 FP4 GEMM，再 BF16 `torch.mm`、乘 alpha、加 correction/bias。调用方还需把 `1/alpha` 折入 BF16 `l1`。 | 文档已直接暴露不同舍入位置；没有在这里看到视频整模质量影响的独立研究。`BF16(B/alpha)` 本身也是额外舍入点，不能只因实数代数相等就声称这与旧 `BF16(BF16(xA)B)` 分支完全相同。此网页不替代本机版本的逐源审计。 |

QUADS 的贡献主体是 MoE RL 的两侧误差对齐与 activation compensation。它对宽泛问题构成重碰撞；“把 RL 改为视频 DiT”本身只是 setting 差别。这里没有宣布视频场景剩余机制已成立。

## 关于 E2M1 × E4M3 的纠正

所给 [Jianyu Huang 7/11 页面](https://jianyuh.github.io/nvfp4/2026/07/11/NVFP4-RL.html) 开头明确是在阅读/转述 humans& 原文，因此不作为该实现的 primary evidence。它说 BF16 缺乏精度来表示 E2M1×E4M3 乘积；**这个不带 global 的字面断言不成立**。原始 [humans& 4/6 段落](https://humansand.ai/blog/nvfp4-rl#four-over-six-for-rl-weights-and-activations) 给 FP16 fast path 的注脚理由是相关 packed 转换指令的支持时间，而不是 BF16 mantissa 不足。不能把转述里的数值错误归给官方实现。

数学上，正常有限 E2M1 与 E4M3 的有效 significand 分别至多2和4 bits，乘积至多6 bits，处于 BF16 足够的指数范围内，因此可精确表示。root 的 [CPU 枚举记录](../06_experiments/results/fp4_e4m3_unscaled_bf16_check.json) 覆盖128个非负有限 E4M3 bitpatterns（含两种零）×8个非负 E2M1 值，1024个乘积经 FP32→BF16→FP32 没有数值差异。本轮未复跑该枚举。

**任意 FP32 global 再乘入并 cast BF16，才可能额外改变输入数值。** 即便把 global 后移，仍必须区分 tensor-core reduction、FP32 epilogue、BF16 输出以及 LR 中间舍入。逐元素 `q*s` 精确不推出整个 GEMM 精确，也不推出 late-global BF16 GEMM 与 native bit-exact；它只排除一个错误归因。

## E007 目前能说什么

- 单层约1e-5与30-block endpoint 4.28%的差异说明局部容差不足以代替整模部署验证。它不是 denoising 多步误差累积实验，也不是质量改善/下降证据。
- 在同一输入下固定 codes/scales 后，对 global 位置及 LR 舍入点做有限归因，有执行审计价值。将模拟改为更贴合 native 合同属于参考实现修正，不为其另取新方法名。
- **新颖性状态：宽问题 heavily overlapped；指定 samecodes/global/LR 隔离在这次有限核查中尚未被指定论文直接覆盖，但剩余部分只是尚待核实的实现/经验差异。** 没有可由当前证据成立的新论文 claim；不扩大搜索、不据此追加训练或 GPU 实验。
