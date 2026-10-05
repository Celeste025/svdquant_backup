# Power-of-two global：有限域性质，不是完整分区不变性

2026-10-04；只复核指定的 [scale-support note](../02_problems/h3_scale_support_screen_20261004.md)、[batch-invariance note](../01_literature/batch_invariance_collision.md)，定向检索并通读下列两份 primary。无代码、GPU、模型调用；未扩展全库检索。

**判断：未在这两份 primary 中核实到“将 NVFP4 tensor-global 限为 2 的幂以消除正常尺度域的分区依赖”这一具体配方；不能写成已被完全直接覆盖，也不能把有限检索未命中当新颖性证明。其核心性质是直接的浮点指数齐次性，且两份本地 note 已明确记录。当前不足以作为新方法立项，更不能宣称完整 partition-independent NVFP4。**

设块最大值为 a，理想块尺度 t=a/6，有效尺度为 d(g)=g·R_E4M3(t/g)。若 g′=2^k g，且两个归一化尺度及舍入邻域都处于可缩放对应的正常数范围，不遇下溢、饱和或特殊 clamp，则 R(t/(2^k g))=2^−k R(t/g)，故 d(g′)=d(g)。相同输入、相同微块边界、相同 E2M1 舍入规则下，FP4 codes 和解码值随之不变。**这是本文推导，不是所引文献的新实验结论。** 理想算术中任意固定 g₀ 的集合 {g₀·2^k} 也固定同一格点相位；取 g₀=1 还简化了二进制缩放。限制的是 global，不是把 E4M3 微块尺度改成 E8M0，不能混同于 MXFP4。

**完整主张的精确反例。** 共同微块取 t=2^−6。某分区的最大值为 2688，用 g=2^ceil(log₂(M/2688)) 得 g=1，块尺度为正常数 2^−6；另一分区加入最大值 86016 的无关块，得到 g=32，此时 t/g=2^−11，小于 E4M3 最小正数 2^−9 的半值，RNE 舍入为零。同一共同块从非零变零，两种 g 均是 2 的幂，最大块也没有溢出。这个合成反例否定无条件保证，**不证明 H3 会发生相同幅度变化或质量损伤**。向上取整 global 避免最大块溢出，却不能排除弱块落入次正规/零区；clamp 最小正尺度也不能恢复齐次性，因为它的绝对下限随 g 改变。

| 两份 primary 的直接覆盖 | 对本候选的准确边界 |
|---|---|
| [humans&，The 4-bitter Lesson，2026-07-10，Baseline recipe](https://humansand.ai/blog/nvfp4-rl?v=3) 明确指出 tensor-global 导致跨 token/batch 依赖及 future→past 路径，用 per-token FP32 global 配合 group16 E4M3 并融合实现。 | 已覆盖问题和一个直接解法；未提出这里的 pow2-global 方案。不能因此认定它解决了同一部署成本问题，也不能凭空假设 per-token 元数据昂贵。 |
| [Finer is Better (with the Right Scaling)，v2，§I–III](https://arxiv.org/html/2605.08565v2) 明确研究 E4M3 有限指数范围、尺度归零与次正规粗网格，并比较防零、层级尺度和 4-over-6。 | 已覆盖使无条件不变性失败的数值边界；未证明 pow2-global 的分区不变性，也不能把其 MXFP4 的块级 pow2 规则当成本候选的直接先例。 |

还须区分三个合同：①上述正常域内**解码值**不变；②packet 并不逐 byte 相同，g 与 FP8 scale 的指数本来就在改变；③完整 GEMM/模型输出逐 byte 分区不变还依赖分块、累加、global 应用位置、cast 与低秩支路的算术路径。只限制 g 不能自动证明③；改变微块成员本身也不在①的保证内。

**决策：保留为常规数值合同选项，不启动 GPU 或 scale 网格。** 若以后确有真实分区依赖导致的任务失败，并且正常域安全范围可被明确约束，pow2 global 是有动机的低成本工程对照；方法贡献仍需一个已有 per-token/fixed-global 等方案不能满足的真实质量—部署约束。目前没有该证据链。这里的止点是“正常域指数吸收本身与无条件不变性表述”，不是对所有后续有证据的尺度设计作不可能性判决。
