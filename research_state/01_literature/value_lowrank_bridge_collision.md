# V 低秩分量跨过 attention：有限碰撞

2026-10-02。仅 primary 文献核查和代数分析，没有注册 claim、实现或 GPU 实验。候选是保留 SVDQuant V projection 的 `V = V_res + U Bv`，用 `P V_res + (P U) Bv` 避免先合并、再量化 V；不尝试穿过 Q/K normalization 或 RoPE。

**决定：PARK。** 低位 V 加独立高精度低秩读出已有直接近邻；在视频 FP4 attention 中绕过 value quantization 保留 rank-one 分量也已实现。复用“上游 SVDQuant 已有 U”在本轮读到的来源中未被逐字指定，但目前只是因子来源/实现层差别，不足以成立研究贡献。E006 的净抵消结果没有证明 merge-requant 正在造成值得修复的损伤。

## 碰撞证据

| Primary source | 已覆盖的结构 | 与候选仍有的区别 |
|---|---|---|
| [GEAR，arXiv v3 2024-08-29](https://arxiv.org/html/2403.05527v3)，§3、§4 implementation；[2024 PMLR 版本](https://proceedings.mlr.press/v262/kang24a.html) | KV 分为量化主体、低秩残差与可选稀疏项；实现明确让 low-rank matrices 走独立先 down、再 up 的 forward 路径以降低计算量。不是要求先重建完整高精度矩阵再做 attention。其 GEAR-L 已省去 sparse 项。 | 因子由 KV quantization residual 得到；正文举的因式计算例子在 K 侧，不能伪称它已实现本项目 SVDQuant V→NVFP4 online-P 的完整合同。即便如此，“量化主体外保留低秩独立乘法”已经不是新结构。 |
| [VC-Attention，2026-09-14](https://arxiv.org/html/2609.15810v1)，§3.2 Eq.6 | 视频低位 attention 把 V block 分成 residual 与 rank-one mean。只有 residual 量化；均值在 online softmax 的 row-sum 路径进入输出 accumulator，避免重新混入 V 量化，且不 materialize P。 | 其 `U=1` 很特殊，`P U` 是已有的 softmax row sum；任意 rank32 U 无法免费复用这个统计量。“把 rank1 换成 rank32”不是自动贡献，反而增加明确成本。 |
| [ResQ，2024-12-18 v1](https://arxiv.org/html/2412.14363v1)，§4.1、§4.3/Fig.3 | 在低秩高方差子空间保留高精度 coefficients；应用于 KV。V 的 basis 融入 v_proj，对应逆变换融入 o_proj，在 attention 中保留分空间表示，避免无意义的高精度重建边界。 | 它用正交补空间与 mixed-precision coefficients，不是 SVDQuant 非正交加性 residual。这个区别不能抹掉“保留 V 高精度子空间跨 attention”的重叠。 |
| [LongLive-2.0 v2](https://arxiv.org/html/2605.18739v2)，§3.1–3.2 | 已有 NVFP4 video backbone + 独立/融合 LoRA，以及 chunkwise KV quantization。 | 所读段落使用窗口内 KV 重建，未找到把 projection LoRA 的 V 因子直接推进 `P U` 的描述；不把其“有 LoRA + KV4”误称为本候选已完全实现。 |

SageAttention3 的已核实内容是 native NVFP4 QK/PV、P scaling 与 layout。它不是本轮最致命的近邻，因此没有重读整篇；VC-Attention 与 GEAR/ResQ 的直接结构碰撞已经足以停止。

## 即使忽略新颖性，成本也不是天然改善

以下为我们的代数推导，不是论文测量。对单 head，`Nq` 个 queries、`Nk` 个 keys、value width `d`、rank `r`：

- 普通路径先做 `U Bv`，成本约 `2 Nk r d`，再做完整低位 `P V`，成本约 `2 Nq Nk d`。
- 分支路径仍保留完整 residual 的低位 PV；新添 `P U`，成本约 `2 Nq Nk r`，再做 `(P U) Bv`，成本约 `2 Nq r d`。
- 对当前双向视频 attention 的 `Nq=Nk`，up projection 只是从前面移到后面，同阶 FLOPs 没有消失；新增的是按 head 计算的 dense `P U`。相同 U 可以共享存储，但不同 heads 的 P 不能共享计算结果。
- `r=32,d=128` 只使新增乘法元素数约为 PV 的1/4。若要保留 BF16 支路，不能把它按 FP4 吞吐定价；在理想 FP4/BF16 4倍吞吐假设下，新增 `P U` 的算术时间量级就可能与原 FP4 PV 相当。这是粗略成本否证，不是 latency 预测；实际还受 tile/occupancy/softmax 重用约束。
- 对自回归单 query decode，`Nq≪Nk` 的重排确有不同成本区间；这恰是已有低秩 KV-cache/latent-attention 工作需要优先比较的场景，不能搬来为当前双向 H3 提供免费收益。

还有两个执行合同不能略去：`P_q` 的低位路径与高精度 `P U` 是否消费同一个概率近似；旧路径在 `V_res + BF16(U Bv)` 上的 BF16 舍入，不能靠实数结合律宣称与新路径 bit-exact。它们会改变比较对象，不能统一称为“只消除了二次量化”。

本轮不安排廉价 GPU 试验，因为已有直接 prior 和基本成本分析都不支持立刻投入。若未来真实端到端 profile/质量证据发现现有方法无法解释的新现象，可重开一个明确问题；当前不以“更一般的 U”填补已经停止的第三种 idea。
