# H3 与 SVDQuant：结构假设还是基线口径缺口？

2026-10-04。**0 个新增方法候选。** 读 current_state、旧机制/表示/部署筛查，检查实际 H3 与 PTQ 源，并核对下列三个一手来源；无 GPU、无模型调用、无实验代码改动。E068 人评尚待，不据 SSE 反转继续机制扫描。

**唯一核查问题：H3 的融合 QKV、QK RMSNorm/RoPE、SwiGLU 和按模态/时刻索引的 AdaLN，是否使 SVDQuant 的表示或校准假设失效，留下一个仅移植官方实现不能解决的问题？目前未发现。** [实际 block](../../../DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:223) 的两条支路均先 norm/调制、再 attention 或 MLP、最后门控加回；[Comfy attention](../../../DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:17) 在融合 QKV 后再做 QK norm/RoPE。这些操作影响误差重要性，但没有破坏在**实际 linear 输入处**进行可逆平滑与低秩分解的代数。不能误把“平滑需要穿过门控/非线性才能成立”作为前提；不跨这些节点搬移即可。

**一个重要反例是当前 H3 PTQ 源本身。** [_error → _stage_metrics](../../scripts/ptq_minimax_h3_svdquant_standard.py:88) 用完整 block 输出给候选评分；[块间缓存更新](../../scripts/ptq_minimax_h3_svdquant_standard.py:300) 使用选中量化 block 的输出。因此它并非从定义上忽略上述门控/非线性，也并非始终只看 BF16 上游输入。每个 linear 的候选单独评分不等于四个 linear 联合优化，更不保证生成质量。**这是当前可读源码的能力；旧产物缺生成时该脚本 SHA，不能反推历史执行已完整采用此流程。** 本轮所读脚本 SHA：`e78fce9486da4cc118e63089094cabace49b7426936ed42ca5b568857c98203b`。

**已知方法为何挡住“结构感知”的泛化 claim。** [SVDQuant 论文 §4.2、附录 B](https://arxiv.org/html/2411.05007v4) 已含迭代更新低秩/残差，并按分解后的输出误差选平滑强度；[官方 weight.py](https://raw.githubusercontent.com/mit-han-lab/deepcompressor/main/deepcompressor/app/diffusion/quant/weight.py) 对 Q/K 可用整个 attention、部分并行结构用整个 block 评估；[官方 smooth.py](https://raw.githubusercontent.com/mit-han-lab/deepcompressor/main/deepcompressor/app/diffusion/quant/smooth.py) 也已有 attention/block 目标及无法折叠时的输入平滑 hook。这些不直接提供本地 Comfy-H3 的适配器和验证结果，但足以覆盖“把 norm/RoPE/后续模块纳入候选评分”的一般解法。所读 upstream main 与本地带 Wan 扩展的版本须区分，不把本地 wrapper 冒称官方功能。

**真正未补齐的是比较身份，而非新的结构定律。** 当前 H3 源先用无 LR 候选选 smooth；E065 则固定旧 smooth，在八份 teacher 输入上按逐 linear SSE 选 carry/restart。它们都不能自动等同于“完整官方 SVDQuant 在 H3 上已做强对照”。E065–E067排除的是各自冻结合同内的解释，不是证明完整校准或模型适配失效；E068也只比较两份冻结导出。

目前无法给出一种 H3 特有约束，使正确 adapter、原有评估模块选择和原有平滑/分解流程必然不足。补齐这些只是必要基线；若问题只能通过同预算的现成配置或正确接线消失，就停止算法 claim。只有独立任务评价确认仍有损失，且能指出**现有方法的具体不可满足约束**，才重新立问题；不是要求实验前已有决定性阳性，也不是据未复现官方基线永久阻止创新。本轮不提出新 loss、门控补偿或追加校准实验。
