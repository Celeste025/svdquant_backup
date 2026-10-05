# 独立审视：暂不足以提出强候选

2026-10-02；只读013–015、current_state、field_map及系统/低秩/批独立性碰撞记录，核查一手论文和源码；无GPU、无代码修改。采用 claim-prior-work-triangulation 的精确claim原则。本页不登记active claim，不扩E017基线网格。

**结论：当前尚无一个既有具体未覆盖机制、又有本地决定性证据的强候选。** 这不是“只要相关论文存在就kill”：下面列出的硬件、算术合同和工作负载差异确实未被抹平；但差异本身不足以支持新论文。E016已经证明整模低位attention有收益，不能再用E006局部阴性否认其部署意义。

## 本轮最有用的新增碰撞

- [MpFA，2026-09-27，§4–7](https://arxiv.org/html/2609.33135v1)：B200/tcgen05/TMEM，QK4PV8，global Q/K centering，先用BF16 GEMV生成并物化每key的ΔS，再用额外BF16 MMA广播到score accumulator。它移动的是**correction的消费**，不是在attention内即时生成Qmean·Kᵀ。head_dim=128有实验；仅报告LLM causal prefill/decode，未证明H3 noncausal/packed varlen或SM120可用。P/V改FP8、ΔS引入BF16舍入，因此既不等价E016 FP32 correction，也未解决block-mean物化矩阵。宽泛的“混合PV精度、用MMA消费correction、FP4后softmax瓶颈”已经覆盖；SM120同合同即时生成仍未被该文直接覆盖，但目前仅是实现残余。
- [SageAttention官方core.py](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py)：return_lse接口明确用于Ring，已有`lse / 1.44269504 + lse_correction * sm_scale`。所以“各KV分片独立减K均值，再将被删的行常数补进LSE做合并”已有直接实现。由`Q(K−μ)ᵀ + Qμᵀ`推导这个修正不是新分布式方法。是否能一次pack后在当前FP4路径直接通信并消费，是另一个尚未本地验证的问题。
- [LongLive-2.0，Appendix D](https://arxiv.org/html/2605.18739v2)：已有NVFP4 Q/K/V All-to-All，故“FP4计算＋FP4通信＋视频”不能重开。其AR cache与本项目双向、逐步重算的区别没有被本文直接消除，但也不能仅凭这个setting差异立项。

补充排除：如果后续观察到FP4 P使常值V不再保持常值，“按PV实际消费的量化P累计分母”也已有[MXAttention的PNQ](https://arxiv.org/html/2607.24377v1)和[Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/html/2609.04105v1)，不建议另命名为概率质量守恒方法。

## 真正尚缺的证据与最小决定实验

本地可保留的窄问题仍是旧N1的精确版本：**双向H3在原生FP4 attention中，每层每步重算统计/修正后，量化表示能否沿真实分片边界直接被消费；若不能，是否产生常规尺度/布局处理仍无法隐藏的关键路径？** 目前没有这条trace，不能声称存在系统瓶颈，更不能先建分布式后端制造需求。

最小下一动作只需CPU源码/现有trace核查：对现有有效长段画出P=2、128对齐分片下的Q/K/V codes、block scales、global scales、K均值、Q均值、ΔS的生产者和消费者；逐项扣除已知的局部pack、原样metadata交换、LSE修正与布局复用。若所有依赖都可沿既有ownership本地满足，直接停止，无需GPU。只有留下具体不可本地化的依赖，且实际采用两卡是明确的时延/容量选择，才对**那一个边界**做一次包含pack/通信/correction的两卡计时；不跑新prompt、不建立整套并行baseline。瓶颈若只是普通all-reduce或一次常规重排，也不升级claim。

对当前E017，完成原计划即可。两个已见prompt的质量差异最多决定已有配方的诊断使用边界，不能证明上述并行问题，也不能反向选择一个故事。即使E017为阳性，下一步仍需一个能排除现成解释的具体干预；即使阴性，也不否定低位attention及自然部署目标的研究空间。

后续定向核查另存于[FP4 attention分区可组合性](fp4_attention_partition_composability.md)：该问题区别于K均值/LSE恢复和概率row-sum守恒，源码与代数支持具体入口，但H3实际量级未测；整数log2参考max及逐tile两级缩放已有直接方法近邻。
