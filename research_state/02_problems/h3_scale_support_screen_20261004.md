# H3 两级 NVFP4 尺度的有效支持范围：筛选结论

2026-10-04。独立静态筛选；0 GPU、0新模型调用、无实验代码；3个定向网页查询，随后读取一手原文。E059数值控制失败且没有完整oracle结果，只表示该来源控制不可用，不作为任何尺度机制阴性。

**结论：REJECT 作为当前新研究方向；不推荐新增实验。** 共享 global 导致跨 token 依赖、E4M3 下溢造成全块归零、次正规尺度与 E2M1 粗网格共同恶化误差，均已有直接先例。把它们组合到 H3 的 text/video/audio/pad 上，目前只是具体实现审计。唯一值得精确定义的问题如下，但本轮证据没有越过常规修复与已知方法的边界。

**唯一候选问题。** 在真实 H3 输入中，是否有**对有效输出无计算路径的 pad token**决定 tensor amax，从而使有效 audio/video 微块丢失表示支持或产生有害离散跳变？选择 pad 是为了获得 BF16 的严格零效应对照：真实 text/audio/video 之间本来存在 attention 路径，修改其值不能把输出变化都归给 quantizer。这里不提出普通模态分组尺度、模态重加权或时间补偿。

## 实际数值合同及已有证据

- [`h3_nvfp4_fastpack.py`](../../scripts/research/h3_nvfp4_fastpack.py) 对已经过 BF16 smoothing 的完整 `x_s` 求 amax；典型非极小输入下 `g=M/(6×448)`。每个 token 的每16通道微块独立计算 `s_b=RNE_E4M3(max(a_b/6,1e−12)/g)`，有效尺度为 `g*s_b`。不同模态不共享微块 scale，只共享 g。
- **两种 padding 不同：** `_amax_parts` 的 `N=x.numel()` 排除了 packer 自己补出的 swizzle 行/列；它们只写零 scales。H3原始 `x_s` 中的 pad rows 却属于 N。已核实p1布局的16个pad构成独立attention段，BF16中其输出没有通向最终audio/video的路径；不能把这一单例结构未经核实推广到所有packed输入。[结构审计](../00_state/phase2_systems_feasibility.md)
- 忽略最小值clamp和FP32边界误差，非零微块满足 `a_b/M ≤ 1/(448×1024)≈2.18e−6` 时，E4M3 RNE可将scale舍入为零（精确中点向偶数零）；理想scale低于normal下界对应 `a_b/M<1/(448×64)`。这是格式推导，不是H3新发现。normal范围内g的纯2次幂变化通常由指数吸收；不能假设任意amax变化都损坏码值或质量。
- [`h3_nvfp4_zero_sf_compat.py`](../../scripts/research/h3_nvfp4_zero_sf_compat.py) **允许并记录**非零输入SF0，将其编码为零；它恢复历史QDQ合同，没有恢复丢失数值。全局仍不变，不能称为underflow修复。
- 已有[E009真实block1.fc2记录](../../results/research/E009_h3_sf0_domain_v2.json)：95个SF0微块全部属于audio，丢失输入能量占audio **6.887×10⁻⁶**、全输入约 **1.493×10⁻¹²**；video/text/pad的SF0为0。另有351481个非零subnormal scales，但现有该表未给其模态归属和功能作用，也未给amax拥有者。**这支持“发生过数值下溢”，不支持“有效语义塌缩”或“padding/某模态导致它”。** 能量小也不是无功能影响的数学证明。

## 能改变决定的预测，以及为什么目前不值得立项

保持真实有效rows、形状、cu_seqlens和权重不变，只将原本独立的pad rows设零。如果真实pad不拥有严格更大的amax，g与有效rows的codes/scales必须不变；若pad拥有amax，移除后应只在预先由`a_b/g`预测的尺度边界出现支持变化。对同一固定输入，仅排除pad的amax统计必须复现该有效packet变化。BF16有效输出应保持不变，否则负控制结构有误。**人工把pad放大直到失败只能验证公式，不能建立真实H3问题。** 任意非2次幂global变化带来的普通格点phase变化也不足以称为support collapse。

即使这一预测在真实输入成立，最强简单基线依次是：①不让无效pad参与统计，并让无效行保持安全有限编码；②固定原g，只将非零块scale下限设为最小E4M3正数，检查是否只剩已知下溢；③对有效token采用已有per-token global，并比较4-over-6/可表示scale搜索。只修复其中任一项就恢复有效输出，应记工程修复。clamp不能保证全部原值非零，且可能改变E2M1误差；必须看重构/功能结果，不以SF0计数降为零当收益。

**只有以下新证据才会改变REJECT：** 真实pad的统计干扰被严格隔离、会重复破坏teacher已有的具体生成行为；去pad统计与已知per-token/scale-selection强基线仍存在一个明确的质量—部署预算矛盾，并且不能归结为本机接口缺功能或某个kernel未移植。当前三项皆无证据。E003/E004的模态取舍与有限refit结果没有测这个条件；其阴性不能证明该机制不存在，也不能为它提供新的正动机。无需先跑GPU去重复一般下溢定律。

## 精确碰撞与部署边界

| 一手来源 | 已覆盖内容；不能据此声称什么 |
|---|---|
| [humans&，The 4-bitter Lesson，Per-token Activation Scaling](https://humansand.ai/blog/nvfp4-rl?v=3) | 明确指出tensor-global使一个token的量化依赖其他token，甚至建立future→past路径；采用每token FP32 global＋group16 FP8 scales，并给出融合量化与推理实现。覆盖统计域泄漏及普通隔离解法；没有证明本地SM120 H3的完整性能。 |
| [Finer is Better (with the Right Scaling)，v2 §II–III，2026-06-08](https://arxiv.org/html/2605.08565v2) | 明确分离scale归零与非零subnormal粗网格问题；对比prevent-zero、4-over-6、层级scale、穷举，并做LLM权重/激活与功能评估。其主要现象是粒度悖论，并非H3模态实验；但“零以外还有subnormal损伤”已经不是残余机制。 |
| [NVIDIA Model-Optimizer，nvfp4_tensor.py](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py) | 当前官方源码的`_cast_per_block_scale_to_fp8`直接clamp到`[2⁻⁹,448]`再cast，防止下溢/溢出；所读调用主要涉及权重导出/量化，不能冒称已验证H3动态A。它足以说明最小正scale不是新修复。 |

已有[batch-invariance碰撞](../01_literature/batch_invariance_collision.md)还核实FlashInfer per-token NVFP4接口；“再按模态/请求拆分global”不构成新贡献。单行额外FP32 global只需4字节；相对该行FP4 payload为`8/K`，不能凭空假定metadata负担巨大。实际GEMM消费、cast前scale、bias/LR顺序及SM120时延需要独立验证；补接API或拆GEMM本身仍是工程。排除无效pad统计更不要求第二份权重、历史状态或新的校准目标。

本轮不新建claim、不恢复E059、不修改历史配方。若未来做正确性维护，保留上述问题作为有严格负控制的审计项；当前研究选题应换方向。
