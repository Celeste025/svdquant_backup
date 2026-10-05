# P002 批独立性：有限碰撞

2026-10-02；最多三份 primary 来源，CPU/web-only，无 GPU。**结论：KILL 将“批独立推理 / 按请求分离 global / grouped GEMM 或行 epilogue”本身作为新贡献；P002 的真实服务成本与质量问题仍无证据，不为此新增实验。** 本次不能声称所有视频 NVFP4 serving 已解决完整 batch invariance，也不能由某个 API 缺参数反推研究空白。

## 已知事实与接口边界

| Primary 来源 | 已核实事实；边界 |
|---|---|
| [Thinking Machines，Defeating Nondeterminism，2025-09-10](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，batch invariance / matrix multiplication / attention 段 | 已明确区分单次执行可重复与请求对批组成/批大小不变，并处理固定 reduction、MMA/tile 及 attention 分块。它主要解决算术执行路径的变化，**不等于已经处理动态共享量化 amax**；但“同请求随无关 batch 改变输出”及一般批独立性不是新问题。 |
| [FlashInfer 0.7.0，nvfp4_batched_quantize](https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_batched_quantize.html) | 输入 `[B,M,K]`，每 batch 有 swizzled block scales，但 `a_global_sf` 文档形状仍是 **`[1]`**。不能因名称 batched 或输出含 B 就声称支持 B 个独立 FP32 globals。量化器接收调用方提供的 global，接口本身也未强制调用方采用本 batch 的动态 amax。 |
| [FlashInfer，nvfp4_quantize_per_token_cute_dsl](https://docs.flashinfer.ai/generated/flashinfer.quantization.kernels.nvfp4_quantize.nvfp4_quantize_per_token_cute_dsl.html) | 已有按 token/row 独立量化尺度，输出 FP4、E4M3 block scales 和 FP32 `[M]` per-token scales；另有外层 scalar global。**细于请求的尺度隔离已经是公开原生量化能力。** 固定外层 global 时可避免把其他请求的 amax 当本行统计，但本轮未核实本机 SM120 GEMM/低秩 epilogue 对该返回格式的完整消费合同。它也改变现有 per-request tensorwise recipe，不能宣称无需验证就逐位等价。 |

P002 旧结论保留：在 scale 正常且不越界时，global 的纯 `2^k` 倍率可由 E4M3 指数吸收；非二次幂可能改变 scale/codes，但不保证质量更差；极端下溢合成控制不是服务证据。E009 单请求中的 SF0 也不能替代跨请求归因。当前完整原生 H3 仍是固定单请求协议，没有动态 batching trace、SLO 或真实同批请求分布。

## 工程解法与剩余判断

对请求 i 独立计算 `g_i`、pack，并以独立/分组 GEMM 消费该 metadata，是直接的统计域隔离；共同主 GEMM 后按请求/行恢复 FP32 scale，则是标准线性分解与 epilogue 组织问题。需在 cast 前正确处理 scale、bias、低秩支路；BF16 输出后另乘一次不保证与原分请求合同同舍入。**这是工程判断，不是上述文档已经证明本地 torch API 一次调用全部支持。** 旧审计确实发现本机 torch 两级 recipe 只接受 TensorWise；移植/补接这一接口不足以立题。完全逐位批独立还需处理 GEMM/attention reduction，隔离 global 并不单独保证它。

没有核实到同时满足“实际视频动态服务需要批独立、当前原生配方确有有害跨请求耦合、现有分段/按行/固定统计域解法造成不可接受成本”的证据链。即使之后发现两个 batch 的码值不同，也只说明实施合同依赖上下文，不能直接称质量失效；而按请求 scale 恢复独立性也不自动产生方法贡献。**因此当前残余只是未验证的部署需求与成本，不是一个值得 GPU 立项的新机制；不扩建 serving 或新 kernel 来制造这个需求。**


## 2026-10-03 补充直接近邻，不重开 P002

[humans&，The 4-bitter Lesson，2026-07-10](https://humansand.ai/blog/nvfp4-rl?v=3) 的 Per-token Activation Scaling 段直接说明：共享 tensor-global FP32 scale 会使一个 token 的量化依赖同批其他 token；同序列未来位置还会影响更早位置的 scale。其采用每 token FP32 scale 加 group16 的 FP8 block scale，并链接 TE、cuDNN、FlashInfer、SGLang 实现。原文以 LLM/RL 为对象，不证明本地 SM120 视频路径已完整支持，但已直接覆盖所设统计耦合及普通按 token 分离解法。此补充强化已有停线，不新建候选或 GPU 实验。
