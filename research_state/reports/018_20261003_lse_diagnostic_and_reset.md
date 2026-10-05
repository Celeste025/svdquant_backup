# 阶段018：关闭分片方向的小审计，转向主权重 QAD

2026-10-03。**本轮没有得到新的量化方法，也没有完成原定分片数值对照。** 实际完成的是一次失败定位：不同 LSE 返回路径带来极小的输出差异，严格 byte gate 不适合直接跨模板使用。累计3次 attention、0次完整DiT、0视频；GPU5任务均已退出。

## 尝试与实际结果

原E019计划固定H3真实Q/K/V量化包，比较完整与两片KV执行，包含已有PoT reference、Sage tile scaling、PNQ对照。CPU物理packet切分/重接与选定head解码检查通过，数学脚本解析自测通过；但首个 BF16 full＋LSE 调用没有逐字节复现E018，原队列按协议停止。**两片调用、真实score数学分析和模态mass统计均未执行；不能写成分片机制阴性。**

随后只固定block0输入，切换 return_lse=False/True 各一次并保存完整输出。独立CPU复算得到：

| 返回LSE | 相对E018的变化元素数 | 相对误差能量 |
|---|---:|---:|
| False | 0 / 161,559,552 | 0，逐字节复现 |
| True | 39 / 161,559,552 | 2.84546×10⁻¹² |

True−False 的 RMS 为1.58684×10⁻⁶，最大绝对差0.0078125；两条路径输入codes、scales、correction一致。源码确实切换CUDA模板实例。这定位了差异与返回LSE路径的关联，但未追到具体机器指令，不称算子bug、研究新机制或生成质量问题。39个坐标及数值全部保留。

原失败输出在断言前未保存，这一缺陷已在独立诊断中修正。诊断第一次CPU预检发生文件记录解析错误（0调用），随后GPU尝试又被重复的“显存必须绝对零”检查拦下（3MiB、0调用）；均保留原记录。新v2只将worker检查改为排除其他计算进程，调度器稳定空闲检查及数值参数不变；两次诊断14.12秒完成，独立CPU复算1.33秒。

## 结论与后续

停止扩展当前P4分片方向，依据是[直接先前工作碰撞](../02_problems/fp4_attention_partition_composability.md)，而非这次未完成对照的数值结论。现有工作已直接讨论E4概率尺度范围与稳定化成本；“模态V均值＋精确分组质量修正”也被VC-Attention覆盖。保留原协议、源、自测和失败，不追加kernel/多卡框架。

同时承认流程问题：近期反复“找算子边界→小审计→park”，审计负担过重，尚未形成可以持续改进的模型。[流程复盘](../00_state/process_reset_e019.md)决定下一阶段只推进一条主线：**rCM-Wan主权重QAD，导出后无在线LR的真实NVFP4部署**。普通QAD不是新方法，但应先建立这条强基线，观察训练收益能否兑现为独立视频的质量—成本改善，再由真实失败选择研究贡献。

只读[实现检查](../00_state/implementation_readiness_wan_mainweight_qad.md)确认：模型实际1,418,996,800参数；300个目标矩阵共1,391,984,640参数。现有trainer只训练LR或LR＋smooth，不能当主权重训练已完成。下一步新增薄STE训练模块与无LR导出/安装入口，复用teacher缓存和原生GEMM；固定四步采样与BF16 attention，以独立完整视频、延迟和显存为主读出。尚未写训练器或开始训练，当前仍无可投稿核心贡献。

复现记录：[原失败](../../results/research/E019/probe_run.json)、[两调用诊断](../../results/research/E019/lse_diagnostic_v2_run.json)、[独立CPU复算](../../results/research/E019/lse_diagnostic_independent.json)、[原计划](../06_experiments/E019_h3_partition_contract_plan.md)、[补充诊断协议](../06_experiments/E019_lse_diagnostic_amendment.md)。
