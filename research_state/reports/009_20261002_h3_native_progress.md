# 阶段报告 009：H3 原生基线与融合检查

2026-10-02。**完整H3原生NVFP4基线已验证：当前固定输入的DiT前向相对BF16加速1.22×，前向峰值allocated显存降低58.9%。尚无新的论文贡献或heldout质量结论。**

## 真实性能

本地剪枝版H3，22400个packed tokens；同GPU5、同BF16 SDPA、模型常驻，三种模式各独立进程，1次warmup、3次正式计时、1次独立profile。全部5次输出均匹配各自已验证参考SHA。计时包含native的200次合法性检查；旧QDQ权重仅在启动时解码一次。

| 模式 | 完整DiT中位耗时 | 模型常驻storage | 前向峰值allocated | 前向峰值reserved |
|---|---:|---:|---:|---:|
| BF16 | 8.260 s | 37.46 GiB | 40.86 GiB | 42.94 GiB |
| 旧QDQ | 20.778 s | 37.74 GiB | 44.12 GiB | 49.04 GiB |
| 原生NVFP4 | 6.787 s | 11.95 GiB | 16.80 GiB | 30.09 GiB |

原生相对BF16为1.217×、相对旧QDQ为3.061×。这是原校准prompt1/step0的一次完整denoiser调用，不含文本编码、采样循环或VAE；不同模式数值输出并不相同，不能写成质量等价的视频生成加速。当前BF16是同一eager实现，并非优化上限。native启动仍先加载BF16再转换，启动峰值allocated为37.62 GiB，不能宣称整个加载过程只需16.80 GiB。

独立按Chrome真正CUDA kernel归因：native中attention占49.3%，other占23.5%，原生主支GEMM占14.9%，低秩分支7.0%，packing 2.9%，smoothing 2.4%。other包含残差、norm和最终主支/低秩相加，不能全称为量化开销。52次cu_seqlens元数据读回造成CPU同步等待；实际D2H仅624 bytes，不能把CPU等待的6.6秒算成数据传输成本或额外加到GPU时间上。当前不声称支持CUDA graph。

## 已完成

- 对本地剪枝版H3的200个主干linear导出原生FP4权重。19.27B个元素全部通过旧QDQ数值重建检查；没有重新校准或改变rank32状态。两层token_refiner的8个linear保留BF16。
- 原磁盘offload与常驻GPU的BF16模型，50个block及video/audio输出SHA完全一致；各有102次BF16 SDPA，常驻前向零磁盘加载。
- 四种代表linear的原hook重放一致，原生主干对同codes独立QDQ参考的NMSE约4.4e-6–8.3e-6。快速packing已通过合成边界和四种完整真实输入的codes/global/SF逐字节检查。
- 完整slow/fast原生各执行200次FP4 GEMM、102次BF16 SDPA、零disk load。50block与双输出SHA全一致；200次fast检查无非法值，仅已证明等价的block1.fc2有95组SF0。
- 原生输出对BF16的video/audio NMSE为0.02334/0.04169，旧QDQ为0.02649/0.06878；原生对旧QDQ为0.00372/0.01342。这里只是原校准样本prompt1/step0的数值诊断，不是heldout视频质量。

## 现成融合算子的结果

固定同一份packed输入和权重，加入动态低秩缩放，使用FlashInfer已有SM120 fused-up。下表为M22400的组件中位耗时，**含down/动态rescale，排除packing和smoothing**：

| 组件 | torch原生＋独立低秩 | FlashInfer融合 |
|---|---:|---:|
| QKV | 10.16 ms | 6.64 ms |
| FC2 | 5.68 ms | 5.80 ms |

M512两种形状融合均更慢。四个真实case主干数值与torch一致；融合后的完整linear有约8e-6 NMSE的舍入差异，需整模另验。QKV上的1.53×组件收益不能写成整模加速，也不是新方法。

## 已解决的边界与下一步

严格原生路径完成block0后，在block1.fc2的95个音频激活组遇到E4M3 scale下溢为零，被新增的保守检查阻止。对实际失败输入的3.21亿个元素复验，旧QDQ与E005 canonical-zero解码数值完全一致，仅零符号不同；这些组原本已被旧公式归零，占本层音频输入能量约6.9e-6。原生同codes主干NMSE4.56e-6、全部finite。此为合法域处理缺口，没有改变量化算法；原失败/source保留，用独立adapter修复。

正确性及三种常驻模型的计时/显存/profile均已完成，原BF16/QDQ参考通过哈希继承。E009的GPU任务已退出。后续E010已在GPU0启动：固定两个未用于旧PTQ校准的原始prompt/seed，沿用旧20步协议、原采样器及同一VAE，分阶段完成配对生成；当前尚无其质量结果。稀疏router候选因缺少已验证的完整运行路径暂不执行，不将其记作机制阴性。

KV聚类重排及V低秩跨attention候选均因直接先前工作而park，没有消耗GPU做方法实验。fake/native差异也只作为执行归因，不包装为新颖性。

证据：[性能独立汇总](../../results/research/E009_profile_summary.json)、[完整原生闭环](../../results/research/E009_h3_native_resume.json)、[独立摘要](../06_experiments/results/E009_native_summary.json)、[导出结果](../../results/research/E009_h3_export.json)、[严格域停止记录](../../results/research/E009_h3_full.json)、[真实零scale验证](../../results/research/E009_h3_sf0_domain_v2.json)、[快速packing](../06_experiments/E009_h3_fastpack_validation.md)、[融合结果](../06_experiments/E009_fused_up_results.md)、[候选碰撞](../01_literature/value_lowrank_bridge_collision.md)。
