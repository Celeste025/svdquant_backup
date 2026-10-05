# E040 后：目前没有新的系统方向 GO

2026-10-03；只读本地源码、实际结果和已完成的近邻审查。未运行GPU、安装依赖或修改执行源。**本轮没有找到一个同时具备真实失败证据、成熟替代未覆盖、且超出小算子优化的剩余问题。** 这不是断言该领域没有机会；是当前证据不足以再指定一轮系统开发。

原 phase4 的两个信号已被后续实验消化。LR开销有plain/QAD、现成融合和E039/E040消费接口对照；E040 decoded-X相对原输出的valid NMSE为1.83e−6，不能靠side相对更小的5.93e−8宣称高精度信息在视频中必需。二次correction缓冲已有global、query chunking和coarse16的完整路径对照；E038中coarse相对global的MJ均差−.012914、跨动作/seed不一致，不支持用局部attention误差优势继续证明质量需求。相关成本是事实，额外融合、更多卡、换布局或长序列放大都不会单独改变新颖性判断。[phase4原判断](../00_state/phase4_systems_signals.md)、[E038](../reports/034_20261003_center_video_results.md)、[E040](../reports/036_20261003_projection_information_control.md)

**唯一仍未完整实测的合同是P4在KV分区下的可组合性；不推荐以它重新立项。** 本轮重新读到的实际代码仍是：先用完整N128 score更新running max；未量化P进入row_sum；P的FP32组scale另转E4M3后才进入PV。这意味着其分子与返回LSE不是同一represented-P归一化合同。源码位置是本机FlashInfer的`compute/consumer/softmax.cuh:317–413`、`compute/mainloop.cuh:778–849`及`compute/epilogue/lse_writer.cuh:80–86`，根目录为`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/`。Python入口`nvfp4_attention_sm120.py:397`已支持外部packed QKV、独立KV长度、LSE，故单卡分片重放没有ABI阻塞，也无需先写Ring kernel。

E019不是这个机制的阴性证据：原任务在第一调用因`return_lse`不同模板的极微输出漂移停下；full/split实验和真实行数学归约没有完成。已保存block0的72条score重建材料，另有LSE诊断输出。不能把`math_selftest*.json`当真实H3归约结果，也不能把恢复此审计称新发现。[E019原结果](../../results/research/E019/probe_run.json)、[LSE诊断](../../results/research/E019/lse_diagnostic_v2_run.json)

但最简单的强替代已有：保持canonical 128-token tiles，以Sage论文的逐tile FP32因子移出低位P编码；正常数值范围下还可用既有整数log2参考max。PNQ须使用匹配的represented-LSE，BF16 FlashAttention则是精度/容量基线。本地已核的Hardware-Aware FP4 FA4甚至明确讨论过E4 scale范围稳定化与关键路径成本冲突；VC-Attention覆盖量化前块概率质量的V均值修正。因此“分片有差异”“PNQ没有保住模态份额”“SM120需要实现这些控制”均不足以产生新方法。[现有逐项近邻与推导](../02_problems/fp4_attention_partition_composability.md)

若日后出现**真实必须分割KV的部署需求**，最小证伪不是扩大prompt或重写kernel：先只用已存block0材料，在相同tile/microgroup、score和V下比较当前数学合同与上述已知tile-factor控制，检查full/split差异是否还有不能由范围/归约解释的部分；必要时才用现成packed API补一次原生full/split。若差异被已知控制解释，结案为合同维护。只有正确强控制下仍有具体遗漏，并且该遗漏在实际需求下迫使不可忽略的质量—资源取舍，才值得重新查新。**当前没有这条证据，因此不安排该实验。** 这里缺的是一个超出已知机制的使用约束，不是更快的P转换或更漂亮的单层NMSE。

结论：保留现有native部署作为研究基础；本次不登记新claim，不重启C006、P分区或center网格。普通动态global/尺度同步、缓存ownership和side-channel ABI已分别有现成设计或本地验证；未实现最优融合属于实现差距，不能替代新的科学问题。
