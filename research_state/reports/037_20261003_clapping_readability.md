# 阶段037：拍手仍在发生，但低位画面更难读

2026-10-03。E041完成：复用E038两seed、三attention臂的全部六条拍手媒体，逐帧检查，不新增生成或GPU实验。**停止把当前光流下降解释为统一的动作变慢；保留手部轮廓可辨性下降这一具体观察。尚无新方法或可投稿核心贡献。**

E038固定同一套native SVD线性权重，只改BF16/global_mean/coarse16 attention；其中“BF16”仅指attention。既有四动作共16个FP4-vs-BF16配对的RAFT均值全部下降，14对在前、中、后三段均下降。但自由生成的构图、主体尺度和纹理也会变，标量光流不能单独确定动作损失。

本次按固定协议展开每条124帧、24fps原视频，全六条共744帧、48页；CPU渲染26.80秒，0模型/0GPU。两位agent各读一个seed的全部三臂；replica0另补看8张原尺寸静帧。已知臂、逐帧联系图观察，**不是实时播放、正式盲评或两位观察者的重复标注**。root核对观察表及部分联系图。

| seed | BF16 attention可辨完整开合 | global_mean | coarse16 | 能支持的解释 |
|---|---:|---|---|---|
| r0 | 22个 | 至少5个；后段不可靠计数 | 至少5个；后段不可靠计数 | 三臂前段共同闭合相位约4/10/17/23/29帧，没有前段统一变慢证据 |
| r1 | 20个 | 总数未知 | 至少10个，其中前0–42帧9个、97–102帧1个 | coarse前段仍快速开合；global重影使周期不可读，不能填零 |

“开合”仅指投影轮廓分离→靠近/重叠→再次分离，**所有实际掌面接触标签均不确定**。下界不是全片动作数，不能用22比5推量化频率下降，也不能跨不可读间隙拼接或除以全片时长。r1各臂主体尺度和取景不同；两seed都能见低位手掌/手指重影，但没有据此定位QK、P、V或解码器的因果责任。

这也不是宽泛的新发现。Attn-QAT已报告仅attention量化时Dynamic Degree从BF16 .3923降至FP4 .1160，并通过QAT恢复至.3039；因此“attention低位导致动态指标下降”本身已被覆盖。QuantWM的时空token选取错误、VC-Attention的P/V误差修正也必须面对。[限定最近邻复核](../01_literature/post_E040_frontier_review.md)

**后续只保留可定位的读出问题，不新建运动loss或中心网格。** 先用现有源码与媒体排除“重影全部由VAE最后时间拼接产生”这一简单解释；同时核查视频VAE误差敏感性是否已被SSVAE等工作解释。H3已保存音频还可检查事件读出入口，但没有音频事件或同步结果前，不宣称量化破坏音画同步。以上只是问题筛选，不是新的论文claim或GPU计划。

产物：[固定协议](../06_experiments/E041_existing_clapping_readability_plan.md)、[r0逐事件观察](../06_experiments/E041_clapping_replica0_observations.md)、[r1逐事件观察](../06_experiments/E041_clapping_replica1_observations.md)、[媒体索引](../../results/research/E041/frame_index.json)。全帧页和原媒体在DATA1，E038原结果未改。最新资源快照为8张RTX PRO 5000 72GB，0–5空闲、6–7其他任务；本轮未占用GPU。

补充只读核查已完成：[实际decoder上下文](../02_problems/h3_decoder_context_readonly.md) 给出精确17帧主段/5帧blend映射；多个完全非时间blend帧仍有重影，排除“全部由最后时间blend产生”，未定位更广义根因。[decoder最近邻](../01_literature/decoder_quantization_prior_review.md) 表明一般生成latent误差/decoder鲁棒性已由SSVAE研究；自由分叉终态的随机通道扰动又混有内容变化，不足以归因量化，因此当前不执行。另已固定[E042现有PCM读出](../06_experiments/E042_existing_audio_readout_plan.md)，只画原始能量结构与各自已有视觉区间，0模型/0GPU，不先报同步分数。
