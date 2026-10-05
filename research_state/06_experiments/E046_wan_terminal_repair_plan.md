# E046：完整 native 轨迹的 BF16 末步修复强控制

2026-10-03，exploration / baseline control。上轮E045为progress：同teacher状态末步SVD已产生碎片，但不能推出完整量化轨迹可被单步修复。DSAQuant已有CFG误差/晚步drop直接近邻，PTQD已有按步提精度；本轮不是新方法或新颖性验证。

问题：已经经历完整SVD native前49步的状态，只将最后一步换回原版BF16、保持CFG6，能否修复现有缺陷？固定E043/E044四动作×两seed全部八例，真实旧noise/embedding、81帧480×83216fps、50步UniPC/shift8、FP32状态与VAE。八例均已观察，属于诊断集，不冒充独立泛化测试。

每case只运行一次完整native50步，在最后两支DiT的量化hook前捕获并clone原始args/kwargs，在最后scheduler.step前保存sample、CFG和完整history（CPU sigmas/CUDA model history保原设备，排除wrapper闭包）；capture与native终态先保存，再做诊断signature。native终态为native_full。另载入独立原版BASE BF16 Transformer，用相同已保存输入、timestep、条件跑cond/uncond，保持原BF16合成CFG6顺序，在复制的原pre-last UniPC状态上执行一次原class.step，得bf16_last。不能使用已更新scheduler历史、native_final反推输入、SVD smooth后的输入或残留其非目标PTQ的伪BF16模型。整个部署配方最后一步被替换，不能归因单独FP4/LR。

两终态各用相同官方FP32 VAE、once mean/std denorm、无tiling/slicing、原媒体编码输出完整视频和九帧图。16新媒体全部独立MJ四主项及AMT/RAFT/DINO，与同轮native_full配对；E044旧native终态/分数仅作数值复现参考，漂移如存在如实报告，不设逐位科学gate，也不替换这轮实际baseline。原E043 BF16质量仅作上下文，不把native与BF16不同轨迹差当同语义MSE。MJ不是概率，RAFT不是动作成功率，AMT高不自动等于好。

资源：GPU0/1/5，四worker每prompt两个seed，0先161后316，1为192，5为269；每worker1800秒，总3600秒，实际空闲检查/命名tmux/独立进程组及期限，只清理己方任务。每case100native+2BF16DiT、30000native主GEMM、6120DiT SDPA、50原scheduler+1BF16替换、2VAEdecode。总816DiT/240000GEMM/48960SDPA/408scheduler/16decode/0TE/0训练。大文件/cache均DATA1，失败及执行源码保留，修复新版本。

BF16备用模型每worker只载一次，原native模型保留GPU，记录实际原版加载/H2D时间、增加的allocated/reserved、参数/缓冲区storage与两模型同时驻留峰值。分支/加载/完整worker时间均含各自诊断开销，非成熟部署benchmark；2/100 DiT比例不能当2%延迟或零内存开销。此直接双驻留控制不代表最优切换实现。

决策：若多数质量/可见碎片恢复，先将单步保护记为强实用基线，检查剩余部署成本和真实残余缺口，不发明decoder或CFG损失；若改善有限/混合，保留全部样本，仅说明保护一步不足，不能自动证明需要新机制，也不反证已有CFG公式或所有晚几步保护。无CFG-drop臂、无末步数扫描、无训练或新量化器，结果出现前不调整阈值/样本。后续只有新的、未被强基线覆盖的可检验残余才进入候选claim流程。
