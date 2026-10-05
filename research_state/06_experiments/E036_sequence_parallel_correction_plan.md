# E036：同 coarse16 合同的两卡完整 attention 边界

2026-10-03，正式运行前固定。E035已完整闭环，预算膨胀故事停止；本次来自[N1真实所有权核查](../02_problems/sequence_parallel_correction_dependency.md)，不重新命名稀疏故事，不预立新方法claim。问题：源侧原生FP4打包能否在保持BF16-K校正语义下减少真实两卡attention边界耗时，还是中心交换／K统计／T生成串行依赖抵消载荷收益？普通分布式matmul重排本身不是创新。

只用已有E034的block0（原E018 p36/s14 native轨迹）post-RoPE QKV；不是E035同输入配对数据，也不新增teacher捕获、DiT或视频。N=22539，Npad=22656，H=56，D=128。源R0 tokens[0,11264)，R1[11264,22656)，各有全部56heads；最后117是原padding。head-owner R0取heads[0,28)，R1[28,56)。128对齐不等长token split使用显式split-size NCCL A2A，不冒称现成等长产品集成。

三臂均沿E034 fixed coarse16边界 floor(j*177/16)×128，直接对padded BF16 Q取均值，最后中心包含117零。valid K先减序列BF16均值再pad零；校正为C.float()@Kc.float().T，TF32关闭。复用E033 private共享行consumer，D128/BF16/noncausal，无LSE；K有效长度22539。固定center IDs作为静态metadata，在所有臂驻留。

- bf16_a2a：交换原BF16 QKV后，在head-owner生成C、Kmean、全部packets和T。
- mixed_k16：源侧生成本地8个C、Qcenter4和V4，交换Q4/V4、原BF16 K及各head对应C；owner合并16个C，按valid K生成Kmean/K4/T。BF16 K替代K4，不重复传两者。
- fp4_table16：源侧交换两组各8个C，并对valid K的FP32 sum做AllReduce；除22539转BF16。源侧生成Q4/K4/V4与全部16个C对应的本地T列，按head交换packed数据及T。统计归约次序不同于owner本地mean，数值差须完整报告，不设置跨算法byte一致门槛。

全路径计时从两卡resident token-shard BF16 QKV开始，到原生attention后**逆BF16 A2A恢复原token/all-head ownership**结束。包括中心/均值统计、所有通信、必要连续复制、打包和scale swizzle、T GEMM、attention和输出回传。I/O、冷JIT、初始化进程组与固定metadata上传不在稳态计时；冷启动仍受总900秒期限限制。主计时为每rank同步wall及两者max，另保留GPU event、峰值allocated/reserved与驻留基线；不将单卡组件时间相加。

采用0/1同NUMA空闲卡（最近topo=NODE，无NVLink，启动前重查），一named tmux，NCCL原生默认transport，记录版本/拓扑。3次warmup/臂，再10轮每轮三臂，顺序按ABC/BCA/CAB轮换；每rank每臂13次native，总78次attention、0DiT。每轮入口同步在计时外，结果统计/跨rank汇总在计时外；单臂内所有真实collective均计费。保存最后一次本地输出、实际中心/Kmean与原始计时数组，执行计数逐rank记录；大tensor放DATA1。

先CPU检查实际E034输入与冻结consumer、split/centers/有效长度及wire拼接合同。Q/K scale按物理row tiles，V scale按64-row/4-col tiles拼接，不能逻辑维度直接串接；复用既有E019 slice/rejoin知识但不改已执行源。CPU带标签字节验证分发/重组与逆输出映射，无CUDA。正式GPU直接跑实际shape，不另开toy性能网格。数值比较三臂输出及E034同coarse参考、BF16参考，报告每head/pooled误差，不要求跨归约byte一致。若出现无法解释的布局/数值异常，先保留失败证据并诊断，不报告速度收益。

实际通信账本区分remote双向发送与self-copy/接收：前向BF16 QKV464.625MiB；mixed241.992MiB+约.109MiB C；fp4三包130.676MiB+38.719MiB T+.219MiB C交换，另小K统计；共同逆输出154.875MiB。这些仅是理论payload，正式run记录实际dtype/numel及每collective remote字节。两卡实际attention形状[1,28,22656,128]第一次运行，既有56heads验证不冒充新shape实测。

本次只决定是否有值得继续处理的实际关键路径。global T1、K4近似T、QAD无中心已是不同数值合同的强参照，但不混入三臂同coarse合同速度排名；未来若方法值得深入必须面对它们及视频质量。无新kernel、rank/中心数/分片/拓扑网格，无完整模型加速或同视频质量claim。硬期限900秒含首次NCCL/JIT；失败终止全部子进程，完成释放两卡。
