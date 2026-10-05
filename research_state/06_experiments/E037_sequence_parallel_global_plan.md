# E037：两卡完整边界的 global-Q/T1 强对照

2026-10-03，正式运行前固定。上一goal turn是progress：E035/E036和独立复核均complete，现场进程已退出。E036同coarse16数值合同下FP4+T16快于mixed K16，但遗漏更便宜的现成global-Q路径，尚不能支持方法贡献。按D058只补此对照，不扩中心/卡数/拓扑网格。

固定E036同一E034 block0（原E018 native轨迹）QKV、N22539/NP22656/H56/D128，token分片11264/11392，head各28；GPU0/1同NUMA，正式前重查。没有新捕获、DiT或视频。两臂：

1. fp4_table16：完整复用已执行E036 Execution，8+8连续coarse BF16中心交换、valid K FP32 sum AllReduce、原BF16 K校正T16，原E033 private consumer。
2. fp4_global：原global-Q量化合同的分布式实现。每源对padded Q和valid K分别求FP32 sum，将两组sum合并为一次AllReduce；分别除NP22656/N22539后转BF16。原zero Q padding参与global Q中心，Qcenter padding=-C；K只对有效行去均值、padding回零。源侧生成Q4/K4/V4及C.float()@Kc.float().T的T1列，按head交换原packet/SF及T1。head-owner使用原官方nvfp4_attention_sm120_fwd(per_block_mean=False)，有效K22539；不同center合同允许不同官方consumer分支，不冒称完全同算术输出。

使用同冻结transport完成Vsf物理tile拼接、forward split-size A2A以及共同逆BF16 output A2A。真实scope与E036相同：resident分片BF16 QKV到恢复原token/allheads的有效BF16输出，全部统计/通信/必要buffer复制与重组/pack/T/attention/逆输出均计时。临时dense/中间量在不再需要时释放；保留每rank驻留/allocated/reserved/增量峰值。CPU记录/保存、入口control barrier及初始化/JIT在稳态计时外，总900秒硬限时含冷启动。

每臂3次warmup，再10轮AB/BA交错顺序；每rank26attention、65数据A2A、26AllReduce，合计52native/0DiT。主指标每轮max(rank0 wall,rank1 wall)的median与range；保留每rank CUDA events，不累加组件时间。共同输入、固定id驻留、线程/TF32与NCCL默认transport沿E036，不额外调参数。

global的前向FP4主数据同130.67578125MiB，T1约2.419922MiB，逆output154.875MiB；没有coarse中心交换，合并Q/K AllReduce逻辑peer send合计0.109375MiB。计数以实际dtype/numel/split记录复核，不把API逻辑载荷当NCCL协议链路流量。T16账本应复现E036。

CPU先核对冻结E036 source/check/run绑定及当前输入引用、T1形状/wire字节重组/中心有效分母等新增合同。已验证的transport不要重新搭框架；不进行新的toy GPU或额外正确性attention调用。run每rank保存最后一次两臂有效token输出、小C/Kmean及完整计时/内存/collective收据至DATA1/results。独立CPU合并rank输出：T16对E034 coarse16、global对E034 global分别报告漂移；两者对同BF16参考作pooled/perhead误差并与成本并列。分布式sum与整序列mean归约次序允许数值漂移，不以跨实现byte相同作为科研门槛，也不要求两个不同中心合同相同。

本次只判断精度—完整成本边界。global-Q/T1本身是强现成方案；T16若只体现普通粒度取舍，不升级为新方法。单层NMSE不代表完整视频质量；与E036/E034历史计时若有差异照实记录，主比较只用本轮配对数据。只有留下明确的实际需求和未覆盖约束才考虑整模/视频或新的机制实验，不能为维持阳性故事扩搜索。
