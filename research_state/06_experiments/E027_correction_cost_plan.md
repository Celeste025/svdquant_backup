# E027：块均值校正的真实成本与query分块强基线

2026-10-03，启动前固定。上轮E026是实质进展：固定μ复用K4的误差增加，但不构成所有融合方案的下界。本轮问题是保留原BF16 μ/Kc输入时，消除二次correction中间量的实际成本是否值得进一步开发。

固定E026同一H3真实层：p36/seed59526/step14/block0，N22539、pad22656、56head、D128、非causal、原softmax scale。复用E018 QKV与E026原输出，不loadDiT、不新视频。保持原BF16 K全序列均值、128-Qmean、padding、原生FP4量化与attention合同。

**强基线实验。** 两臂是完整物化correction与query chunk4096（128/64对齐，5块4096+1块2176 padded，末尾有效2059）。不扫描chunk。显式复用官方预处理次序但不调用其会先物化完整correction的函数，三次官方Q/K/V pack只做一次；每chunk只生成自己对应的μKcᵀ，再调用原consumer并组装输出。K全局mean/KV packets/分组相同，Q与scale切片所需连续复制计入成本。报告两种边界：①已pack的correction+attention（包括GEMM、转FP32、切片、launch、输出组装）；②已驻留原QKV到output（额外含一次公共preprocess+pack）。每臂3warmup+10重复，CUDA event、同步wall与allocated/reserved峰值；保留原始样本和运行顺序，未使用CUDAgraph。测实际输出相对E026原block差异，不设逐byte输出gate，不称新方法。

**独立BF16 stripe成本。** 源码检查确认原consumer广播已融合；在线版还需BF16 K staging/同步、M16 MMA与广播。直接沿用当前3stage会多约96KiB shared memory，但这是朴素照搬的成本，不是所有调度下界。先测真实BF16 M16×N128×K128、每Qgroup一条有效μ加15填充/重复行，覆盖全56×177 Qgroup与完整K长度；3warmup+10重复，核实际PTX含BF16 MMA并检查FP32 accumulation输出。独立kernel包含自己的输入读取/输出存储、没有FP4 consumer资源竞争；它不是融合时延、不能与baseline用max/sum拼接出加速结论。

两个任务均GPU0顺序运行，启动前检查空闲，各600秒硬上限；root监督，所有大文件/JIT/cache在DATA1，源和报告在仓库。systems实现stripe，audit实现强baseline，root审阅并负责解释/阶段报告。若原query分块已解决主要容量问题且朴素在线代价不吸引，不为该实现残余重写整套调度；阴性不淘汰其他融合方案，也不包装已知分块为贡献。
