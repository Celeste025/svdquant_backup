# E031：自适应中心的实际构造成本与原生精度

2026-10-03，GPU执行前固定。上一goal turn为progress：E027–E030完成成本、谱、原生干预、跨文本迁移及独立复核。当前无外部阻塞，目标仍active，无可投稿贡献。

问题：同样本rank16中心精度较好，但单文本固定basis迁移明显退化。若每输入自适应basis，其构造成本是否已经抵消压缩校正的机会？只量这个问题；经典SVD/随机range finder不是新算法，不将低秩结合律作为贡献。

数据只用已有E018 H3 p36/seed59526/step14，block0/24/48全部56heads。N22539/NP22656、G177、D128、rank16。GPU按官方BF16顺序计算padded Q块/global均值及valid K减均值后padding。δ=μblock−μglobal在FP32作差，X=√wδ，w为176×128+11。TF32关闭。

固定三臂：
- full_svd：GPU FP32 torch.linalg.svd(X,full_matrices=False)，B=Vh[:16]；不用近似gesvda。
- range0：Q0=qr(XᵀΩ).Q，B=Q0ᵀ。
- range1：Q0如上，Q1=qr(Xᵀ(XQ0)).Q，B=Q1ᵀ；一次幂迭代，包括两次QR，不物化Gram。

Ω只用CPU seed20261031生成一次标准高斯[56,177,16]，各head独立子矩阵，同一个冻结batch用于三层、两range方法和全部重复；保存工件，不按结果重抽。Ω/√w在计时前驻GPU，随机数生成不计入该构造成本。固定rank、无oversample/seed/迭代网格。

最终A=δBᵀ（不能用X替代δ），c32=global.float()+AB。Qcenter=(Qpad.float()−c32).BF16，T17=[global;B]Kcᵀ FP32；原生精度阶段在计时外物化correction=T0+A Tb，与E030可分解FP32合同一致。固定原KV packets，只重pack Q，三层×三臂共9次native attention、0DiT、0新视频。保存完整输出及小A/B/c32，报告全部head对BF16/block/global/E029同样本Euclidean/E030固定FP32的变化及投影残差，不择优丢层/head。

每层两类成本各3warmup+10repeat，不用CUDAgraph/compile：
1. resident GPU BF16 μ/global → δ/X/basis/A/c32。包括SVD/QR及全部构造，没有CPU basis或CPU SVD。
2. resident HND-contiguous有效长BF16 Q/K → Qpad/均值、K-valid mean/sub/pad、basis/A/c32、Qcenter、T17。包括K.transpose.float必要复制；不偷缓存Kc/Kᵀ.float。该范围不包括输入layout转换/上传、V处理、任何FP4 pack、A Tb完整correction物化或attention。

范围2另含global/block强预处理baseline：同输入Q/K、官方BF16center/pad顺序，输出Qcenter与其真实1行/G行FP32 correction。其产物不同，记大小/生命周期，不把两种预处理相减预测完整consumer速度；E027的31.780ms完整QKV→output也不是这里同一边界。

同时记录CUDA event与同步wall时间，每scope基线allocated/reserved、reset峰值及增量，清理前scope输出；hash/finite检查、CPU指标、随机生成都在计时外。小工件与结果明确device/torch/CUDA、计时调用次数和边界。全GPU0/600秒预算，CPU结构检查和独立源码审查后root唯一tmux启动。失败原样保留，不静默续跑；独立CPU复算9输出与计时汇总。

结果只决定是否值得下一项真正consumer工程验证。即使prep快且原生误差接近oracle，也还没有完整consumer速度、固定质量视频收益或新颖性；若这两者没有同时出现可用的取舍，停止为本中心构造追加网格。
