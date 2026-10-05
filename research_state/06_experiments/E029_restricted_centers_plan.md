# E029：同步重中心化的原生精度干预

2026-10-03，GPU 前固定。E028 全部504项谱完成：rank16的K4-error剩余能量在block0/24/48为0.89%/8.07%/3.63%，但这没有包含Q重新量化项。下一问题是受限中心能否在真实consumer中保留自由块均值的精度优势，而非以谱代替量化实验。

只复用E018 p36/seed59526/step14的block0/24/48，每层全部56head、22539有效token，补至22656。固定rank16和三种geometry（Euclidean、Kc-score、K4-error-score），加官方block/global两端点，每层5臂共15次native attention，0 DiT、0新视频；不择优丢head/层、不追加rank网格。本例基准误差及按head分布全部保留。

GPU现场按官方顺序得到BF16 padded Q的块均值和全局均值、valid K减均值后padding。δ=μblock−μglobal转FP64，各组按有效Q数加权（128，尾组11）。对C=I/KcᵀKc/EᵀE（valid Kc/E去列均值），Z=√w δ C^(1/2)，取前16个左奇异向量U。δ16=inv√w U(Uᵀ√w δ)，避免C求逆；这是相应几何下的同样本最优rank16表示，不是部署时在线SVD方案。报告谱tail与实际投影残差、BF16中心重舍入后的残差，保存小center/factor工件。

明确重构 c_fp32=μglobal.float()+δ16.float()，一次cast到BF16。同一个BF16 c同时用于Qpad−c的BF16减法与FP32 c×Kcᵀ完整校正；只用已验scaled_fp4_quant(...,1)重新pack Q，四个K/V packets原样复用E018。block/global用官方_preprocess_qkv分别True/False，并经同一Q-only pack。原生fwd保留原softmax/P/PV实现、scale、unpadded长度和输出dtype；实际P值随分数改变。

保存全部15个valid输出，FP64计算对BF16参考的pooled和每head NMSE/cosine及对已有block/global输出的变化；现场端点记录与历史数值差异，不以逐位相同为科学准入。独立CPU复算输出，判断是否值得研究校准basis及真实低秩consumer；不是视频质量实验，也不宣称候选质量优于block。

本实验仍物化完整correction，并保留每块BF16重构中心，BF16舍入使中心矩阵严格秩可能超过16。因此不声称已实现R行consumer、显存/速度收益；实际factor消费的算术与成本需单独验证。当前basis依赖同样本Q/K/error，不声称跨prompt/step稳定。600秒硬预算、单GPU，先CPU结构检查再启动；失败保留，不隐式续跑。
