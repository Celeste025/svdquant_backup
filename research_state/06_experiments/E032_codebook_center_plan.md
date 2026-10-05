# E032：共享中心表，保持每tile只消费一行校正

2026-10-03，GPU前固定。E031完成9native/0DiT：精确GPU SVD构造约211ms，range0/1约1.84/3.61ms；现有rawQK准备10.35/12.15ms，比free-block 8.43ms慢。稠密basis还需要未来consumer每tile组合17行。先比较更简单、同样有表示约束的强基线，避免先实现复杂consumer。

中心改为有限表C[H,16,D]，每Qtile保存一个id[H,G]，Q−C[id]与C Kcᵀ使用同一实际BF16 C。未来consumer只需按id挑一条校正行，保持原单行TMA流量/barrier/consumer，而不做16项FMA。当前仍在旧kernel前展开完整correction用于精度，尚未实现新consumer。聚类/one-hot代数不是新算法；Clustered Attention已有centroid×K共享，VC-Attention已有聚类与去均值低位消费，具体区别记入近邻文件，不按检索未找到就宣称新颖。

只用E018同p36/seed59526/step14、三层0/24/48、全部56heads，N22539/NP22656/D128/G177，w为176组128和尾组11。GPU官方BF16均值、K-valid mean/sub/pad合同不变，δ=μ.float−global.float。

固定K16、6轮Lloyd，没有网格或采样筛选。init Cδ[0]=0（仅初始化，不永久固定global），再15次从真实δ行选取argmax(w*到当前中心的最小平方距离)，ties由argmax最低index决定。所有head独立、无随机数/CPU自适应控制。每轮E-step普通Euclidean最近中心（每组w对所有中心相同，不参与argmin）；M-step按w加权平均δ，用one-hot weighted assignment和batched matmul，空簇保留原中心。FP32距离用norm+batched MM并clamp数值负值至0，避免[G,K,D]大差值张量。

6轮后C=(global.float+Cδ).BF16一次cast，按实际μ/C重新nearest assignment，更新id但不再M-step。Qcenter为BF16 Qpad减同一BF16 C[id]；T16=C.float Kc.floatᵀ。accuracy阶段在计时外按id gather出G行完整correction，只重pack Q、复用原KV packets，3层共3native/0DiT/0新视频。保存C/id、小簇统计及全部输出，报告逐head和pooled对BF16/block/global、E029oracle/E030transfer/E031range1的取舍；不挑层/head。

成本同E031两scope各3warm+10repeat：resident BF16μ/global→C/id（init与6迭代/最终cast重assign全部计入）；resident HND BF16 Q/K→官方mean/Kc、C/id、Qcenter/T16。后者不含layout转换/上传、V、FP4 pack、完整correction gather或attention。同一次运行重测global/block rawQK prep基线（无额外attention），以避免跨运行细小计时偏差。基线产物为Qcenter+其真实correction，候选为Qcenter+T16/C/id，记清bytes和lifetime，不相减冒充完整系统收益。

记录CUDAevent与同步wall、暖allocator基线allocated/reserved、resetpeak与新增量；每repeat输出在下一次前释放。finite/hash、empty簇、postcast distortion统计均在timer外。全GPU0/600秒，root单一tmux启动，CPU结构检查及独立源码审查在先。全部原始结果/失败保留。无理论质量保证、无新kernel、无speedup/video质量claim。只有具体精度和成本显示值得，才封顶实现一种真正单行consumer，不扩basis/cluster网格。
