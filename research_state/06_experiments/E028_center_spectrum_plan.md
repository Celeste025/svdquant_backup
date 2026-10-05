# E028：受限query中心的三种几何，只测表示能力

2026-10-03，CPU计算前固定。E027已证明已知query分块能以本例约7%时延代价降峰值；朴素在线BF16校正不值得为当前形状直接重写调度。转而问是否可改变中心的表示复杂度，而非搬运自由block mean产生的完整矩阵。候选c_i=μ_global+A_iB同时用于Q−c_i的量化及c_iKcᵀ校正；未量化代数恒等式保留。没有新方法claim，Sage/globalmean、MpFA及已公开的query-basis低秩logit侧码是强近邻。

只复用E018同p36/seed59526/step14的block0/24/48，全部56heads，Q/K原BF16数据和真实K4 packets；不新增prompt、生成、GPU或训练。各head ΔM=μ_block−μ_global，原128分组/补零均值合同；Q尾组11有效行，其他128。不按有效Q权重再重中心ΔM，以免改变R0端点。CPU复现BF16均值/减法后作FP64矩统计，不称CPU/GPU均值逐位相同。Khat从已验证真实E2M1/E4M3包与scale/行布局还原。

三个固定几何：①Euclidean：C=I；②score：valid Kc去列均值后的KcᵀKc；③K4 representation error：E=Khat−Kc，valid E去列均值后的EᵀE。对PSD对称C=VΛVᵀ，计算Z=diag(sqrt(w))ΔM V sqrt(Λ)，直接SVD，避免求逆和显式大score矩阵；仅容舍入量级负λ并记录原minλ。报告每head完整σ²与r1/2/4/8/16/32/64/128剩余能量、层内energy-weighted汇总和head分布。

第三种几何的理由：同步改center后，分数差包含(μ−c)(Khat−Kc)ᵀ以及Q4舍入变化[εQ(c)−εQ(μ)]Khatᵀ。仅用高精度score几何评价中心会漏掉这个结构；E028也没有测第二项，故不能以任何一条谱直接证明量化质量变好/变坏。各几何basis依赖当前Q/K，是同样本oracle；三层不是跨prompt/step/length泛化。

本轮只测表示能力，不保存大tensor、不部署basis、不扫生成指标。CPU预算300秒、CUDA_VISIBLE_DEVICES为空。谱若集中，只决定下一次正确重中心化的原生精度干预是否值得；若不集中，也需结合所测具体几何说明停止范围，不能把Kscore谱当所有Q4方案的下界。R=128近于原D，177组仅降128，不能宣传大幅节约；小R consumer读取/组合多条校正的实际代价仍未实现。
