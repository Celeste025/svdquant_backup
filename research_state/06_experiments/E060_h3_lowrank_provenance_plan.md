# E060 — H3低秩基线来源与迭代语义审计

2026-10-04，baseline audit，不是方法实验。来源观察先于本计划：root/agent已读当前脚本和301MB state，发现当前脚本反复初始化同一量化残差，仅改变randomized SVD seed；saved候选数为2–6。此计划在独立、结构化核验执行前固定，不把先前只读观察伪称盲测。

## 动机 / 待判定问题

H3后续机制研究依赖既有rank32 native基线，不能仅凭目录名standard或max_lowrank_iters=50断言其等同官方SVDQuant校准。理想权重截断SVD的左右子空间正交也不能套到量化残差分解。

H1：现用200份native导出的smooth/A/B就是指定legacy state，后续实验的配方身份可追溯；但当前生成脚本的重置候选策略与官方交替更新不同，须降低“强官方SVDQuant基线”口径。
H2：导出因子与state并不相同，或当前源码语义判断错误，则停止归因并先定位身份。
历史生产源码未绑定是独立限制：即使因子一致及候选数模式吻合，也不能由此证明执行时用了今天这版文件。

## 最小核验

CPU-only，不加载模型、不产生新量化权重。

1. 对quant_state重新计算SHA256，与E009 manifest所存绑定匹配；检查200层、rank32及配置。
2. 逐个mmap打开200个已有export，仅读取smooth/lr_a/lr_b，与state逐byte比较。记录文件大小及manifest中的expected SHA；不重新读完整10GB packed数据，不宣称全部export内容重新hash。
3. 汇总selected candidate与evaluated候选数，报告实际分布，不把max50当实际50轮优化。
4. 保存当前standard脚本、LowRankBranch和QuantLowRankCalibrator源SHA与相关代码段；明确前者每次Q0=Q(Ws)，后者有self.qw状态更新。量化残差初始化本身是官方支持路径；SVD精度/随机近似也另列，不能声称差异必然使质量更差。

## 决策 / 停止

全部因子匹配：保留全部同实现历史数值，称legacy smooth+low-rank residual基线，不能用它单独排除标准SVDQuant能解决问题。只有有限真实输出对照才决定是否值得重新校准，不直接启动整模或改变历史checkpoint。

任何身份不匹配：记录具体层/字段，停止从现用配方向官方方法外推。源码证据缺失：如实标记，不能补造历史记录。

本审计不回答视频质量、支路相消、校准优劣或新颖性；不因通过而新增paper claim。避免以synthetic小矩阵/一层权重MSE冒充完整H3强基线质量验证。预算CPU120秒、0CUDA、0模型前向，独立输出不覆盖。
