# E026：复用 FP4 K 的块均值校正上限

2026-10-03；实施前固定。利用 E018 已存真实 H3 p36/s14/block0 的 post-RoPE QKV、phase0 block-mean packet 和 BF16/block/global 输出；零 DiT、零新视频。本实验由[附加query行的窄核查](../02_problems/query_mean_augmented_rows_check.md)导出，测一个明确数值障碍，不立新方法claim。

原score为 Qhat_c Khat_c^T + μ Kc^T。若在attention内借用K4 tile生成校正，即使μ无限精度，仍新增 μ(Khat_c−Kc)^T。将官方七元组前六项固定，只把最后FP32 correction换为μ Khat_c^T，通过相同原生FP4 attention运行；P量化/PV实现固定，实际P及其codes随score合法变化。首轮只做原packet与无限精度μ oracle，不增加一段/两段μ编码网格。K4解码沿已有验证helper与真实scale布局，不能把dequant结果误认原始K。

用完整有效序列、所有head报告：去除key方向行均值后correction差与原correction能量；attention输出相对同一BF16 reference的NMSE/余弦与逐head分布；与已存global-mean、原block-mean参照并列。原packet重放用于检查运行路径，无跨实现逐byte前置门槛、无任意百分比继续门槛。若无限精度μ仍明显失去已有block-mean的精度优势，则双段μ不能解决主要障碍；将考虑保持高精度K的计算/读取成本，而不是继续优化μ码。若差异小，才评估附加MMA能否换回物化张量的实际成本。局部数值不能推出视频质量、可投稿性或实际加速。

预算：GPU1单次任务最多600秒，等待E025时序评价结束后由root启动；复用原native环境与cache，所有大工件DATA1。只两次原生attention前向和必要GPU解码/矩阵运算，不loadDiT，不新profile，不新完整生成。结果新目录 results/research/E026；旧E018源、packet、输出不改。

结果解释补充（完成后）：本实验的oracle仅保持官方实际BF16 μ、不再增加μ编码误差，不能认作允许重新优化μ或其他结构的数学最优解/全方法误差下界。它测固定μ复用K4的实际代价；两次native已完成，独立CPU复核进行中。
