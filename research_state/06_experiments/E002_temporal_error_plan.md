# E002 — P001/C001 的探索性跨步误差诊断

假设：真实视频DiT输出的NVFP4误差有移除channel bias后仍保留的跨步方向相关；固定权重可能是主要来源。
设置：原Wan2.1-1.3B，已有33frames/50step/CFG6 BF16缓存；prompts0019、0022，steps22–27连续6步；完整latent、两CFG分支。BF16缓存回放NMSE必须≤1e-8。
对照：同teacher输入、固定checkpoint，W4A4与仅禁用activation quantizer的W4A16；保持smoothing/低秩。W4A4−W4A16记activation increment，不能标W16A4。
统计：cosine Gram矩阵、去除每通道空间时间均值后的矩阵、相邻cosine；等权误差和能量只作描述，不能叫UniPC终点误差。
决策/停止：12个step-case完成即停止；若两个prompt去bias相邻cosine都≤0.2，停止互补舍入构想。若稳定>0.5，优先查输出比例偏差与闭环传播，再考虑方法。中间区间不足以决定，选择一个有区分力的对照而非扩规模。
局限：校准prompt、teacher-forcing、当前QDQ模拟、仅中期窗口；不能作论文或质量改善结论。样本单位为prompt，不把latent维数当独立样本。
输出：results/research/E002_wan_temporal_errors.json；error tensors写数据盘research/20261002/E002；日志results/logs。
