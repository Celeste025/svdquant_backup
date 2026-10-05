# C007 残余候选

VC-Attention覆盖V clustering/中心化/row-sum均值恢复，DIDB-ViT覆盖Haar QK频率保真。恒等式不是创新，潜在残余是量化前保留P相对差异、PV双侧同预算编码及可见细节因果收益。与等scale、随机pair和V-only强对照未做，novelty partially verified，方法untested。

无方法效果、论文充分性或部署收益结论；本轮无GPU。详报告078。


2026-10-04追加核查：DIDB-ViT §3.2还直接分析attention差分信息损失并增加shortcut/局部差分补偿，不能只按Haar近邻处理；C007与已有“差分保真”动机碰撞更强。窄残余仍需同bit/scale PV双侧量化证据。兼容性与来源详[报告079](../../reports/079_20261004_vc_didb_compatibility.md)。


D111/E080更新：窄残余首轮未通过同预算强对照与总误差门槛，当前表示停止；不把V中心化收益转认领为新机制。报告080，未进入视频验证。
