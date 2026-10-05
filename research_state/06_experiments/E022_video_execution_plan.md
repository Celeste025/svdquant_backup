# E022 独立视频执行安排

2026-10-03。上一目标轮为progress：E020/E021训练—原生部署—完整视频评价闭环已完成，E022现场数据288次teacher已完成，主权重训练实际启动。本轮核实训练PID1890988与监督PID1890981仍live，继续原90分钟deadline1790968253.118431，不重启或改训练源。

沿用已经固定的video_test_manifest.json与视频评价补充协议，8prompt×2seed、四臂64视频。不等待训练结束才准备全部对照：

- baselines阶段：GPU0完成BF16、E022 plain step0000、冻结SVD三臂48视频，共192DiT、38,400实际FP4 GEMM、11,520BF16 SDPA。一次保存16份shared初态/逐步noise/embedding，此后三臂及未来selected臂加载这些实际tensor。初始packed artifact绑定已完成export0记录；不把仍在更新的train_run整体SHA当冻结结果。
- selected阶段：训练终态complete且原worker退出后，绑定实际candidate_selection.json及最终训练结果，只生成预定native开发选择的16视频，共64DiT、19,200FP4 GEMM、3,840SDPA。复用前一阶段shared文件，不重抽seed、不换checkpoint。若selected=0仍如实执行该臂，不换非零更新。

两阶段保持rCM4步、77帧480×832/16fps、BF16 FLASH与E021同VAE。每次启动现查GPU0空闲，baselines20分钟、selected10分钟、60GiB，tmux与独立日志；其他用户卡不干预。生成记录是诊断墙钟，不拿并行训练期间的生成时间作严谨speedup。

这16条测试轨迹不进入训练或checkpoint选择；提前看到baseline媒体也不允许改已固定数据/学习率/训练计划。独立评价沿用已定MJ+AMT/RAFT/DINO定义、逐prompt内平均两个seed再八prompt平均，保留所有样例。当前尚未生成新测试视频；generator新文件实现后由root审核启动。训练主线保持，普通QAD不是新贡献。
