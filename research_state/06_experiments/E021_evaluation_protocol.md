# E021 固定视频评价协议

2026-10-03。在全部16视频生成后、未查看媒体与评分前固定。全部四prompt×四臂均评价，不筛选成功案例。固定64-update模型；不按已知开发NMSE换32-step模型。四prompt仅一个seed，为开发诊断，不作为最终独立基准。

1. MJ-VIDEO-2B：复用仓库评价器及本地模型，8个均匀segment、max_num=1、BF16、deterministic forward；记录全部28criteria和5aspect，主要报告total/alignment/fineness/coherence_consistency。safety/bias保留原始结果但不用于量化质量比较。8帧奖励不代表全77帧的运动检查。
2. 本地原VBench AMT motion_smoothness：偶数39帧预测38个奇数帧，与实际奇数帧比较；实际覆盖完整77帧，不使用静态场景才适合的相邻帧MAE/flicker。记录逐次插值误差。较静止/模糊画面可得高分，不能独立解释质量。
3. 原VBench RAFT dynamic_degree：原规则取8fps，即0,2,...76共39帧、38流场，20 iterations；保存top5% flow幅度及阈值判断。binary运动有无不是越大越好质量。
4. 原DINO subject_consistency：全77帧，平均76个当前帧相邻/首帧cosine项。显式本地权重、不下载；旧函数返回单video为sum，按76归一化与其sim_per_frame相同。固定每两视频调用避开其未用的单例除零；不改变实际分数定义。主体稳定性不能证明语义正确。

逐prompt、逐arm并列，不人为合成总指标，不对四例作显著性/泛化推断。静态contact sheet固定帧0/19/38/57/76用于外观描述，不据此宣称完整运动或听过音频。

启动前校验生成已complete、16媒体hash和帧结构。MJ独立GPU5、时间指标GPU0，启动前现查空闲，各30分钟硬超时；长任务tmux+log，部分评分保留。不占用他人卡。第一轮CPU加载发现MJ模型目录缺自定义分词源码，缓存tokenizer异常正作离线修复，尚未实际MJ GPU调用。时间指标CPU check已通过，使用已有AMT/RAFT/DINO权重，不重装环境。
