# E044：原版 Wan 同输入 native W4A4 完整配对

2026-10-03，exploration / baseline。上轮 E043 完成可读但并不全成功的原版参照，属于 progress。当前无合格论文 claim；本实验建立量化研究的实际质量缺口，不将旧 SVDQuant 或基础设施当新方法。

问题：在没有旧 rCM 共同条纹的原版非蒸馏 Wan 中，普通 NVFP4 与现成 rank-32 SVDQuant 是否仍有明显、值得定位的完整生成损伤？此实验回答整个已部署配方的差别，不能单独归因低秩分支或某个量化组件。

最小有效设置：E043 全部四动作×两 seed，直接加载当时保存的 FP32 initial noise、BF16 正负 embedding；保留原版 Diffusers0.40 pipeline、81帧/480×832/16fps/50步 UniPC/CFG6/shift8、FP32 state与VAE、BF16 attention。BF16 视频和分数继承 E043，不重生成。新生成 plain_nvfp4 和 svdquant_nvfp4 各8条；全部保留，不按基线动作成功与否挑样本。

plain 从已确认的原版 BF16 权重按现成 Wan legacy E2M1/group16 配方打包，300主Linear，无smooth/LR，不训练，不复用rCM导出。SVD载入已有非rCM checkpoint 的saved residual、静态W scale、smooth与rank32 branch，再用既有native转换与fast activation packer。SVD旧校准为33帧/CFG6/50步、原shift未可靠恢复；当前81帧/shift8是明确外推。旧checkpoint来源有原日志/config及小片权重支持，本轮五文件SHA只绑定当前资产，不补称原始训练祖先已全部验证。

每次DiT实际检查300 native主矩阵、60原BF16 SDPA及fastpack有限域；CFG正负两次分别调用，不合并其动态global统计域。保存每条50步标量、终态latent、原始媒体与固定9帧图；native和BF16输出不要求逐位相同，不设置任意NMSE通过阈值。300 saved-weight roundtrip仅验证打包无新增误改，不作为科学质量gate。首条完整生成自身验证执行，不另生成少步smoke视频或运行昂贵旧QDQ整模参照。

读出：新16视频全量MJ四主项/原AMT/RAFT/DINO，BF16继承原评分；按原case配对差分、按prompt报告两个seed方向。奖励分数不是概率，RAFT不是动作正确率，原版失败样例不隐去。自由生成终态NMSE如记录仅是轨迹差，不当同语义局部噪声；不预设动态收缩/同步失步解释。时延包含加载、诊断与写盘，不作等质量部署速度主张。

决策：若两低位配方在完整媒体上基本可用且没有稳定质量需求，不从小NMSE差发明loss；转向有具体瓶颈和强基线的系统问题。若稳定损伤集中在可读内容，下一只定位能区分机制的少数完整状态/组件控制，并先核强近邻；不直接套普通QAD。如果SVD更差或收益混合，保留结果并考虑旧校准/配方整体差异，不把“LR无效”或“原版不适合量化”当结论。技术加载/非有限失败记录为未完成，不计作机制阴性。

资源：实际空闲GPU0/1/5；六worker依次分组，0跑plain[161,192]再SVD[269]，1跑SVD[161,192]再plain[269]，5跑plain[316]再SVD[316]。每个worker仅启动一次，单worker1800秒，总3600秒；每轮启动前重查该卡无其他占用。16视频=1600DiT/480000 native主GEMM/96000SDPA/800scheduler/16decode/0TE/0训练。所有长任务命名tmux、独立进程组与期限；失败保留，修复新版本，旧源和结果不覆盖。大文件/cache均DATA1。
