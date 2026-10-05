# 026：已知query分块已提供明确的显存—时间取舍

2026-10-03。**E027在固定真实H3层上，query4096分块使QKV到输出的时间增加6.91%，allocated峰值下降26.09%，输出不变。结合独立BF16校正stripe成本，当前不值得为朴素在线校正重写整套调度。** 这是已有分块方法的实测强基线，不是本项目创新。

与E026同一p36/step14/block0，全部56heads、22539有效tokens。两臂使用相同官方BF16中心化、padding、原生FP4 pack与attention；分块不重算K均值或重复量化KV。每臂3次预热、10次测量，包含correction GEMM、必须的连续复制、launch和输出组装；输入已驻GPU，无CUDAgraph。完整流程每次额外包含一次QKV预处理与pack。

| 测量边界 | 完整物化校正 | 固定query4096分块 |
|---|---:|---:|
| 已pack后的correction＋attention | 26.246 ms | 28.347 ms |
| 已pack时allocated峰值 | 2.616 GiB | 1.698 GiB |
| QKV到output | 31.780 ms | 33.975 ms |
| QKV到output allocated峰值 | 3.518 GiB | 2.601 GiB |
| QKV到output reserved峰值 | 3.857 GiB | 3.020 GiB |

完整流程两臂常驻allocated均.910719GiB；运行新增峰值为2.60775/1.68986GiB。峰值变化包含correction与输出组装生命周期，不把整个.918GiB差值都归给correction矩阵。完整流程10次CUDA读数范围31.748–31.873 / 33.933–34.061ms，同步wall中位31.801/34.002ms。固定顺序、单形状的小实验，不外推整模型吞吐或更长视频。

四份最终输出均与E026原block输出一致，独立CPU从保存tensor复算NMSE/maxabs为0。这是实际观测，未把逐位相同设为研究gate。总计182次attention、27次预处理（含1次启动检查），零DiT、零新视频。

独立BF16 stripe使用真实原μ/Kc，每tile M16/N128/K128，一行有效μ与15行重复，覆盖56×177×177 tiles。实际PTX含BF16 Tensor Core MMA、FP32累加；中位14.815ms，范围14.753–14.832ms，数值校验误差约1e−13量级NMSE。该时间包含自身读取、归约和checksum写出，cache/占用与融合不同，不能与attention时间加减拼出“融合后速度”。其4.122GiB历史峰值还包含完整FP32 reference与预处理，不能当在线workspace。当前编译资源32KiB shared、122 registers、无spill；照搬到已有attention不是一个免费回调替换。

[独立成本汇总](/home/wjq/workspace/svdquant-exp/results/research/E027/cost_summary.json)、[分块原记录](/home/wjq/workspace/svdquant-exp/results/research/E027/chunk_run.json)、[stripe原记录](/home/wjq/workspace/svdquant-exp/results/research/E027/stripe_run.json)。CPU最初FP8有限值检查不受旧torch支持，转换为FP32做检查后通过；失败档案保留，失败阶段未用GPU。正式两任务均正常完成退出。

下一步E028仅复用已有三层数据，在CPU测“块中心相对global中心”的三种谱：普通、K-score、K4-error几何。若采用受限中心c_i=A_iB，必须同时用于Q−c_i和c_iKᵀ，不能只近似校正项；谱也漏掉Q4码/scale随中心变化的项。此次只判断表示能力，不训练、实现kernel或声称新意。当前仍无可投稿核心贡献。
