# 阶段报告 010：H3 配对视频与独立评价

2026-10-02。**两例旧PTQ校准集之外的H3提示，BF16/原生NVFP4共四段视频已完成；生成、轨迹和媒体校验通过。现有证据没有显示质量提升，也不足以证明质量等价。** 当前仍无成立的论文贡献。

## 本轮做了什么

预先固定原提示表p30/seed49771（中文文字动画）和p36/seed59526（香氛洗衣液产品微距），均未用于旧PTQ的8个提示。共享文本embedding与两模态初始噪声，之后独立递推。沿用旧实验20步、CFG1、video/audio flow shift 12/3；使用原H3 pipeline、model_fn、scheduler及VAE。**20步不是pipeline默认的50步协议。** 本地模型仍为剪枝版H3，主干200个linear原生NVFP4，低秩分支和attention保留BF16。

四段视频均为124帧、1024×576、24 FPS，含32 kHz立体声音轨。独立CPU审查核对了160个逐步文件、4个最终latent、共有输入和每步实际DiT输出的对应关系；原生两次生成共8000次FP4 GEMM及8000次packing检查通过，DiT内部无磁盘加载。全部输出finite。原生加载阶段仍先构造BF16模型，本轮峰值37.60 GiB，不能将E009的16.80 GiB前向峰值当作启动需求。

| 提示 | BF16视频 | 原生NVFP4视频 | 对照图 |
|---|---|---|---|
| p30 文字动画 | [播放](/data1/models/svdquant-wjq/research/20261002/E010/decode/p030_bf16.mp4) | [播放](/data1/models/svdquant-wjq/research/20261002/E010/decode/p030_native.mp4) | [六帧对照](/data1/models/svdquant-wjq/research/20261002/E010/independent_summary/E010_p030_paired_contact.png) |
| p36 产品微距 | [播放](/data1/models/svdquant-wjq/research/20261002/E010/decode/p036_bf16.mp4) | [播放](/data1/models/svdquant-wjq/research/20261002/E010/decode/p036_native.mp4) | [六帧对照](/data1/models/svdquant-wjq/research/20261002/E010/independent_summary/E010_p036_paired_contact.png) |

分阶段总耗时约15.5分钟，包含资产哈希、加载、逐步落盘和解码，**不作为生成加速比**。E009的1.22×仅针对固定输入、常驻完整DiT的计时。

## 看到了什么

人工检查对照图：p30均生成三行中文，字体、笔画、布局和出现时刻有所不同；p36均呈现原料微距后切换绿色产品瓶，花瓣、木片和瓶型不同。BF16的旋盖与native的泵头均属于原提示允许的选项，不能以不同瓶型直接判定量化错误。已查看帧中无明显全黑或完全崩溃；这不是逐帧运动、文字准确性或音频质量的完整审查。

辅助VisionReward使用本地官方29题、权重及取帧逻辑，保留每题完整输出和token。全部116条回答均为可识别的小写yes/no，严格官方分数与归一化分数一致；root独立重算分数和视频SHA通过。

| 提示 | BF16 VisionReward | NVFP4 VisionReward | 差异 |
|---|---:|---:|---|
| p30 | 0.010473 | 0.010473 | 29题回答逐题相同 |
| p36 | 0.196032 | 0.180285 | 两项“开头物体形状”判断由yes变no |

这是6帧辅助评价；分数单位不是百分比。p30的文字动画与物理世界问题并不完全匹配，不能跨提示比较绝对分数，也不能以两例推断总体损伤。**音频尚未评价。**

核查还发现旧用户评价脚本按大写`Yes`的token判定，而本轮官方推理实际输出小写`yes`；旧映射会误判本轮所有肯定回答。本轮使用独立冻结评价器，保留原用户文件。未审计历史原始回答，因此不据此宣称所有历史分数无效。

| 提示 | 最终video latent NMSE | 最终audio latent NMSE |
|---|---:|---:|
| p30 | 0.2261 | 0.9995 |
| p36 | 0.3253 | 0.2915 |

上述是同初始噪声的两条自由轨迹之间的分歧，分母为BF16能量；不是局部量化误差或质量分数。后续步骤的输入已经不同，不能将velocity差异误称为相同输入的量化噪声。[完整轨迹图](/data1/models/svdquant-wjq/research/20261002/E010/independent_summary/E010_summary.trajectories.png)。

## 结论与下一步

1. 原生部署已贯通真实完整生成；没有足够证据扩大到质量等价或方法改进主张。当前两提示主要覆盖文字/产品，不能代表人物运动、口型、长语音；两者现已成为看过的诊断集，未来方法评价另设未看过的提示。
2. 停止为已有直接先前工作的候选做GPU实验。新核查的continuity/Stein校准目标被gauge与FDM等直接覆盖；朴素版本不进入实验。E011稀疏router仍因未验证完整运行路径而not_ready，不计为机制阴性。
3. 量化与ODE求解器候选也已完成筛选：SAQ/QuAKE及低精度RK工作有直接覆盖，当前H3只用Euler，没有自适应controller；本轮不扩建新sampler。条件响应筛选同样完成：DASH/GAMP等已覆盖差分诊断/监督，新增损失不立项。只保留一个待验证异常：语义编辑是否比幅度匹配的普通改动更易发生响应衰减。必须先有可靠teacher与明确对照才能预注册微型诊断，当前没有现象或新颖性证据，不增加GPU任务。

本轮全部GPU任务已退出。证据：[生成计划](../06_experiments/E010_h3_heldout_paired_plan.md)、[四阶段执行记录](../../results/research/E010_launcher.json)、[独立轨迹与媒体审查](../../results/research/E010_summary.json)、[原始逐题评价](../../results/research/E010_visionreward.json)、[评价说明](../06_experiments/E010_visionreward_auxiliary_results.md)、[continuity先前工作碰撞](../01_literature/continuity_calibration_collision.md)。源文件与结果已冻结，后续变更另开版本。

本轮额外筛选记录：[求解器碰撞](../01_literature/solver_quantization_collision.md)、[求解器可行性](../00_state/velocity_regularity_feasibility.md)、[条件响应碰撞](../01_literature/conditional_response_collision.md)、[条件响应概念审查](../02_problems/conditional_response_screen.md)。
