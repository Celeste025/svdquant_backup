# 阶段020：QAD完整视频对照完成，评分上升主要由一个异常样例驱动

2026-10-03。E021完成预先固定的四prompt×四臂，共16段77帧视频及全部内容/运动评价。QAD第64步相对plain的MJ总分两例上升、两例下降；均值虽从0.663升至0.896，主要来自BF16本身就严重失真的campus样例。没有足够证据宣称质量普遍改善或等质量加速。继续建立更充分数据的QAD强基线，不把本轮已知训练方法包装为贡献。

四臂均使用同一rCM四步、480×832、16fps、相同VAE与BF16 attention；每prompt四臂加载实际相同的初始latent、逐步noise及文本embedding后独立递推。四prompt使用同一个seed，因此它们也共享实际noise序列；不是四个独立噪声重复。固定第64次更新，未按已知开发结果换成第32步。SVD仍为旧整套配方，不能单独归因有无LR。

| 提示词 | BF16 MJ | plain NVFP4 MJ | QAD64 MJ | SVD MJ | QAD−plain |
|---|---:|---:|---:|---:|---:|
| 斑马弯腰饮水（066） | 1.2617 | 1.2768 | 1.2244 | 1.3033 | −0.0524 |
| 无人机飞过雪林（091） | 0.9358 | 0.8569 | 1.0662 | 0.9426 | +0.2093 |
| campus（182） | −0.2075 | −0.3619 | 0.6548 | −0.3864 | +1.0167 |
| 斑马奔向同类（067） | 0.8922 | 0.8799 | 0.6400 | 0.8165 | −0.2398 |
| 全四例均值 | 0.7206 | 0.6629 | 0.8964 | 0.6690 | +0.2334 |

MJ alignment/fineness/coherence_consistency均值在BF16为0.7598/0.0061/0.5815、plain为0.6943/0.0016/0.5679、QAD为0.9619/0.0612/0.6182、SVD为0.7002/0.0035/0.5640。全部28个criteria和5个aspect原值保留；safety/bias未用于量化比较。该评分器实际只看帧0/9/19/28/38/47/57/66，不覆盖末尾全部帧；分数是学习式奖励，不是人工质量或概率。

| 部署臂 | AMT平滑度均值 | RAFT运动阈值通过数 | DINO主体一致性均值 |
|---|---:|---:|---:|
| BF16 | 0.97697 | 4/4 | 0.93874 |
| plain | 0.97499 | 2/4 | 0.95194 |
| QAD64 | 0.98766 | 1/4 | 0.95221 |
| SVD | 0.97730 | 3/4 | 0.93529 |

AMT用39偶数帧预测38个奇数帧，与实际奇数帧比较，覆盖完整77帧；RAFT按原实现8fps计算38个流场，DINO使用全部77帧。QAD的平滑度上升不能直接当运动质量提升：其运动阈值通过数下降，且模糊/静止也可能提高一致性。RAFT本例阈值为top5% flow幅度11.25、至少10/38项超过；若干结果贴近阈值，例如BF16饮水为10项、SVD奔跑为9项，不应把binary结果解释为动作绝对有无。campus的BF16畸变产生极大flow，进一步说明运动大并非质量高。逐项数组和每视频读出均保留，没有合成新总指标。

root查看了固定5帧×4臂的全部四张图：QAD饮水/无人机背景改变且较模糊；奔跑例左侧出现提示未要求的蓝球；campus四臂均有明显绿色条纹，QAD可辨建筑部分更多。这里只描述抽帧外观，未声称逐帧完整观看运动。训练四文本的当前映射为划艇、竖琴、滑板车、抚摸动物；原缓存缺文本/seed，虽然SHA未变且teacher输出精确复现，也不能据蓝球反推特定样本复制或证明过拟合原因。campus保留在全部统计中，不事后删除。

独立CPU核验确认64次DiT、14,400 FP4 GEMM、3,840 BF16 SDPA；shared/final tensor及46个记录文件匹配，16媒体完整软件解码均77帧/832×480/16fps。生成约296秒、运动评价约128秒、最终MJ评分约14秒，含各自加载/记录，不用于速度比较。MJ首两次因使用PTQ环境入口缺decord/av在模型推理前失败，记录保留；改用已有MJ环境成功，原30分钟截止时间不变。虽Python binary是符号链接，sys.prefix及依赖目录不同；实际torch2.11、transformers4.49、模型eager attention。缺失分词源码用已有缓存补入独立目录，词表/编码已核验，无下载或原模型改写。所有这些GPU任务已退出。

下一轮E022从原BF16重启主权重训练，现场采集32新训练prompt×2独立噪声×4步=256状态，以及4新开发prompt×2×4=32状态；固定较小学习率、四timestep梯度累积、128次更新，预先指定开发检查点选择再做独立视频验证。该轮同时改变数据和优化配方，不能单独归因，更不能先称新方法。已有真实成本参照为plain/QAD约1.66秒DiT；需要先把质量稳定性做实。

独立CPU汇总已完成：AMT/RAFT逐数组复算、DINO保存的sum/76、MJ全16×28criteria/5aspect与实际抽帧索引均通过。DINO未保存逐帧项，不声称重新进行了DINO推理。campus对MJ均值差贡献+0.2542，大于全四例平均差+0.2334。

来源：[独立汇总](../../results/research/E021/summary.json)、[生成核验](../../results/research/E021/generation_validation.json)、[MJ原始分数](../../results/research/E021/mjvideo_scores.json)、[时间指标](../../results/research/E021/temporal_scores.json)、[固定评价协议](../06_experiments/E021_evaluation_protocol.md)、[训练文本关联](../../results/research/E021/training_text_context.json)、[E022计划](../06_experiments/E022_expanded_qad_plan.md)。抽帧图：[饮水](/data1/models/svdquant-wjq/research/20261003/E021/contact_sheets/vbench_066.png)、[雪林](/data1/models/svdquant-wjq/research/20261003/E021/contact_sheets/vbench_091.png)、[campus](/data1/models/svdquant-wjq/research/20261003/E021/contact_sheets/vbench_182.png)、[奔跑](/data1/models/svdquant-wjq/research/20261003/E021/contact_sheets/vbench_067.png)。
