# 016：更小的张量误差，没有在本轮带来更好的视频评价

2026-10-02，E017。四段新视频、80次完整DiT与新旧8视频同次评分全部完成，273文件独立CPU核验通过。**p36中，block-mean比global-mean更接近BF16的局部输出和最终latent，辅助视频评分却更低；global-mean在两个样例的全部回答均与原SVD相同。** 这是两个已见样例上的有限观察，不是总体质量排序或新方法贡献。

## 做了什么

固定E010的文字动画p30/seed49771和产品广告p36/seed59526，直接复用其文本embedding与两模态初始噪声。原20步/CFG1、双sigma shift12/3、576×1024、124帧24FPS与32kHz立体声均不变；20步并非模型默认50步。新两臂均保留200个SVD原生linear，只切换E016验证过的官方block/global FP4 attention，各自自由递推。

原pipeline、scheduler与VAE不重写。四条轨迹的160次模态更新在CPU用原scheduler逐byte重放，全部实际输入与连续状态链匹配；共16000次原生linear调用/packing检查和4000次FP4 attention调用。四个新视频采用与旧结果相同的VAE和编码设置。整轮658.6秒，含加载、诊断、落盘、解码和评价，不能当生成加速比分母；E016固定完整DiT的1.325×/1.367×收益仍有原范围限制。

## 同一次辅助评价

旧BF16/SVD四段与新四段放在同一次VisionReward模型进程，使用原全部29题、权重、取帧与完整prompt，共232条有效yes/no回答。strict与归一化分数相同；旧四段的raw token、完整答复与输入token逐题完全复现，排除了本轮历史复评漂移。

| 提示 | BF16 | SVD＋BF16 attention | ＋FP4 block-mean | ＋FP4 global-mean |
|---|---:|---:|---:|---:|
| p30 文字动画 | 0.010473 | 0.010473 | 0.010473 | 0.010473 |
| p36 产品广告 | 0.196032 | 0.180285 | 0.100116 | 0.180285 |

p30四臂29题全部相同。p36的global-mean也与原SVD逐题相同；block-mean额外四题由yes变no，涉及运动平滑、运动真实、画面稳定及细节精细（0-based题号20–23），分差−0.080169。分数单位不是百分比；29个相关问题不能当29个独立样本算显著性。

p36在E016相同teacher状态s14的video NMSE：block/global为0.016821/0.018314；此次完整生成的最终video latent NMSE为0.313178/0.321062（均相对BF16）。**数值更近没有对应本例更高辅助评分。** 但这不能证明一个普遍误差机制，也不能由此设计新loss。完整自由轨迹已不同，终点差异不是同输入量化噪声或感知距离。

## 实际媒体观察与范围

已查看每个提示四臂的同六个固定时刻（f0/25/49/74/98/123）。p30都保留黑底、逐行白色中文与胶片颗粒，字形、布局和出现时机不同；没有据此证明文字准确率。p36都完成液体/花瓣/木片到绿色产品瓶的转换；block-mean在约2.04秒的花瓣边界更模糊，与细节项下降方向一致。静态接触图不能独立证实运动平滑或真实度下降，仍以有限观察报告。

[p30四臂对照图](/data1/models/svdquant-wjq/research/20261002/E017/independent_summary/E017_p030_all4_contact.png) · [p36四臂对照图](/data1/models/svdquant-wjq/research/20261002/E017/independent_summary/E017_p036_all4_contact.png)。新视频全部保留：

| 提示 | FP4 block-mean | FP4 global-mean |
|---|---|---|
| p30 | [视频](/data1/models/svdquant-wjq/research/20261002/E017/decode_block_mean/p030_block_mean.mp4) | [视频](/data1/models/svdquant-wjq/research/20261002/E017/decode_global_mean/p030_global_mean.mp4) |
| p36 | [视频](/data1/models/svdquant-wjq/research/20261002/E017/decode_block_mean/p036_block_mean.mp4) | [视频](/data1/models/svdquant-wjq/research/20261002/E017/decode_global_mean/p036_global_mean.mp4) |

旧BF16/SVD媒体见[报告010](010_20261002_h3_heldout_results.md)。本轮完整核查了八段媒体的帧数、尺寸、音轨和PCM有限性，**没有完成音频听感评价**。VisionReward只观察六帧，不替代完整时序、文字或音画同步评价；两个提示也不覆盖人物动作/口型/长语音。

## 决策与下一步

global-mean在本轮两个样例提供更快的已有配置，且未改变SVD的辅助回答；保留为后续部署参照，不能称质量等价。block-mean没有在这里证明其额外成本能换来更好的生成结果。保留全部结果，停止扩大当前基线网格，不追加提示或按结果换seed。

下一阶段转向具体可证伪机制。独立核查已区分FP4概率量化的分区依赖与概率质量守恒；PoT参考max和Sage论文的tile缩放均有直接先前工作，不能认领为新算法。[精确碰撞与待测边界](../02_problems/fp4_attention_partition_composability.md)。另在核查query分组是否与真实视频布局产生系统性相位作用；尚无观测或新claim，不由p36的单例评分倒推该机制。

证据：[协议](../06_experiments/E017_h3_fp4_video_plan.md)、[运行记录](../../results/research/E017/launcher.json)、[原始232题回答](../../results/research/E017/E017_visionreward.json)、[独立轨迹/媒体/评分汇总](../../results/research/E017/independent_summary.json)。独立汇总SHA `d0366a78d32abc9a5a81e2463ed87b3122e5d6be0e2a407d6ae328d18e691d10`。所有GPU任务已退出；顶会研究目标仍未完成。
