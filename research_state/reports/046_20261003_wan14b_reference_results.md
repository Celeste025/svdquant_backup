# 阶段报告046：原版Wan14B八例参照完成

2026-10-03 22:40（上海）。**14B完整BF16参照已建立，可以继续同模型量化对照；尚无新的量化方法或可投稿贡献。** 原自动链的生成、MJ/AMT/RAFT/DINO评价和CPU独立汇总全部complete，所有八例保留，未挑seed。root先看八张固定九帧预览并记录，再读取新分数；不是盲评或完整动作标注。

采用官方固定revision38ec498c的HF Diffusers参数CFG5/shift3/81帧480×832/50步UniPC，复用E043真实初噪和文本embedding。与1.3B的CFG6/shift8同时改变模型和采样设置，下表仅作描述参照，不能作为模型规模的单因素因果结果，也不是量化收益。

| 指标 | 1.3B BF16，E043 | 14B BF16，E049 | 逐例升/降/同 |
|---|---:|---:|---:|
| MJ total | 0.629829 | 0.851667 | 6/2/0 |
| MJ alignment | 0.662476 | 0.912109 | 6/2/0 |
| MJ fineness | 0.040802 | 0.072021 | 4/4/0 |
| MJ coherence | 0.601074 | 0.650879 | 6/2/0 |
| AMT smoothness | 0.966857 | 0.976930 | 6/2/0 |
| DINO consistency | 0.926998 | 0.914935 | 3/5/0 |
| RAFT dynamic | 7/8 | 8/8 | — |

14B四提示MJ总分（每提示两seed均值）为拍手0.984170、叠衣0.730179、汽车0.745008、大象0.947311。DINO均值反而更低，说明一致性不能替代内容/动作质量；MJ原始分数不是概率，RAFT动态不是动作完成。当前八例反复用于诊断，不是独立泛化测试。

**视觉结论：** 没有此前突出的共同条纹或大块不规则手/衣碎片；拍手双例手部较连贯，汽车r1有明确弯道序列，大象r1有抬鼻与水花。但叠衣两例未清楚展示完整折叠，汽车r0转弯不明确，喷水的水流来源及接触/手指细节仍有局限。全部记录见[视觉观察](../06_experiments/E049_visual_observations.md)，不把teacher当物理真值，也不以这些局限无限推迟量化对照。

**执行成本：** 四卡并行生成1897.129秒（31.62分钟），评价70.902秒；800次DiT、64000次BF16 FlashSDPA、400次scheduler、8次FP32 VAE decode、0次TE/原生量化GEMM。独立CPU汇总1.373秒/CUDAfalse，核实际输入、端点、400步记录、800调用及媒体/原始分数，不重放中间整轨迹或重hash大权重。加载后allocated约27.242GiB，每worker双视频1834.5–1887.6秒；这些含诊断/解码开销，非优化后的serving benchmark。链及四生成worker已核退出，评分两worker均rc0。

**下一步（D074）：** 保留这个完整参考，E050采集本模型自己的64条匹配校准记录，复用同身份文本embedding以省掉TE。随后保持完整校准样本和搜索预算，测首个真实block的smoothing/LR资源成本，再决定全模PTQ。Plain原生逐层pack已完成CPU结构入口检查，尚未GPU验证；它不替代SVDQuant强基线。14B原始权重与全FP32 master共存最低78.96GiB，不能照搬旧整模安装。

[完整汇总](../../results/research/E049/evaluation_summary.json)，SHA48e880410ce36e3c6a4be75fbb0f5ba43c9f15136967678dd7262cf09d6f3dff；[执行链](../../results/research/E049/chain_launcher.json)。研究目标继续active，当前没有等质量部署优化或成立的新claim。
