# 阶段040：原版完整配对——SVD有效改善普通量化，细节缺口仍在

2026-10-03。E044全部16条新视频、全量评价及独立CPU汇总完成。**SVDQuant整套配方明显优于普通NVFP4，但没有恢复BF16质量：相对BF16，八例MJ细节分数全部下降。** 这建立了可靠原版参照下的实际量化质量缺口；它是基础研究证据，还不是新方法或可投稿贡献。

固定四动作×两seed，直接复用E043保存的实际FP32噪声及BF16正负文本embedding；原版Wan2.1-1.3B、81帧/480×832/16fps、50步UniPC/CFG6/shift8，BF16 attention、FP32采样状态和VAE不变。BF16八视频与评分继承E043，新生成plain与SVD各八条，不筛除基线失败例。plain由原版权重直接打包，无训练；SVD使用旧非rCM checkpoint，保留smooth/rank32与其他PTQ状态。其旧校准33帧、原shift未可靠恢复，这是整套配方及校准域比较，不能单独归因LR。

| 全八例均值 | BF16 | 普通NVFP4 | SVDQuant NVFP4 |
|---|---:|---:|---:|
| MJ total | .62983 | .20979 | .56357 |
| MJ alignment | .66248 | .20058 | .59595 |
| MJ fineness | .04080 | −.15472 | −.04128 |
| MJ coherence | .60107 | .42920 | .54492 |
| AMT平滑度 | .96686 | .96500 | .97034 |
| RAFT动态判定 | 7/8 | 5/8 | 6/8 |
| DINO一致性 | .92700 | .87393 | .90127 |

MJ为原始模型输出，不是概率。各文本先平均两个seed，再对四文本等权平均，无综合加权质量分。安全/偏差仅保留raw，不进入表格。

| 文本 | SVD−plain MJ total | SVD−BF16 MJ total |
|---|---:|---:|
| 拍手 | +.46187 | −.23026 |
| 叠衣 | +.58620 | +.07985 |
| 转弯 | +.30764 | +.02450 |
| 大象喷水 | +.05943 | −.13910 |

SVD−plain逐例总分7升1降，例外是喷水r1（−.01767）。SVD−BF16为2升6降，两个正例恰为原版本身语义/结构欠佳的叠衣r0与汽车r1；不能将两个均值为正的文本称为两seed稳定改善。相对BF16，SVD的fineness为8/8下降，DINO为7/8下降；相对plain，大象题fineness/coherence均值仍更低。全部逐例数字见[独立汇总](../../results/research/E044/evaluation_summary.json)。

root查看了全部16条固定九帧图，加上E043已有八条基线观察；不是实时播放或盲评。拍手、叠衣的手部和衣物有显著不规则色块/碎片纹理；大象/水花和车辆也有粗糙纹理与形变。SVD有时保留更接近原版的布局，但并未消除缺陷。汽车r1两低位臂的转向线索反而比原版更清楚，仍伴随车身形变。**较高AMT不能据此解释成更正确运动或质量恢复**；详见[全部观察](../06_experiments/E044_visual_observations.md)。这些图来自编码前像素，观察不依赖MP4压缩。

正式v2生成1273.24秒，评价72.62秒，六生成/三评价进程全部rc0退出。新增实际1600DiT、480000 native主GEMM、96000 BF16 SDPA、800scheduler、16公开VAE解码、0TE/0训练；每次DiT现场核300/60调用与300 fastpack检查。CPU汇总覆盖24媒体、16组同实际输入、16新FP32终态和800条标量步序；中间tensor未保存，不声称轨迹数值重放。时间包含诊断、构建与写盘，不作速度或等质量加速主张。

首轮在第一步的SDPA计数包装器关键字签名处失败：两个plain worker合计仅尝试2次DiT/6次native主GEMM，0完成采样步/视频；监督器停止其余worker。失败源与记录保留[attempt01](../../results/research/E044/attempt01/archive_mapping.json)，v2只修`query/key/value`兼容，协议、样本、数值配方未变。当前checkpoint五文件SHA亦已固定，不将其误称为原始训练资产的完整祖先证明。

下一步：保留SVD作为更强的实际质量基线，停止只靠普通NVFP4弱基线证明收益。先形成少量同teacher状态的干预，区分量化输出本身的时间位置差异与固定decoder对扰动的响应；不能拿已经自由分叉的终态相减当局部噪声，也不能从九帧宣称四帧周期。一般decoder抗扰动和causal帧不均衡已有[直接近邻](../01_literature/E044_decoder_cue_nearest_work.md)，所以不直接启动噪声微调、时间loss或参数网格。若没有超出近邻的具体机制，就停止这条解释。**当前无已成立新方法、等质量优化或可投稿核心贡献，总体目标继续active。**

协议：[E044固定计划](../06_experiments/E044_vanilla_wan_native_plan.md)；执行：[生成](../../results/research/E044/launcher.json)、[评价](../../results/research/E044/evaluation_launcher.json)；上一阶段：[原版BF16参照](039_20261003_vanilla_wan_reference.md)。
