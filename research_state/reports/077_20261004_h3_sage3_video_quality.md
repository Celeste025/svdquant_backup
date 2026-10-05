# H3 SageAttention3＋SVDQuant 基线画质检查

2026-10-04，E079，完成。**有可见的局部下降，以拍手的手指轮廓模糊、重复边缘最明确；当前8例不支持普遍整体崩坏。** 按用户要求，从此将官方SageAttention3＋现有SVDQuant作为工作基线，BF16和仅SVD保留作参照；历史实验标签不改。

## 动机与实验

E078发现叠加Sage3会增加完整模型相对BF16的输出误差，但没有证明视频误差超加和放大，也没有给出画质判断。本次检验这种变化是否对应可定位的可见缺陷。

固定拍手、叠衣、汽车转弯、大象喷水四种动作，各2 seed，共8例。复用E038的精确初始噪声与提示embedding，匹配E073 BF16和E038仅SVD参照。H3：20步、CFG1、video/audio shift12/3、1024×576、124帧、24fps，原BF16 VAE。现有SVD：200个native NVFP4 W4A4 linear、group16、BF16 rank32。官方Sage3使用E078相同pin与实现，50个主block有效长序列切换到Sage3，refiner/padding保留SDPA，per_block_mean=True，调用前clone K避免原地中心化污染。

新增8视频、160次完整DiT、8000次官方Sage3 attention；8视频及8音频VAE调用。GPU0/1各4例，生成加解码launcher墙钟约496/491秒，两者同时运行；全部成功。音质没有评价。执行源与协议冻结，未训练或重新校准，未使用MJ打分。

## 结果与结论

| 动作 | 相对仅SVD的新增缺陷 |
|---|---|
| 拍手（2 seed） | 两例均看到更持续的手指融合/重复轮廓，r0最清楚；动作仍可辨认。 |
| 叠衣（2 seed） | 两臂已有模糊，部分组合帧更差；不能确认两seed一致的明显额外下降。 |
| 汽车（2 seed） | 未见明确额外崩坏，车体仍稳定；构图/车型变化不当作缺陷。 |
| 大象（2 seed） | 均保留喷水；鼻子姿态和水雾时间有变化，未见明确额外崩坏。 |

这是模型有标签顺序抽帧观察：实际查看32页、每例每臂32帧，非完整实时播放或独立人评。因此暂不判断速度、接触次数或细粒度时间一致性。详见results/research/E079/visual_observations.md。

8例等权平均，所有指标使用原尺寸全部124帧、RGB[0,1]；LPIPS-Alex FP32，L1为MAE，L2为RMSE（不是未归一范数）：

| 配对 | LPIPS | L1 MAE | L2 RMSE |
|---|---:|---:|---:|
| 仅SVD vs BF16（E074同8例） | 0.44336 | 0.11692 | 0.17510 |
| SVD＋Sage3 vs BF16 | 0.44881 | 0.11820 | 0.17986 |
| SVD＋Sage3 vs 仅SVD | 0.31074 | 0.08171 | 0.13366 |

整体距离变化不大，不能转换为质量下降百分比。拍手r1对BF16的LPIPS甚至从约0.3995降到0.3683，但手指仍更模糊：构图变化和局部缺陷会使距离与质量判断不一致。16个配对的逐帧整数绝对差/平方差归约及MAE/RMSE已用stdlib独立复算通过。

## 视频与证据位置

- [全部8例三路播放页](/data1/models/svdquant-wjq/research/20261004/E079/review/index.html)
- [最清楚的拍手r0三路视频](/data1/models/svdquant-wjq/research/20261004/E079/review/vbench0161_r0_triplet.mp4)：从左到右BF16、仅SVD、SVD＋Sage3；并排版无声。
- 原始含音频视频：/data1/models/svdquant-wjq/research/20261004/E079/decode_svd_sage3_r0/ 和 decode_svd_sage3_r1/。
- 配对及输入哈希：results/research/E079/triplets.json；指标：metrics.json；资源：resource_summary.json；抽帧与媒体哈希：visual_manifest.json。

## 后续判断

基线采用已落实。拍手提供了可定位的研究对象，但“快速细结构是否因叠加量化而特别受损”目前只是候选假设，四类动作不足以证明。若继续，应围绕该缺陷做固定输入的局部误差与传播干预，并补仅Sage3对照，区分独立attention损伤与SVD交互；不启动层/步排列组合搜索，也不把组合本身当创新。当前任务到此完成，不再追加GPU实验。
