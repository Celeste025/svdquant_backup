# 既有拍手音画事件读出的入口核查

2026-10-03。只读源码、六个既有 PCM 的 CPU 元数据/完整性和三个定向 primary 来源；没有听音频、检测冲击、生成评分、选阈值、运行 GPU 或修改主状态。

**结论：文件与时间轴允许一次很小的既有媒体读出，但现在不能声称 BF16 音画同步，更不能声称量化破坏同步。** E004/E015 停止的是局部误差/跨模态传播解释，不是最终事件同步的阴性结果。E038 又固定了全部 native SVD 线性层，只改 attention：即使后续发现效应，首先也是这个 attention 干预在该 H3 配方中的结果，不能泛化成 W4A4 相对原始 BF16 的同步损伤。

## 已核实的时间轴与 PCM 合同

CPU 实际读取 `E038/decode_{bf16,global_mean,coarse16}/vbench0161_r{0,1}.audio.pt` 六个文件。全部字典仅含 `waveform`、`sample_rate`，前者为 CPU FP32 `[2,165600]`、finite，后者为 32000；六份文件 SHA 和 waveform SHA 均与原 [BF16](/home/wjq/workspace/svdquant-exp/results/research/E038/decode_bf16.json)、[global](/home/wjq/workspace/svdquant-exp/results/research/E038/decode_global_mean.json)、[coarse16](/home/wjq/workspace/svdquant-exp/results/research/E038/decode_coarse16.json) decode 报告一致。CUDA 未初始化。六个 waveform SHA 互异；不是从 MP4 再解 AAC 得到的缓存。

- PCM 时长 **165600/32000 = 5.175 s**；124 帧/24 fps 的呈现跨度 **5.1666667 s**，末帧起点 **123/24 = 5.125 s**。不要把末帧起点误作视频时长。
- [NoiseInitializer](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:241) 使用 `round(num_frames/24*40)`，本例为 207 个音频 latent 时刻；[AudioVAE](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_audio_vae.py:417) 的 hop 为 `2*4*4*5*5=800`，故输出 207×800 个样本。多出的 **8.333 ms** 是末端长度取整；源码没有把这 8.333 ms 分摊成时间伸缩，也没有居中裁掉/平移开头。
- [decode_audio](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_audio_vae.py:474) 做原 latent mean/std 反归一化和原 VAE 解码；[output_audio_format_check](/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:151) 仅去 batch 维、转 FP32/CPU。[E038 decode](/home/wjq/workspace/svdquant-exp/scripts/research/prepare_decode_h3_center_video.py:229) 未加音频移位/重采样/响度归一化。
- [mux writer](/home/wjq/workspace/DiffSynth-Studio/diffsynth/utils/data/audio_video.py:39) 对 AAC 副本执行 clip→int16 编码；独立 `.audio.pt` 保留编码前的浮点 PCM。因此后续事件时间应优先用这份 PCM，避免将 AAC 编码的帧/延迟结构当作模型相位。实际六个 MP4 的 video/audio `start_time` 都为 0，time_base 分别 1/12288 与 1/32000；stream duration 也分别对应上述两种时长。
- [packing 坐标](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:588) 以 40 Hz 音频 latent 为共同时间单位：video frame 跨度缩放为 5/3，audio 时刻直接递增，二者起始均加 `text_len`（[实际赋值](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:685)）。这证明设计上的共同时间轴，**不证明 VAE/生成内容的事件实际同步**；没有从源码推断或补偿未知解码器群延迟。

## 三个最近邻：直接覆盖到哪里

已读旧 [joint_av_propagation_collision.md](/home/wjq/workspace/svdquant-exp/research_state/01_literature/joint_av_propagation_collision.md)，本轮没有再展开 Jacobian、模态重加权或新 loss。

| Primary，本轮核对位置 | 对这个入口的约束 |
|---|---|
| [Efficient Audio-Visual Generation via Synchrony-Aware Cross-Modal Sparse Attention, arXiv:2608.15522v1](https://arxiv.org/html/2608.15522v1)，§1/§5/Appendix B | 已正面提出加速可能损害最终音画对应，并用跨模态显著性保护及 Sync-C/D、音频指标评估。因此“加速要关注同步”没有新意。所核方法/实验为稀疏与缓存保护，未给本地 SM120 NVFP4 attention 配方的逐拍相对相位因果分解；其最佳匹配 offset 的置信/距离也不等同本例逐事件绝对时间差。 |
| [NVFP4-DiT 作者仓库](https://github.com/theraihanrakibb/NVFP4-DiT)，2026-10-03 读取 README 的 Methodology、Results、Code→paper map | 已宣称 FP4 下的同步保持、SyncNet loss 和同步指标，故不能声称首次研究“NVFP4＋AV同步”。但当前 README 仍列非标准 E2M1 数值集合（含0.75、漏0）且声称 H100 有 NVFP4 tensor cores；这份材料不能作为已验证的原生 NVFP4 基线。本轮只核其可见作者声明，未验证训练/测量，也未发现该 README 提供逐事件相位读出。 |
| [Synchformer, arXiv:2401.16423v1](https://arxiv.org/html/2401.16423v1)，§3/§4.2/§5 | 稀疏非语音同步、时间 offset 和“是否可同步”本身已是成熟对象。其标准 offset 分类以0.2秒为粒度、允许相邻一类容忍；与本例约0.2–0.3秒一次拍手相近，不能直接用一个总分判定逐拍是否丢失或错配。可同步性须先于 offset 解释；不能默认所有生成音轨都具有可对应视觉事件。 |

有限检索没有证明精确问题无人研究。仅做既有六条媒体的人工相位核查，是补足未测维度，不是方法贡献，也不复活已停的跨模态传播主张。

## 最小、可反证的读出；尚未执行

**第一步只看两个 BF16-attention clip 的自身音画关系。** 保持 E041 的视觉事件区间和不可读标记不变，先独立听/查看原始 PCM，判断是否真有可辨的单人拍手冲击，而不是连续人群掌声、音乐、背景敲击或无法分离的混合声。音频事件先于音画叠加独立标注 onset 区间，保留两声道，不用调阈值逼近视觉节奏。然后按原零起点叠加，不按每个 clip 拟合一个最佳移位。视觉 `[f_lo,f_hi]` 最多先解释为呈现区间 `[f_lo/24,(f_hi+1)/24)`，并保留“接触不确定”；音频 onset 也有定位不确定性，报告差值区间，不捏造亚帧精度。

重复拍手近似周期信号，跨相关在整周期移位处可能有多个相似峰：**频率相同不等于事件相位同步**。必须保留事件次序及无法匹配/多重匹配，不能事后只取最近峰、删除冲击或平移一整拍。这两个 BF16 clip 若本身没有稳定的可对应关系，结论应是“基线内容/读出不具备判定量化破坏同步的条件”，不是量化机制阴性，也不是据此新造修复算法。

**仅 BF16 可对应时，再检查 FP4 clip 自身的可读窗口。** E041 replica0 前段以及 replica1 coarse16 的部分窗口可提供视觉区间；global_mean replica1 等不可读段继续 unknown。最关键的混杂是自由轨迹：不能拿 BF16 的闭合帧当作 FP4 视频的真实闭合帧。`audio_Q` 偏离 `video_B` 的时刻，可能只是 Q 的音画共同改变了节奏；只有 `audio_Q` 与 `video_Q` 的内部关系才是 AV 同步对象。反之，两者共同平移/变速仍保持对应，会反证“音画相位被破坏”，即使各模态都不再接近 BF16。

所以当前状态是 **可读取、时间轴清楚、BF16 内容同步尚 unknown**。下一步有决策价值的产物是这两条基线能否建立事件对应，而不是六条总分、学习同步 loss、量化保护层网格或新生成。若后续 FP4 的视觉事件本身不可辨，最多报告同步不可评估/可辨性受损，不能把未知填成失步。
