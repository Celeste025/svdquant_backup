# E073：八例配对视频中的主干 W4A4 行为发现

2026-10-04；执行前冻结协议；初稿时0代码/GPU，当前尚未启动生成。**复用 E038 的 native SVD＋BF16 attention 八视频，再生成同一底模的原始 BF16 八视频，可以检验主干量化替换的整体行为后果。** E038 的 `bf16` 标签仅指 attention，不能把它当原始 BF16 模型。本轮是一次固定的行为发现窗口，不是新方法、显著性试验或扩大 benchmark。

**固定八例。** 完整沿用 [E038 manifest](E038_center_video_manifest.json) 的原文与哈希；索引是公开 VBench 的零基索引，不是旧 H3 prompt ID。全部保留，不依输出删例或补好 seed。

| 原 prompt | case / seed（r0，r1） | 预先固定的可读任务裁决 |
|---|---|---|
| A person is clapping | vbench0161；38001610，38001611 | 是否能读出拍手式手臂/手部动作；手部形状是否持续可辨，还是明显重影、变形妨碍识别。不数拍手次数，不裁决实际物理接触。 |
| A person is folding clothes | vbench0192；38001920，38001921 | 衣物是否保持可识别，能否看出整理、折叠的形变过程；记录衣物/手部突变或混融。不数折数，不推断接触、穿透或力学。 |
| a car turning a corner | vbench0269；38002690，38002691 | 结合道路/地标与车身朝向，能否辨认车辆自身沿弯曲路径转向。镜头平移、绕车拍摄或背景整体旋转不能单独算汽车转弯；缺少相对参照则 unknown。另记车身明显非刚性变形。 |
| an elephant spraying itself with water using its trunk to cool down | vbench0316；38003160，38003161 | 象与象鼻是否连贯可辨；是否能看出从象鼻区域向自身方向喷洒的水流。不要求证明水接触皮肤、降温或流体物理；来源/方向被遮挡则 unknown。 |

**唯一实验差异与复用合同。** Q 臂直接使用 [E038 denoise](../../results/research/E038/denoise_bf16.json) 与 [decode](../../results/research/E038/decode_bf16.json) 绑定的八个媒体：200 个 native linear、rank32 BF16 支路、50 主 attention 均 BF16；记录为 160/160 完成 DiT，0 FP4 attention 调用，八视频均完整。B 臂恢复同一个约20.1B pruned H3 原始权重的对应 200 个 BF16 linear，不使用反量化后的 `R+L` 冒充原权重，不换成另一份完整 H3。其他权重、非目标精度、两 refiner、QK norm/RoPE、attention 后端及有效 token/padding 边界保持 E038 语义。

直接读取 E038 `prepare/{case_id}.pt` 的 embedding、text tags、video/audio 初始噪声与 cu 元数据；逐例比对已记录的文件、embedding 和两个噪声 tensor 哈希，**同 seed 不足以替代缓存一致**。沿用实际 `schedule_cpu` 的 video/audio timestep、sigma 数组、20 次自由递推及相同更新算术；CFG1、flow shift12/3、CPU BF16 noise、576×1024、124帧。记录原模型/运行源码与关键环境身份；已有报告不代表这些资产本轮已重新核验。

新 latent 用与 E038 相同的 video/audio VAE 权重、精度及 decode 路径：tiled、tile256/overlap64、24fps、32kHz 双声道和相同视频编码。加载/生成/解码分开记录；不在本轮顺便切 FP32 scheduler、更换 VAE 或修采样配方。若必要资产或合同不匹配，该例记 `contract_invalid` 并保留，不能声称隔离量化，更不能偷偷换 seed。新生成上限为原始 BF16 八例、160 DiT；不新增量化视频网格。

**观看与裁决。** 每对隐藏方法标签，先以原速、同分辨率完整观看；主裁决静音以免音频暗示动作，音轨仍保留在原媒体。只在需要说明缺陷时回看连续片段并记双方各自时间戳；不强行按像素/相位对齐。每臂独立记任务 `可辨 / 明确未体现 / unknown`，另记可见形态缺陷及其是否妨碍任务。最后配对记录 `B较可读 / Q较可读 / 无明确差别 / 两者均有问题 / unknown`，附一句可复核的视觉理由；不同构图本身不判退化，unknown 必须说明遮挡、尺寸或参照不足。若观看者是模型，明确称模型观察并记录覆盖范围，不能称人评；抽帧不能冒充完整视频观看。

**输出与停止。** 输出固定八行，保留失败和 unknown，按四题列两个 seed 是否同向及具体缺陷是否重复；不把帧当独立样本，不依赖 MJ、RAFT、动作次数或像素 SSE。BF16 自己未体现的任务不能据此说 Q 丢失该能力；Q 更差也只支持这个冻结 W4A4＋LR 替换方案的配对总效应，不分解 W/A/LR、video/audio 路径或宣称特定机制。如果有跨 seed 重复、可清楚描述的任务损伤，再据它提出一个后续机制实验；若无清晰分离、结果混合或不可读，则结束本发现窗口，保留阴性/不确定记录，不自动扩样或调配方。


## 固定执行预算与入口

每个replica四例，分别GPU0/1；每卡生成+解码共享1800秒墙钟上限，native环境执行BF16 DiT，recovered环境执行原VAE。每卡最多80 DiT、4 video VAE、4 audio VAE；allocated峰值上限60 GiB，新数据上限4 GiB。检查当前GPU空闲后才启动，超时或合同/非有限/内存失败即保留状态并停，不自动重跑或增预算。全体没有新增TE、PTQ、native视频、MJ或RAFT调用。

复用E038源，仅移除native安装、保留bf16 attention route，复制相同decoder到独立入口scripts/research/run_h3_bf16_action_reference.py。manifest记录原prepare/运行来源及原schedule，运行前CPU检查验证缓存与源。执行后只检查160/320调用与步数、输入/调度绑定和八媒体合同；不搭建新的通用审计框架，也不把复制入口算科研贡献。

观察限制：当前工具若不能实际实时播放，模型只按原帧率顺序展开联系图并如实报告覆盖范围，绝不将抽帧或顺序静帧称为原速观看。最终人评与模型发现须分开。
