# 视频 DiT 量化误差与 temporal VAE 解码：有限近邻核查

2026-10-03。按指定范围核 SSVAE、VidTwin，另核四篇直接量化工作；只读原文与本地源码，无GPU、方法实现或新claim。覆盖状态：**partially verified / 一般问题高度重叠，具体机制未观测**。未在此范围找到直接覆盖，不代表新颖。

**待检验问题只有一个：** 在固定视频decoder、固定latent误差能量时，实际DiT量化误差是否特别对齐恢复压缩时间内子帧差异的方向，造成不成比例的重影？E041目前支持“FP4拍手仍重复，但重影妨碍辨认周期”，不能推出动作变慢，更没有证明decoder是放大来源。这与E025的解码dtype对照不同。

## ① 已直接覆盖的一般内容

| 一手来源及精确位置 | 已有证据 | 对当前问题的边界 |
|---|---|---|
| [SSVAE / Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability，v3，§3.3、§5.3](https://arxiv.org/html/2512.05394v3) | 明确把生成latent误差与decoder鲁棒性联系：VAE后验方差很小，decoder训练分布不能覆盖生成误差；LMR增加鲁棒性，另有固定encoder、注入方差0.005高斯噪声微调decoder的消融。§4直接分析3D时空频谱。 | **“视频decoder对生成latent误差缺乏鲁棒性”及噪声鲁棒训练已覆盖**。但这里不量化DiT、不做等能量方向比较，也不把误差关联到某个时间压缩子帧相位。latent低频更利于diffusion训练不等于低频误差在解码后更安全。 |
| [VidTwin，§3.2–3.3、§4.4.1、图1/4](https://arxiv.org/html/2412.17726) | 架构分出structure/dynamics latents；分别解码和跨视频重组表明dynamics携带纹理、颜色与快速局部运动，structure携带较慢结构趋势。 | **“不同latent信息对快运动作用不同”已有明确干预**。这是专门训练的双分支表示，隔离分支也改变信息量；不是普通H3 latent通道已经解耦的证据，不是等能量量化误差或decoder子帧敏感性的证据。 |

因此，普通decoder Jacobian各向异性、pixel/perceptual/decoder-feature加权、把时间高频赋更大权重，均不能仅换到NVFP4便宣称创新；这里不提出任何loss方案。

## ② 只相邻的直接量化先例（四篇封顶）

| 来源 | 精确已有构造，与本题差别 |
|---|---|
| [Q-VDiT，§3.2–3.3，式8–15](https://arxiv.org/html/2505.22167) | token/frame尺度量化误差补偿，加模型输出token的跨帧关系分布蒸馏。故“逐帧MSE不能保持时序关系”不新；该构造没有经过temporal VAE子帧恢复算子。论文把token输出称作frame information，不能据此当作已解码RGB。 |
| [S²Q-VDiT，v4，§3.2–3.3](https://arxiv.org/html/2508.04016v4) | 用输入二阶量近似量化敏感性，并以attention分布重加权token蒸馏。敏感性对象是DiT校准/模型输出，不是VAE解码相位。 |
| [PulseQuant，v2，§4.2式4、附录C.1](https://arxiv.org/html/2609.33384v2) | 单block-step扰动后继续dense推进，风险明确由后续**latent**归一化误差比定义；另报告temporal high-pass误差。因此“等局部误差不同最终影响”及时间高频评价都已覆盖，但没有把终点放大归因到temporal decoder或上下文拼接。 |
| [TCEC / Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models，v4，§3.1–3.2、附录6](https://arxiv.org/html/2508.12094v4) | 量化误差通过solver与**denoiser Jacobian**传播并作近似补偿。这里的time是去噪步，不是解码后视频帧/子帧；不能混用为temporal VAE机制。 |

上述结论来自实际方法/公式，不以关键词搜索无命中充当穷尽性证明。所核六篇未给出“量化latent在不同tile/chunk上下文下解码不一致”的直接实验；本轮没有扩展拼接文献搜索。

## ③ 真正还需观测的内容与最小反证

**本地结构提供了可测接口，但不提供机制结论。** H3为36层ViT decoder，末端Linear输出每token的`4×16×16×3`像素块（[minimax_h3_video_vae.py:249](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:249)、[268](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:268)、[297](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:297)）。因此latent时间轴的FFT不能直接代表输出四子帧间的运动：相位信息还可通过通道和非线性decoder恢复。5个主latent加2个上下文、裁帧与5帧时间blend是另一条依赖（[345](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:345)、[489](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:489)）；空间tile也重叠blend（[385](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:385)、[431](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:431)）。**单上下文内部的子帧失真，与两个上下文预测不一致再混合，是两个假设。** audit已用非时间blend帧仍有重影的反例，排除“全部由最后时间blend造成”；这不排除空间上下文或其它解码作用。

**最小反证的识别性修正：原单R终态对照当前不执行。** 原设想比较 `D(z)`、`D(z+δ)`、`D(z+δR)`，其中`δ`为E038终态FP4与BF16-attention latent之差，`R`为预先固定通道正交变换。但这两个终态来自已经自由分叉的完整生成；r1构图、人物尺度已有明显差异，故`δ`同时包含有效内容变化，不能视作同一内容上的局部量化噪声。`Rδ`还可能离开生成流形。即使同一R用于所有位置、保留总能量、每token范数和时空Gram，也**不保语义、每通道分布或decoder基准敏感度**。因此单R放大对比不具量化根因识别性，阳性或阴性都不足以归因或否定终态的子帧量化机制；撤回上一版将其称为该机制“最小反证”的建议。

若以后继续，须使用**同状态单步真实扰动**，或明确将问题限定为**局部decoder几何**，不得从后者泛化终态量化机制。裁帧/blend前四子帧共同/对比分量与总RGB误差至多是读出，不会自动解决上述识别性问题。本次仅补限制，不另立GPU计划。

此处不重跑DiT、不拟训练loss，也不把一个decoder控制与两个上下文比较混成大网格。若以后专查上下文假设，必须分别比较同一重叠位置在两个合法上下文中的**量化前后增量**，不能只看到原decoder自身有缝便归因量化；本轮不追加该实验。
