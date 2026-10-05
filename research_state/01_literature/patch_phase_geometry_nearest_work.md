# 输出 patch 相位：有限先前工作核查

2026-10-03；仅核查，三条针对性 search query 后读一手原文，无 GPU、张量读出、新 loss 或实验改动。**结论：普通子像素输出头已经足以产生“同物理 lag、不同起点 phase 的协方差不同”。因此该统计阳性本身不能建立量化特有机制；在本轮有限范围未核到同一输出头普通扰动控制下的视频量化特例，不代表新颖性已成立。** E048 完整结果出现前不触发此方向。

| 一手来源与精确位置 | 已覆盖 / 未覆盖的边界 |
|---|---|
| Aitken 等，[Checkerboard artifact free sub-pixel convolution](https://arxiv.org/pdf/1707.02937)，§1、Eq.3（PDF第4页），§2 | Eq.3 将各子像素相位写成独立子核组作用于同一低分辨率特征，再周期重排；不同子核独立初始化便能产生 checkerboard，ICNR 通过复制初始子核消除初始化伪影。**输出行/子像素相位结构导致伪影不是新发现。** 原文是超分辨率与初始化，不是训练好 Wan 内部量化误差的实测；不能宣称 ICNR 已解决本地问题。 |
| [FP4DiT v2 §3.3、Fig.6](https://arxiv.org/html/2503.15465v2#S3.SS3)；[Quanto/Diffusers 官方实践，INT4 小节](https://huggingface.co/blog/quanto-diffusers#how-about-int4) | 前者量化粒度对应输入 patch/token 的激活范围，采用在线 token-wise 量化；不是输出 patch 内四组行的几何控制。后者明确展示保留 `proj_out` 高精度的实践；本地 300 个 native 目标本来排除它。因此普通 patch-aware scale / 高精度输出头不能重报为解法。 |
| [Q-VDiT §3.3，Eq.10–15](https://arxiv.org/html/2505.22167#S3.SS3) | 在模型输出 token 上增加跨帧关系分布蒸馏，已覆盖“单点输出 MSE 不保视频关系”的一般问题；公式没有同物理 lag 的输出子相位条件，也没有经过 VAE 的相位风险度量。不能把其 frame information 表述当成已解码 RGB。 |
| [SSVAE v3 §3.3、§5.3](https://arxiv.org/html/2512.05394v3#S5.SS3) | 已将视频生成 latent 误差与 decoder 鲁棒性联系，并做固定 encoder、注入高斯噪声微调 decoder 的消融。属于 decoder/latent 几何相邻工作，不是 DiT PTQ/QAT 或 unpatchify 子相位控制；普通噪声鲁棒训练、Jacobian/decoder-feature 加权本身也不能作为残余创新。 |

**本地源码事实与代数推导分开。** 实际 [transformer_wan.py:723](/home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/diffusers/models/transformers/transformer_wan.py:723) 先做末端 norm/modulation，再转回 hidden dtype、经 `proj_out`，`:726–730` 将四组输出重排到 2×2 格点；[目标清单](/home/wjq/workspace/svdquant-exp/scripts/research/wan_mainweight_qad.py:28) 不量化该投影。对**固定同一 head**，令其四组行为 (W_p)，紧邻投影输入的扰动为 ε_n，则实数线性部分为 δ y_{n,p}=W_pε_n；实际 BF16 另含输出舍入差。普通独立白噪声的 patch 内协方差是 σ^2 W_pW_q^T，跨 token 为零。相同横向 lag=1 从偶相位出发落在同一 token，从奇相位出发跨 token，所以没有量化也会出现所提 phase 差异。这是从实际重排与线性代数得到的零假设，不是原文声称的 Wan 实验结论。若两路径实际 head 参数或末端调制不相同，还须另计其差，不能直接套“同 head”。

**当前欠缺的关键证据。** E045/E046 同状态实际输出足够做描述性 CPU phase 读出，但尚未证明残差结构超出上述普通 head 传递。输出域同谱 Fourier surrogate 仅能排除所选平稳解释，**不排除 subpixel head 机制**；同能量白噪声直接加在已 unpatchify latent 上也不是正确的 head 控制。最便宜的普通对照应把独立 token 噪声通过实际固定 (W_p)，再用同一 unpatchify、同一 lag/support 和按时间片能量归一比较。若它已复现主要 phase 模式，停止“量化特有 patch phase”解释；若未复现，也不能自动排除普通有色/各向异性 token 噪声。进一步识别需要同状态末端投影输入的真实扰动及其空间/通道统计，不能把64维输出伪逆成隐藏特征并称为实测。

因此，即使 phase 读出阳性，也不据此启动新 loss 或 decoder 干预。现阶段保留的只是一个有限 unknown：匹配校准后仍有实际质量问题时，量化残差是否存在**超出相同 head 普通扰动传递**的结构与实际损害。尚无这一观测，更无独立方法 claim；E048 结果之后由 root 决定是否值得读出。

检索范围留痕（不扩搜）：`diffusion transformer quantization "checkerboard" "patch"`；`"unpatchify" "artifacts" quantization`；`video diffusion quantization "output" "decoder" geometry patch`。非一手结果未作为证据。前置入口：[wan_patch_phase_entry_audit.md](../02_problems/wan_patch_phase_entry_audit.md)。
