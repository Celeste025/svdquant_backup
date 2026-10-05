# Wan 输出 patch 相位误差：有限入口审查

2026-10-03；只读源码与已有结果，外部检索限定两条 query。未读取张量计算新指标，未运行模型、解码或新脚本。

**判断：有一个不等于“高频更多”的最小 CPU 读出，但尚无量化特有机制，更没有 decoder 因果归因。** 可以检查误差的二阶统计是否依赖固定 2×2 格点相位；不能只画 FFT 高频能量或把 patch 内协方差大叫作新现象。当前优先完成 E046，不启动新的生成/解码。

实际源码合同：本机 Diffusers `transformer_wan.py:598` 的 Conv3d kernel/stride 都是 `(1,2,2)`，`:670–671` 按 T/H/W flatten 为 token。`:623` 的 `proj_out` 每 token 输出 `16×1×2×2=64` 个数；`:726–730` reshape 为 `[B,T,H/2,W/2,1,2,2,16]` 再置换。因此原 latent 的 `(h mod 2,w mod 2)` 对应固定输出行 `16*(2*p_h+p_w)+channel`，空间 phase 是明确的，不需猜视频纹理周期。

`proj_out` 自身不在这轮 30×10 个 NVFP4 主层中，见 [目标清单](/home/wjq/workspace/svdquant-exp/scripts/research/wan_mainweight_qad.py:28)。量化主层中的 token 特征误差经过同一个 BF16 输出投影，就可能天然产生 patch 内相关；这并不要求输出投影被量化。标准的白色 token hidden noise 经四组输出行后，协方差也呈 `W_p W_qᵀ` 结构。因而“测到 2×2 相关”不能单独证明 NVFP4 特殊失真、aliasing 或新校准目标。VAE 空间放大率8只给几何尺度，不保证 RGB 中存在固定16像素因果晶格；卷积/非线性/上下采样会混合邻域。

已有材料足够，不需要重新 capture。E045 的 native_outputs 与同一 teacher 输入 capsule 可取 `δ=output_SVD−output_BF16`，分 cond/uncond/实际 BF16 CFG 三种；E046 的 native capsule 与原版 BF16 outputs 可取同方向 δ，但这是 **native 自由轨迹状态**，与 E045 两例 teacher-state 分开报告。两者是整个部署配方相对原版的差，不能归因单独 FP4。已有末步 latent 差也可读；不能用两条自由生成终态直接充当同状态预测误差。[E045 报告](/home/wjq/workspace/svdquant-exp/research_state/reports/041_20261003_terminal_intervention.md)已显示后时间片的预测误差在 VAE 前就更大，首片传播解释已停，pre/post-clamp 主模式保留。

若后续决定读出，最小问题应固定为：**相同物理 lag 的误差协方差，是否依赖 patch phase？** 对每 channel/latent-time 的 δ 单独去空间总均值，保留并另报四相位均值；固定 lag `(0,1),(1,0),(1,1)`，相同 interior/support，分别统计四个起点相位的协方差及零 lag 能量，再看相位间差异。沿横/纵方向能比较同一距离的“patch 内”与“跨 patch”，不能拿距离不同的项比较。按样本/时间片报告，避免把场景边缘或后帧误差更大混入 pooled 指标。

最便宜的普通噪声对照是 **保持实际 δ 完整二维功率谱的平稳 surrogate**，而不是同方差白噪声；公共 Hermitian Fourier phase 随机化可保留跨 channel/time 的二阶平稳谱，消除固定格点的非平稳相干性。这里只是离线统计对照的定义，未实施/挑种子。纯 checkerboard 的 Nyquist 能量也会保留，所以它不会仅因“高频”而给出阳性。即使真实 δ 超出该对照，还必须面对上段输出头的普通 token-noise 解释，不能立刻把结构归因量化。相反，若相位协方差无稳定额外结构，或只得到宽泛高频能量，应停止这条 patch-lattice 解释，不追加 VAE 敏感性实验。

近邻边界：两次定向检索没有核实到直接研究“量化误差在 unpatchify 的 2×2 输出子相位上非平稳”的工作，但这不是不存在先例的证明。[FP4DiT §3.3](https://arxiv.org/html/2503.15465v2#S3.SS3) 已直接讨论 patch/token 依赖的激活范围并使用在线 token-wise 量化，因此一般“patch-aware scales”不能当新意。[Diffusers/Quanto 官方实践](https://huggingface.co/blog/quanto-diffusers#using-quantized-models-with-diffusers) 已明确保留输出投影高精度以减少低位损伤；本地300层范围本就避开它，不能把这一成熟措施重新包装为解法。

已有输出误差与 RGB 差的空间叠图最多建立相关，不能证明该结构比同谱普通误差更易造成碎片。任何后续因果 decode 都需另行明确控制内容对齐与相同误差谱/能量；本轮不提出新干预或 claim。


## 2026-10-03 后续近邻核查

优先任务已推进至 E047 匹配校准及 E048 完整读出；本页没有触发新张量实验。后续[一页原文核查](../01_literature/patch_phase_geometry_nearest_work.md)找到直接结构近邻 ICNR Eq.3：子像素输出的相位子核本身能将普通 token 噪声转成相位相关结构。因此上文的输出域同谱 surrogate **只排除所选平稳解释，不足以排除输出头几何**；若 E048 后仍决定测量，普通扰动对照须经过实际固定 head 和同一 unpatchify。白噪声对照失败仍不能排除有色/各向异性 token 扰动；不由统计阳性直接启动新 loss。
