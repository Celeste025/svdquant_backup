# 条件响应差分：有限碰撞与诊断门槛

2026-10-02；primary 文献及代数核查，未运行 GPU、未注册 claim。**KILL“提出新的响应差分损失”；研究方向暂 PARK。最多保留一次有停止条件的微型异常筛查，不能由局部差分推出视频控制能力。**

## 已有工作覆盖的合同

| 原始来源与核实位置 | 已做；不能外推的部分 |
|---|---|
| [DASH，初稿 2026-05-30，核查 v2](https://arxiv.org/html/2606.00798v2)，§3.4 Eq4–8、§4.1/4.2、Table A6 | 压缩 CFG 模型，直接研究低 guided-output loss 下 guidance gap 幅度衰减/塌缩；比较 gap matching、双分支独立 MSE、multi-guidance-weight、pruning。Eq4 的零损失集满足 `w e_cond+(1−w)e_null=0`；Eq5/6 分别监督两分支，另有真实噪声 anchor。报告差分 norm ratio、cosine、MSE。**差分损伤诊断、差分蒸馏和其锚定问题已有直接先例。** 实测为 class-conditional 图像 UNet；不是 CFG=1 视频中的两个真实语义近邻提示。 |
| [GAMP，2026-07-09](https://arxiv.org/html/2607.08241v1)，§III-B/C Eq2–5 | PTQ 的 Eq2 正是 `||(f_q(c)−f_q(null))−(f_F(c)−f_F(null))||²`；证明共同分支偏移对此不可见。Eq5 改用 guided-output MSE，按此分配 activation bits；实测为 CIFAR-10 UNet、整数混合精度。**“量化应保护条件差分”及 response-only loss 的缺陷已经正面覆盖。** 未验证原生 NVFP4、最小语义编辑或视频时序选择性。 |
| [Knowledge Transfer with Jacobian Matching，ICML 2018](https://proceedings.mlr.press/v80/srinivas18a.html)，§4；[2ndMatch，核查 v2](https://arxiv.org/html/2506.05398v2)，§4 Eq8/9 | 前者早已蒸馏输入响应导数，并联系局部输入扰动下的输出匹配；后者在 diffusion pruning 中令 `J=∇_x s(x,t)`，匹配随机方向上的 `||Jv||²`，作为局部敏感度 proxy。**后者的导数变量是 noisy latent x，不是条件 embedding c；不能说它已实验覆盖语义 Jacobian。** 但把有限 prompt 差分命名为 Sobolev/响应保持新方法，没有足够残余。 |
| [MixDQ，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02212.pdf)，§3.1/3.2；[MBQ，CVPR 2025](https://arxiv.org/html/2412.19509v2)，§3.1/3.2 Eq6–17 | MixDQ 分离 text alignment/content 与视觉质量，保护 BOS，并对内容相关层使用 SSIM、质量相关层使用 SQNR。MBQ 用模态敏感度/任务梯度加权重建。它们覆盖“整体数值误差小不保证条件信息保留”的上位动机，核实内容中未见同状态最小 prompt pair 的输出差分约束。 |
| [Q-Diffusion，ICCV 2023 supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Li_Q-Diffusion_Quantizing_Diffusion_ICCV_2023_supplemental.pdf)；[SAQ，CVPR 2026，§4.1 Eq12](https://arxiv.org/html/2505.02242v2)；[P4Q，§3 Eq14/15](https://arxiv.org/html/2409.17634v1) | Q-Diffusion 的 text calibration 已同时采集 conditional/unconditional features，不能说旧 PTQ 全忽略 CFG。SAQ 对齐量化与高阶 teacher 的采样方向，并非条件对差分。P4Q 的 contrastive supervision 和 cosine-prediction distillation 对齐低比特 CLIP 图文表征，并非生成 velocity 的条件导数。普通“contrastive quantization”命名不形成区分。 |

相邻概念也不能混用：[Contrastive Prompts §3.2](https://arxiv.org/html/2402.13490v1)已经使用只差少量 tokens 的两提示 score 差分进行解耦控制；[CONFORM](https://openaccess.thecvf.com/content/CVPR2024/papers/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.pdf)用对象/属性 attention 的正负关系改善组合绑定；[GenCtrl 官方摘要](https://machinelearning.apple.com/research/genctrl)研究带 PAC 界的可控集合。前两者不等于压缩保持该差分，后者也不能由一次 velocity 差分推得。既有 [GCBT §3](https://arxiv.org/html/2610.00930v1)还直接编码 CFG 两分支的相关结构；不能把共享尺度/分支旋转改称新语义方法。

## 代数边界：不是缺失约束，而是误差方向的权重

固定相同实际输入状态 x、t，令 `e±=f_Q(c±)−f_F(c±)`、`d_F=f_F(c+)−f_F(c−)`、`δd=d_Q−d_F=e+−e−`、`m=(e++e−)/2`。则

`L_pair=||e+||²+||e−||²=2||m||²+(1/2)||δd||²`，且 `||δd||²≤2 L_pair`。

因此：两个提示都做输出 MSE 已约束绝对差分误差，没有 DASH 单 composite 约束的精确 null space。弱 `||d_F||` 可使相对差分误差很大，即使按完整输出能量归一化的单提示 NMSE 小；这首先是分母及误差方向问题，**并不证明量化偏好损伤语义**。附加 pair-difference loss 是加重 difference mode、保留 mean mode 的二次度量；它含跨提示交叉项，不能等同于分别给两个 prompt 乘标量权重，但就是已知关系匹配/图 Laplacian 形式 `tr(EᵀLE)`。在可微、足够小的连续条件扰动下才近似 Jacobian matching；离散换词不等于无穷小导数，原生 quantizer 也不可直接用 STE Jacobian 当真实响应。

**对 GAMP 也要批判核查：** 其 gap-loss 共同偏移反例正确；但文中把独立双分支 MSE 也归为同一 null space 并不成立，上述恒等式已反驳。其 Eq5 只排除“差分已准确时仍任意共同漂移”的特定子空间；单个固定 w 的 guided loss 仍有 DASH 的另一 null space。不要借来源里的过强叙述制造我们的新贡献；输出空间允许某种漂移，也不证明有限 clipping 参数必然能实现任意漂移。

## 是否值得 18-forward 级筛查

**仅条件式值得；不是启动方法实验。** CFG=1 H3 没有显式 cond/null 合成，也没有对应的训练未识别机制，所以它不是 DASH 失败原因的原样复现；但同样使用两输出差分的数学并不新。剩余可问的是：在实际 native W4A4 合同下，是否出现对明确语义编辑的**选择性沿 teacher 方向衰减**，而 teacher 响应幅度相近的普通编辑保持正常；若存在，之后才找独立的表示瓶颈原因。

预算内预指定目标编辑与普通编辑对照，同一 x/t、同一 text encoder、合法真实 packing/mask、相同精度合同，禁止先看 Q 结果再选 prompt。普通编辑的 teacher 差分幅度必须已知或在预算内检查；未匹配就记不确定，不能无限搜索。至少报告 `a=<d_Q,d_F>/||d_F||²`、`r_perp=||d_Q−a d_F||/||d_F||`、norm ratio、绝对 `||δd||²`、`||m||²` 和两单提示误差。预设 teacher 差分能量下限，并相对 BF16 重复数值基线检查；`a≈1` 而正交噪声大不是响应塌缩。实际完全塌缩需 norm 也小。不要把 code 翻转数或大相对 NMSE 当因果证据。

**强朴素基线：** 两提示均衡覆盖的普通输出 MSE/DASH 式双分支监督，及同一数据上的显式 mean/difference 加权 MSE；任何后续方案必须超出这一基线。幅度匹配普通编辑对照先用于诊断“只是所有微弱差分都更易受噪声影响”，无需先训练新 loss。

**STOP：** 仅正交噪声、仅弱分母放大、所有小响应同等变差、换一个预指定编辑不复现、或 BF16 自身不可靠响应目标语义，均停止。一次 18-forward 筛查最多决定要不要验证，不证明稳健选择性。只有独立预指定编辑上复现额外 gain 损失、普通编辑保留，才值得在下一阶段预注册 decoded minimal-pair 语义核验；局部 teacher 差分本身不是真实运动/时序控制的 ground truth。**本轮没有这样的证据，也没有可成立的新方法贡献；保持 PARK，不强推 GPU。**
