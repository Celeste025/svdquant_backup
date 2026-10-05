# 跨 denoising 步量化误差：有限定向查新

核实日期：2026-10-02。范围仅包括固定偏差/跨步相关、随机与互补舍入、沿 solver 时间轴补偿，以及交替两套低位权重。本文不是完整系统综述，也不是 novelty 证明；未运行实验。证据等级：A = 已读原文相关方法/公式；B = 官方摘要；C = 官方论文被搜索引擎索引的摘要/片段，全文访问受限。会议状态只按官方可见记录报告。

## 结论先行

1. **“量化 bias 可能比局部 MSE 更重要，随机舍入能改善 diffusion”已经被明确提出。** De-biasing Diffusion 是必须补齐的危险近邻，其 partial stochastic-rounding of weights 的具体实现尚未完全核实。
2. **“沿采样时间轴补偿量化误差”已有直接方法。** TCEC 使用当前和上一 denoising 输出估计累计误差；QuAKE 用输出历史进行递归估计；Q-Drift/DNS 从采样分布或 scheduler 修正。不能把 error feedback、history correction 或 solver-aware 本身当作贡献。
3. **有限查新尚未确认一个完全对应的方法：在同一条生成轨迹内，以 solver 权重/传播敏感度为目标，交替两套原生 NVFP4 权重并设计其跨步负相关。** 这不是“没人做过”的结论；De-biasing Diffusion 的未核实部分足以阻止当前作 novelty 宣称。
4. 可保留的机制问题是：在真实低步数视频 W4A4 中，控制单步误差后，**传播后的跨步协方差**究竟解释多少质量退化；是否能在同等权重存储、带宽和计算预算下改变它。它是待证伪问题，不是现成创新点。

## 最接近证据与技术边界

| 工作、日期与直接源 | 已做内容 | 与候选关系及未核实边界 |
|---|---|---|
| **De-biasing Diffusion: Data-Free FP8 Quantization of Text-to-Image Models with Billions of Parameters**；官方 PDF 标记 ICLR 2025 在审，未核实最终接收状态。[官方 PDF](https://openreview.net/pdf?id=nExUJBF5tR)，[官方 PDF 固定路径](https://openreview.net/pdf/ef26b54e0d9905ba01ede3c62d26e99cc1749b7c.pdf)。证据 C。 | 摘要明确比较 bias 与量化 MSE，提出 stochastic rounding 及 partial stochastic-rounding of weights；作者报告图像与全精度参考的偏差随 diffusion 步数增加而下降。已索引方法片段讨论 SR 可增大线性层 MSE 同时降低偏差。 | **最危险碰撞。** FP8、图像与 NVFP4、视频不同，但这一差别本身不足以形成贡献。全文请求被浏览器验证/403 阻止；尚不能判断是否每步重新舍入、保存几份权重、是否互补配对、如何摊薄运行开销。下一次查新优先补齐这篇，而不是继续广撒网。 |
| **PTQD: Accurate Post-Training Quantization for Diffusion Models**；2023，NeurIPS 2023。[原文](https://arxiv.org/html/2305.10657v3)，[会议记录](https://papers.nips.cc/paper_files/paper/2023/hash/2aab8a76c7e761b66eccaca0927787de-Abstract-Conference.html)。证据 A。 | 将量化扰动分解为与网络输出相关的部分及残差，做 correlated noise correction、bias correction、variance schedule calibration。 | 已覆盖量化噪声分解与采样噪声补偿。这里的 correlated 主要指**同一时刻与 denoiser 输出相关**，不能自动等同于不同 denoising 步之间的 residual autocorrelation。 |
| **Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models / TCEC**；arXiv v1 2025-08-16。[方法原文](https://arxiv.org/html/2508.12094v1)。证据 A，§3、式 13–18、§4。 | 先推导逐步误差传播，离线拟合逐时刻逐通道 K，使误差近似 K 乘量化输出；在线用当前及前一次输出、scheduler 系数构造累计误差补偿，缓存前一输出。含 SVDQuant W4A4 与 OpenSORA 视频实验。 | 与“solver 时间轴 feedback/correction”直接碰撞。它反馈的是**估计的输出误差**，不是在线持有全精度真误差，也不是权重列量化反馈。文中 W4A4 为 group 64 对称量化；不能直接当作原生 NVFP4 证据。其误差界的条件与实际 few-step flow matching 是否成立需另验。 |
| **Q-Drift: Quantization-Aware Drift Correction for Diffusion Model Sampling**；2026-03-18。[方法原文](https://arxiv.org/html/2603.18095v1)。证据 A，§3.3–4.4。 | 把量化看作逐步隐式随机扰动；校准每步/通道条件方差，匹配一步噪声方差并调整 drift。支持 Euler、flow matching、DPM-Solver++；实验包含 SVDQuant。 | 已明确指出局部预测精度不保证采样边缘分布正确。校准的是每个 t 下的条件均值/方差结构；本轮所读方法未给出残差跨时刻的完整协方差模型。“逐步方差匹配”不等于已控制实际误差的跨步相关性；也不能仅凭未见相关公式推断作者假定了全部 temporal independence。 |
| **Quantization-Aware Kalman Estimation for Diffusion Sampling / QuAKE**；2026-09-18。[方法原文](https://arxiv.org/html/2609.21407v1)。证据 A，§3、附录 B.2。 | 把最近多个全精度 denoiser 输出作为隐状态，利用平滑轨迹先验与量化观测递归更新整段历史；供高阶多步 solver 使用。 | 已覆盖 history-aware joint correction。§3 与附录明确假定 process/observation noises 在时间上相互独立；被建模的历史相关首先是**真实输出轨迹相关**。若真实 NVFP4 残差存在重要的跨步相关，这构成可检验的假设张力；尚未证明会导致其方法失效或提供足够贡献。 |
| **Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models / DNS**；ICML 2026。[PMLR 官方记录](https://proceedings.mlr.press/v306/yang26aa.html)。证据 B。 | 官方摘要将量化误差解释为 timestep shift，通过可变形 noise scheduler 吸收误差；覆盖随机/确定性 sampler，并扩展至满足高斯条件路径的 flow matching。 | “改时间/步长吸收误差”已被占据；本轮未逐式阅读全文，不对其跨步相关或随机舍入实现作断言。 |
| **Sampling-Aware Quantization for Diffusion Models**；初稿 2025-05-04，CVPR 2026。[arXiv 官方页](https://arxiv.org/abs/2505.02242)，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zeng_Sampling-Aware_Quantization_for_Diffusion_Models_CVPR_2026_paper.pdf)。证据 B。 | 官方摘要指出量化误差扭曲方向估计、尤其影响高阶 sampler；采用 mixed-order trajectory alignment。 | 不能仅以“高阶 solver 与量化耦合”为新意；具体与互补权重的关系尚未逐式核实。 |
| **Antithetic Noise in Diffusion Models**；2025-06-06。[论文](https://arxiv.org/abs/2506.06185)，[作者代码](https://github.com/jjia131/Antithetic-Noise-in-Diffusion-Models-page)。证据 B。 | 配对初始噪声 z 与 −z，得到负相关的**不同生成样本**，用于多样性与统计估计方差缩减；分析近似 affine odd symmetry。 | 术语接近但作用轴不同：不是同一 trajectory 相邻时间步的量化扰动。不能用这篇直接证明“互补 rounding 无新意”，也不能忽略其 antithetic 基础思想。 |

附带已知碰撞：本项目通用地图中的 [PulseQuant](https://arxiv.org/html/2609.33384v1) 已用单个 block-step 量化 pulse 加 dense continuation 评估最终敏感度；[AccuQuant](https://arxiv.org/html/2510.20348v1) 已做多 denoising 步重建。因此“最终质量由传播误差而非单点 MSE 决定”也是拥挤的上位叙事。

## 与 GPTQ 权重列间 error diffusion 的区分

[GPTQ 原文](https://arxiv.org/abs/2210.17323) 的典型过程在**离线量化一个权重矩阵时**，对已量化列造成的输出重建误差，用近似二阶信息补偿尚未量化列。其序列轴是矩阵列/量化顺序；最终部署通常是一份固定低位权重。它不是 denoising 第 i 步向第 i+1 步传递动态 residual 的 sampler。

这种作用轴不同只排除了术语混淆，不能证明贡献成立。真正需要面对的是 TCEC、QuAKE 等在线采样补偿，以及 De-biasing Diffusion 的权重随机舍入。经典 error-feedback 还能跨多个领域出现，本次有限检索没有穷尽数值积分、sigma-delta 或控制领域的对应方法。

## 候选问题应如何缩窄

在同一输入 x_i 上定义局部误差 e_i = f_q(x_i,t_i) − f_fp(x_i,t_i)。在线性化范围内，终点偏差近似为各步传播后的误差和 Σ_i P_i B_i e_i，P_i 是后续状态传播，B_i 是 sampler 系数。这个表达是诊断框架，不是本文提出的新定理。它提示需要区分：

- 逐步均值/bias、逐步误差能量，与 i≠j 的交叉协方差项；不能把同一 prompt 上的误差 cosine 直接当作跨样本统计协方差。
- teacher-forced 同一参考 trajectory 上的量化误差，与自由运行后输入漂移引起的新增误差。两个序列都应记录。
- 固定权重 residual 与动态 activation/scale residual。固定 W 并不自动意味着 denoiser 输出误差在所有时间方向相同。

最低限度的受控比较是：固定一次随机舍入并重复使用；每步独立随机舍入；保持边际舍入分布的 antithetic 配对；原有 deterministic round-to-nearest。控制每步幅值/单点重建误差，观察传播后终点误差和视频质量是否仍分离。若只做“随机化后单点 MSE 降低”，无法支持跨步机制。

**互补两权重的两个容易忽略的问题：**

1. 在非均匀 NVFP4 网格上，向上与向下两个最近值的平均一般不等于原权重；u 与 1−u 的 antithetic stochastic rounding 可构造负相关，但不保证每一对误差精确相消。动态 scale、clipping 和后续非线性进一步破坏简单抵消。
2. 两份完整 4-bit 权重的 payload 已约等于 8-bit 权重，还增加 scales/索引；低秩分支可共享但必须计入。应与一份 W8A4、单份 W4A4 加校正及等内存方案比较。即使相邻 raw error 为相反方向，不等步长与传播 Jacobian 也可能使终点偏差变大。

## 本轮已停止处与下一次唯一查新优先项

检索覆盖了 diffusion + quantization 与 stochastic/antithetic/complementary rounding、bias、temporal correlation、error feedback 组合，并沿近期 sampler correction 原文追踪近邻。没有搜到完全匹配标题不构成不存在的证据。当前最关键缺口是 **取得 De-biasing Diffusion 完整方法，核对 partial weight SR 的跨步采样和存储实现**。在填补该缺口前，只把互补权重当作机制实验候选，不作为论文中心 claim；本轮不再扩展通用地图。
