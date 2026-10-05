# E045：末步 CFG 与精度保护的有限近邻核查

2026-10-03；只读本地证据并核实 **2 篇原始论文**，无 GPU、无新实验。结论：**CFG 放大分支量化误差、晚步细节敏感、晚步提高精度均已有直接先例；简单 BF16 末步是必要强控制，不是新方法。** 本次不是完整综述，也不据文献否定真实部署对照的价值。

本地已有 [phase2_candidates](../02_problems/phase2_candidates.md)、[temporal_error_prior_work](temporal_error_prior_work.md) 与 [legacy_audit](../00_state/legacy_audit.md) 的相关记录；旧审计已测到 CFG6 绝对 MSE 放大约 32.3 倍，不能将 E045 的同类现象称首次发现。

| 原始来源及定位 | 直接覆盖 | 不能替它声称的结论 |
|---|---|---|
| [DSAQuant，arXiv v1，2026-09-03](https://arxiv.org/html/2609.04031v1)，§3.2、§4.1–4.2，附录 C Eq21–29、D/Table8 | 视频分阶段量化发现中晚步细节受损；附录 C 固定同一 latent/prompt/timestep，以全精度输出为参照，分解两支误差及相关项。动态网格不同可形成 mismatch，CFG 再放大。评估 W4A4/W3A3；Table8 的 Wan2.1-1.3B 对照为**对称 W4A4**。训练端调整 KD/原始 target loss 随步权重；推理端末若干步只用 conditional、跳过 uncond（CFG-drop），**不是残差估计修复**。附录 D 对 BF16/QVGen/Base 使用同一 drop，低位两者收益更大。 | CFG-drop 改变指导，**不等于 CFG6 保持不变的 BF16 末步**。这些证据不是原生 NVFP4/SVDQuant、UniPC 末步四角与 causal VAE 时间响应的完整实验；也不能将作者的晚步趋势当作本地两个末步点已证明的跨步规律。 |
| [PTQD，NeurIPS 2023，arXiv v3](https://arxiv.org/html/2305.10657v3)，§4.3 Eq13–15、§5.1 | 直接分析低噪声末期量化预测 SNR 下降；按步选择更高位宽保护后期。具体实验是**固定 W4，共享同一权重，激活在 A4/A8 间选择**；避免多份权重的存储/加载。 | 是图像 LDM，不是本地 Wan 视频；不是全权重、全激活换回 BF16，也未在该节分解 cond/uncond 的 CFG 放大。不能引用 PTQD 为本地“只保护最后一步足够”背书。 |

Q-Diffusion 的本地既有记录是同时收集 conditional/unconditional calibration features。本轮 CVF 主文/补充 PDF 请求均返回 403，arXiv全文接口亦不可用，故**不新增“Q-Diffusion 已实现末几步 BF16”之类未经核实的归属**；上述两篇已足以回答本轮问题。

**E045 实测与推断边界。** 直接读取 [summary_v2.json](../../results/research/E045/summary_v2.json) 的 21 latent 位置等权 MSE：

| seed replica | conditional | unconditional | 实际 BF16 运算 CFG6 | CFG 后时间位置 1–20 均值 / 位置 0 |
|---|---:|---:|---:|---:|
| r0 | 0.0371743 | 0.0368317 | 1.162326 | 2.4165 |
| r1 | 0.0427836 | 0.0425865 | 1.438575 | 2.6028 |

后续时间位置误差较大在 **decoder 之前** 已存在；这是视频时间轴，不是 denoising 迭代轴。它支持先处理 denoiser/CFG 的普通解释，不能从 RGB 碎片反推 decoder 是主要放大源。理想实数运算下 `e_CFG=6e_c−5e_u`，能量包含 `−60⟨e_c,e_u⟩`；不应假定两支独立或用单支 MSE 乘固定常数代替实测。表中 CFG 保留原 BF16 合成舍入，不冒充该实数恒等式的逐位结果。

**下一决定仅需一个强控制。** 在同一条量化自由轨迹的真实末步前状态上，比较末步继续 native 与换回原 BF16 denoiser，两者都保持原 CFG6/UniPC/FP32 VAE，记实际切换时间和驻留成本。它回答能否廉价修复完整量化结果。E045 的前 49 步均为 BF16，因此当前 B/Q 只证明该 teacher state 上末步扰动的致损充分性；不能代替上述恢复性对照。若简单保护有效，先将其作为部署基线，不发明 decoder loss；若无效，说明较早轨迹误差可能仍重要，不反证已知 CFG 放大公式，也不自动证明 decoder 新机制。

覆盖标签：**上位现象与按步精度策略 heavily overlapped；本地原生合同上的修复幅度尚未测定。** 未核实范围不等于剩余论文贡献。
