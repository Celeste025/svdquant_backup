# 原生 NVFP4 与高阶 / 自适应 ODE 求解器：有限碰撞

2026-10-02。仅 primary 文献、只读本地合同与推导；无 GPU、无实验代码、无 claim ID。**结论：PARK，不推荐当前立即投入 GPU。** “量化干扰高阶方向估计”“低精度使 embedded error 错判”“额外 NFE 吃掉低精度收益”均已有直接覆盖。尚未核实原生动态 NVFP4 微块跳变在视频模型中的实例，但应用差异还不足以支撑研究。

## 最近邻到底做了什么

| 原始来源 / 已核实段落 | 精确覆盖；剩余边界 |
|---|---|
| [Sampling-Aware Quantization，CVPR 2026 / arXiv v2 2026-04-22](https://arxiv.org/html/2505.02242v2)，§3.3 Eq9–10、§4.1 Eq12、Algorithm 1 | 正面分析量化污染多个中间方向评估，给出 h=ΔlogSNR 时的量化项 `O(δh)` 加截断项 `O(h^(k+1))`，用 Mixed-Order Trajectory Alignment 做 PTQ/QLoRA。其展开使用量化误差的导数序列，并未逐面处理原生 code/scale 跳变；也未核实 embedded accept/reject。**“量化让高阶采样失效”不能再作宽泛新意。** |
| [Mixed-Precision in adaptive Runge-Kutta，2026-05-22](https://arxiv.org/html/2605.23727v1)，§2.2/2.4/2.5、§3.4 Fig3 | BS3(2) 单/双精度及混合 stage；误差估计是两阶输出之差。用解析解检查真实局部误差，明确出现估计值满足 tolerance、真实值不满足的错误接受；还分析额外步数是否抵消低精度速度优势。局限：常规数值 ODE、Matlab 和运行成本 proxy，非 NVFP4 神经网络或视频。 |
| [Mixed Precision Training of Neural ODEs，2025-10](https://arxiv.org/html/2510.23498v1)，§2–误差分析 | 明确指出低精度 RHS 限制可达准确度、adaptive 所需步长可能舍入为零，因而聚焦固定离散化训练；分析量化误差与截断误差的精度下限。它是有关风险的讨论及固定网格分析，不是原生 FP4 controller 实测。 |
| [QuAKE，2026-09-18](https://arxiv.org/html/2609.21407v1)，§3.1–3.5 | 用平滑轨迹先验、条件 Gaussian 观测模型和 Kalman 更新，恢复任意高阶 multistep sampler 消费的 denoiser 输出窗口。没有直接核实原生跳变及拒步机制，但“利用跨评估结构清理量化输出供高阶求解器使用”已被覆盖。 |

补充边界：[Q-Diffusion supplement §D](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Li_Q-Diffusion_Quantizing_Diffusion_ICCV_2023_supplemental.pdf)早已直接测试量化+DPM-Solver++；[DPM-Solver++](https://arxiv.org/abs/2211.01095)也已展示高 CFG 下高阶不一定胜过一阶。它们不是 code-boundary 分析，但禁止把任何“高阶更差”归因 NVFP4。

## 还能严格问什么；不能偷换什么

对一致的 embedded RK 权重 b、b̂，在**同一组冻结 stage states**上，定义 `e_i=f_NVFP4(Y_i,t_i)−f_BF16(Y_i,t_i)`，则估计器的直接改变量为

`ΔE = h Σ_i (b_i−b̂_i)e_i`, 且 `Σ_i(b_i−b̂_i)=0`。

共同 stage 偏差会抵消，而跨 stage 变化会进入估计器；额外的 stage-state 传播误差须另算。code/scale 切换可能使该项按 `h·误差幅度` 而非光滑截断阶缩小，但这只是条件性推断，不保证总误差变大或 NFE 浪费。它也**不是常数 local-error floor**：外面仍有 h。相反，同偏差可能让估计器接近零，而相对 BF16 的漂移仍大。

关键区分：求解器通常估计的是对 **f_NVFP4 自身**的积分误差，不负责检测其相对 BF16 的模型偏差。若 controller 为了更准确积分粗糙的 f_NVFP4 增加 NFE，这不自动是“误判”；若相对 BF16 漂移大而 embedded error 小，也不自动是“漏检”。必须先指定求解目标。同一确定性输入的码值跳变不是独立随机噪声；BF16 输入/输出及矩阵算术本身也有限精度、非全局光滑，不能当连续函数真值。

当前可检验的残余仅为：**在已对 BF16 有真实速度/质量价值的 sampler 中，NVFP4 stage 误差的结构导致容差决策相对其自身数值积分目标失准，且常规控制设置不能消除浪费。** 未见本轮来源直接完成这个原生视频合同；未证实它发生，也还没有对应的新干预。

## 当前模型合同及最小条件式判别

H3 的 [pipeline](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:28) 调用两个 `FlowMatchScheduler("MiniMax-H3")`；[本地 step](/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py:377) 仅更新 `x←x+Δσ·v`。E010 固定 20 次 DiT，无 embedded pair、误差容差或拒步。因此不能用 controller 假设解释 E010；rCM 的少步蒸馏合同也不能未经验证就替换成连续高阶 ODE。

**只有先有合适 sampler，才允许一个局部 probe；不是启动新采样器工程的理由。** 预指定早/晚各一段已有 teacher 状态，固定 prompt、噪声、ODE 状态累加精度，选已有 embedded pair；在原实际步长及两次二分上，用 BF16/native 各自的 stage 评估记录 E、accept/reject、NFE，并在同一初态作短区间子步参考。参考必须对 **同一个数值 RHS**继续二分后稳定，且误差显著小于待判别差；BF16 teacher 路径误差另报。若参考不收敛，不能宣称真实局部误差或 controller 错误。此窗口检查不证明视频质量。

**廉价且强的对照：** FP32 状态/误差累加，合理 tolerance 下限，固定 NFE 的低阶方法与高阶方法；有收益再与 SAQ/QuAKE 区分。scale freezing 只能作改变执行合同的诊断，不能冒充新贡献或不改变误差的因果干预。

**STOP：** BF16 自身不从高阶/自适应获益；实际步长范围内两种精度同样失准；只有码值翻转却无 NFE/实际误差后果；常规 tolerance/低阶选择完全解释收益；或局部参考不可靠。以上任一成立，不进入多视频质量实验。只有跨两段出现可复现的额外浪费、同精度参考验证其失准，并有超出普通控制设置的具体可实施干预，才重新碰撞 prior work。当前不满足这些前提，保留 PARK。

合同澄清：legacy PTQ 的 `num_grids` 是 smoothing-alpha 搜索候选数量，选出每层一套静态配置；E009 导出 200 份静态权重。它不是 10 个 timestep parameterization，也不存在该假设下的 nearest-time-grid 切换。因此不延伸“整套时间量化参数切换”分支。
