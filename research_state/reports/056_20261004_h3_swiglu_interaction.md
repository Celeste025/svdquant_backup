# H3 SwiGLU：乘性交互不是本次误差的有害来源

2026-10-04，E062。**本轮停止 gate/up 二阶交互修正候选；没有新方法或视频质量收益。** 普通伪量化/kernel 修复仍只计工程，主实验继续限定 H3。

## 动机 / 观察

H3 的 FFN 将同一输入投影成 gate/up，再计算 `SiLU(g)*u`。单看线性重构 MSE，可能漏掉两支量化残余的乘性交互。待检验的预测是：该交互有稳定、实质的正净误差作用，因而值得为它安排额外的残余表示能力。这是架构提出的假设，之前没有实证支持。

查新已排除泛化“非线性低秩补偿”的新颖性：[SPEAR](https://arxiv.org/html/2606.11244v1)已有门控量化补偿，[NA-LoRA](https://arxiv.org/html/2606.31717v1)已分析 SwiGLU 残余与 gate 敏感性。本次分解本身也不认领为创新。

## 实验

固定两份原 BF16 teacher 状态 p30/s5、p36/s14，预选 block 0/24/49。完成 **2 次完整 BF16 前向＋6 次局部 native W4A4 fc1**；teacher 实际输入和最终 raw/velocity 与历史精确匹配。每个 fc1 使用完整真实输入量化，随后抽取 512 个固定均匀 video 行，保留原 global scale。

将误差分为单支项 a、b 和乘性交互 `c=ΔSiLU(g)*e_u`；通过同一原始 down 权重和实际 AdaLN gate 的 FP32 线性映射 D（TF32 关闭），以 FP64 统计 `net=||D(a+b+c)||²−||D(a+b)||²`。正值表示 c 有害，负值表示 c 抵消误差；这不是独立能量份额。另用同一 teacher 定义的线性投影去除通道均值/输出比例，并单列实际 BF16 四角与 down/gate 算术控制。

预算内完成：34.437 秒、峰 allocated 41.77 GiB、保存张量 2.666 GB。原 worker/tmux 已退出，GPU0 释放。未训练、未生成视频、未运行 MJ。

## 结果 / 结论

下表是 `net / 实际总误差能量`；全部为负。

| 状态 | block | 原始读出 | 去均值及输出比例后 |
|---|---:|---:|---:|
| p30/s5 | 0 | −0.671% | −0.458% |
| p30/s5 | 24 | −0.017% | −0.749% |
| p30/s5 | 49 | −1.972% | −2.265% |
| p36/s14 | 0 | −1.301% | −0.936% |
| p36/s14 | 24 | −1.235% | −0.241% |
| p36/s14 | 49 | −1.619% | −1.791% |

交互项自身能量占总误差约 0.175%–1.180%，其交叉项使净作用为抵消。所有格的 BF16 四角、实际 BF16 down/gate 差值均保持相同符号与停止决定；FP64 小样本数值检查最大相对 RMS 2.87×10⁻⁶。BF16 四角的 c 向量相对数学 c 有 15.8%–83.7% 的差异，说明不能省略算术控制；本次差异没有改变净作用判断。循环移位 up 残余的辅助读出亦无稳定有害作用，但它破坏内容条件关系，不能作独立因果证据。

预设条件要求同一层在两状态均有至少 20% 的正净作用，三个层均不满足。**停止本次乘性交互修正，不扩层、提示或 adapter 训练。** 这不否定所有 FFN 量化问题、SiLU 单支曲率或一般非线性补偿；两状态/六个局部采样也不是生成质量评测。下一研究入口须另有可区分预测，QK 备选尚未执行，不自动立项。

独立 NumPy 复核 complete 6.350 秒、0 CUDA：12 份 PT 重 hash；两份 teacher 输入与历史实际张量、raw/velocity 全部逐字节一致；全部能量统计最大相对差 4.97×10⁻¹⁶，确认三层均不通过。复核 [independent_check.json](/home/wjq/workspace/svdquant-exp/results/research/E062/independent_check.json)，SHA `f14b606717b0649bec6fd292b697edda816b1ef8ff1d68e473da5871cbdfadda`。

复现：计划 [E062](../06_experiments/E062_h3_swiglu_interaction_plan.md)，执行 [evaluate_v2.json](/home/wjq/workspace/svdquant-exp/results/research/E062/evaluate_v2.json)，源码 [v2 runner](/home/wjq/workspace/svdquant-exp/scripts/research/probe_h3_swiglu_interaction_v2.py)。执行报告 SHA `4f7eaf207ebbc2f9af2307f0a9b4d91713f0d1030cc8e71e99d08eac0f9df3e1`。两次早期 CPU 启动失败（已有 av 未进入查找路径、E014 文件名误写）均保留，0 模型调用；v2 仅修路径，未改变实验或 GPU 预算。
