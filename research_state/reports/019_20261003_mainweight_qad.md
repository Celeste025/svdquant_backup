# 阶段019：完成主权重 QAD 训练与原生部署，开发验证尚未受益

2026-10-03。E020 已完成 rCM-Wan 的主权重训练、NVFP4 导出和独立部署计时。训练集拟合改善，但预先固定的第64次更新在两条开发验证 prompt 上，原生输出 pooled NMSE 上升 **7.77%**。这是一条已跑通、仍需改进的成熟 QAD 基线，当前没有新方法或质量匹配后的加速结论。E021 的四臂共16段完整视频也已生成，后续质量评价见[阶段020](020_20261003_qad_video_results.md)。

从原始 BF16 rCM 权重出发，训练30个 block 内全部300个主矩阵，共 **1,391,984,640 个 FP32 master 参数**；bias 和其余参数冻结。采用真实 W/A QDQ 前向、identity STE、AdamW 64次更新，学习率1e-5、weight decay 0、梯度裁剪1；不加入 smoothing 或在线低秩分支。量化沿用现有 legacy-Wan 动态 NVFP4 配方（group16、E4M3 ties-up、E2M1 中点取较大有符号值），不称标准硬件 RNE。FP32 master 直接导出 packed 权重，未在导出前先降为 BF16。

数据固定为4条训练 prompt×4个 timestep，共16条旧缓存输入，以及2条开发验证 prompt×4步，共8条输入。全部目标由当前原始 BF16 teacher 现场重算。这些是开发数据，8个状态彼此相关，不是8个独立测试样本。独立 CPU 汇总核实64次反向与更新、首步300个矩阵均有有限非零梯度、最终300份 Adam 状态；全部300层导出的 codes/scales/global 都发生变化，非目标权重与 bias 保持一致。共168次完整 DiT 前向，训练进程约725.9秒，峰值 allocated 25.423 GiB；这些资源记录不是部署 benchmark。

| 更新次数 | 开发验证 QDQ pooled NMSE | 开发验证 native pooled NMSE |
|---:|---:|---:|
| 0 | 0.140237 | 0.137708 |
| 16 | 0.136512 | 未测 |
| 32 | 0.127596 | 未测 |
| 64（固定终点） | 0.148900 | 0.148410 |

训练集 QDQ pooled NMSE 从0.141154降至0.092964，下降34.14%；这组训练终点数字来自保存的标量日志，未保存完整输出供独立重算。上表开发验证结果则由已保存输出进行独立 CPU FP64 复算。native 的两个 prompt 分别从0.130861→0.125245、0.144351→0.170884，改善并不一致。第32步是本次已观测 QDQ 检查点中最好的一次，仅可作为后续训练设计的探索性线索；不事后替换预先固定的第64步终点，也没有第32步 native 或视频质量证据。QDQ 与 native 的输出差异亦单列保留，不能用训练侧 QDQ 代替部署读出。

部署使用同一 RTX PRO 5000 72GB Blackwell、同一开发验证首条完整输入（prompt0001/step0，31,200 video tokens）、BF16 FLASH attention。四臂在独立于训练的进程中逐个 fresh-load，释放前一模型后再加载；每臂2次预热、5次重复。计时为 CUDA 同步的完整 DiT wall time，排除模型加载、输出 CPU 复制/hash 和 flags 读取。

| 部署臂 | 中位延迟 ms（5次范围） | 模型常驻 GiB | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|---:|
| 原 BF16 | 1789.409（1783.605–1790.325） | 2.649 | 4.099 | 5.768 |
| 旧 SVD nativefast | 1989.774（1980.555–1990.566） | 0.859 | 3.015 | 7.014 |
| plain native，第0步 | 1658.671（1658.148–1660.377） | 0.785 | 2.331 | 6.379 |
| QAD native，第64步 | 1657.540（1656.181–1657.783） | 0.785 | 2.331 | 6.379 |

包含 flags 检查的中位 wall time 依次为1789.411、1989.904、1658.839、1657.689 ms，未隐藏这项成本。常驻统计去重计入参数、buffers，以及 SVD hooks 持有的 LR/smooth 张量；allocated/reserved 是加载结束重置峰值后、含预热/重复和共同输入的分配器读数。每臂释放后的 allocated 均约16.93 MiB，没有把 optimizer 或 master 权重残留计入部署。低位臂的 allocated 较小，但 reserved 并未低于 BF16，不能把 packed 模型大小当成全部显存占用。

28次完整 DiT 均有限且输出形状正确，每臂7次输出 SHA 完全一致。每次实际计数：BF16 为0 FP4 GEMM＋60 BF16 SDPA；其余每臂为300 FP4 GEMM＋60 BF16 SDPA，并通过300份 fastpack flags 检查。合计6300次原生 FP4 GEMM、1680次 BF16 SDPA。

在这个单状态上，QAD 部署相对 BF16 时延下降7.37%、相对旧 SVD 下降16.70%；plain 与 QAD 使用相同部署结构，二者约1 ms差异不能解释为训练带来的速度收益。旧 SVD 还同时涉及已有 PTQ 权重、smoothing、LR、校准及其余旧 loader 配方，当前四臂对照不能把所有差异归因于“有无 LR”。5次同状态重复仅刻画本机此输入的执行波动，不代表长短视频、不同分辨率或 serving 负载，更不能与尚未建立的等质量条件合并成收益主张。

下一步继续推进这条 QAD 强基线：先完成 E021 已生成16段视频的质量评价和配对检查，联合观察原 BF16、旧 SVD、plain 与固定第64步 QAD；随后以更充分的训练 prompt/状态覆盖、预先约定的开发检查点选择和独立最终测试改进训练。第32步线索可以影响下一轮预注册，不能回改本轮终点。首次开发集结果不佳不足以否定主权重 QAD，也不构成转回算子审计或包装普通 QAD 为贡献的理由；研究机会应来自这条可持续训练与实际部署基线上的可复现限制。

来源：[独立数值汇总](../../results/research/E020/summary.json)、[训练记录](../../results/research/E020/train_run.json)、[独立部署计时](../../results/research/E020/deployment_bench.json)、[E020计划](../06_experiments/E020_wan_mainweight_qad_plan.md)、[E021生成记录](../../results/research/E021/generation_run.json)。首轮smoke因E4 masked-load默认值整数0编译失败，0次完整DiT/0次更新；原失败保留，修为浮点0后通过，训练源随后固定。


![固定训练及开发学习曲线](/data1/models/svdquant-wjq/research/20261003/E020/learning_curve.png)
