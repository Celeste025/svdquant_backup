# E065 — 固定预算的 H3 carry-Q 配对基线

2026-10-04，baseline diagnosis / exploration。在新 GPU 运行前冻结。

## 动机 / 观察 → 问题

E060 发现当前旧 producer 每个候选重新以 Q(Ws) 初始化；本地 DeepCompressor 的 QuantLowRankCalibrator 则携带上一轮量化权重。E061 只更换第一次低秩目标，未测试 carry-Q。不能把其四点 20% 投资门槛扩大为建立必要强基线的门槛。历史 producer 源码未绑定，当前源码差异仍不能反推历史执行。

**假设**：在相同 smooth、rank32、候选数量、种子和 native 选择目标下，携带上一轮 Q 能改变选中分解，并降低完整 H3 同输入预测误差。竞争解释：反复随机初始化已足够；局部拟合收益在完整模型中不兑现；固定 smooth 限制收益。

**本实验回答**：carry 状态是否是当前有限配方的实质差异，是否应优先补齐正式 PTQ。属于必要的基线诊断，不是论文方法贡献。低秩实际激活残余输入的另一候选已有直接近邻，见本轮先前工作笔记，不据此另开方法实验。

## 实验 / 最小有效设置

模型、原始权重、CFG1、native NVFP4/旧 E2M1 tie 规则、BF16 支路、rank32、全部 200 linears、非目标模块保持已有合同。固定 E009/E060 的每层 smooth。

两个新臂：`restart` 与 `carry`。设 Ws=BF16(W*s)，Q0=legacy_Q(Ws)。候选 k=0..7：

- restart：每次 Lk=SVD32(BF16(Ws−Q0))。
- carry：Lk=SVD32(BF16(Ws−Qprevious))，随后 **Qprevious=Qk**，携带最近候选而非 best-so-far。
- Qk=legacy_Q(BF16(Ws−BF16(Bk@Ak)))；A/B 保持 BF16。FP32 randomized SVD，q40、niter2，关闭 TF32；seed=310000+1000*block+50*local+k。
- 两臂都完整评估 8 个候选，不提前停止；按相同校准输出 SSE 选 best，平局选最早。最终 Q/A/B 必须属于同一个选中候选。第 0 候选的量化包、因子、逐样本分数须精确一致。

校准源为原 `results/calib/minimax_h3_svdquant_standard_8p64s`：按原 prompt 顺序隔一取一的 [1,20,46,105] × steps [3,14]，共 8 个缓存 raw DiT 输入。与后续 p30/p36 诊断分离。每条输入重放完整 BF16 DiT；在全部 200 linears 捕获真实输入及原 full-M BF16 输出。

从实际 img/audio/text position_ids 合并排序后的非 padding 联合域，均匀取 512 行，不另作模态重加权。**先对完整 BF16(x/s) 打包，再抽选代码和 logical SF 行，保留完整 tensor global，重新 swizzle 存储；不得重新量化小样本。** 两臂共用保存的同一 packet、未量化 BF16 xs、原 BF16 target。记录行索引及各模态数量，逐 byte 校验包选择。按 8 个样本所有选中输出元素的 native main + 两次 BF16 低秩 GEMM + BF16 相加的 FP64 SSE 总和选候选。每个候选权重量化须由独立旧 QDQ 与 packed decode 检查数值相同。

首校准 p1 s3 / block0 的四种 linear 另以现存 legacy 权重作 full-M 与 M512 的同包读出对照，报告 byte 一致性、最大差及相对量化 SSE；不假定 GEMM 形状改变后舍入逐位相同。此控制只刻画选择目标的计算差异，不能替代最终完整模型验证。

生成两个全部 200 层的新 native 导出。完整模型读出复用 E061 已核验的四个 BF16 teacher 输入 p30 s5/s6、p36 s14/s15；先执行一次 legacy 精确重放，再每臂四次，共 9 native 完整 DiT。BF16 和 plain/legacy 的同输入参照复用现有绑定文件。无 shifted 扩展，无 TE/VAE/视频或 MJ。

## 结果 / 预先规定的解释与决策

主比较为每点 carry/restart 的完整 **video velocity FP64 SSE 比率**，并报告相对 legacy/plain、audio、去通道均值误差。记录全部 200 层的选中迭代与候选曲线，查看局部选择与整模读出是否一致。

- 四点均 carry 优于 restart：优先准备更完整的 matched PTQ 与独立质量评估；效应幅度如实报告，不因未达任意 20% 而宣称基线无价值。
- 异质或无稳定收益：保留候选与局部/整模差异，说明有限 carry-state 对照未兑现一致整模改善；不能排除完整官方校准，不自动扩 seed/rank/提示网格。
- 技术失败：定位实际失败，保留源/产物/消耗；不能记成算法阴性。修复仅在原时间与计算预算内、另版本执行。

局部 loss 不是整模结果的有效廉价替代。四点多次被观察过，只是与本次校准分离的诊断，不是独立泛化/盲测。任何 SSE 改善均不等于画质提升。新 restart 与旧 legacy 的差异包含 native 目标重选，只有 carry/restart 是本轮主配对比较。

## 边界 / 预算

固定 legacy smooth、BF16 teacher 的逐 linear 目标、8 候选截断，不是官方含 LR 的 smooth 搜索、block OutputsError、student 顺序 PTQ 或收敛结果。量化 residual 的 global/SF/code 随迭代变化属于总效应，不能单独归因为低秩方向。512 行目标可能漏掉稀有 token，联合域被 video 主导，不声称 audio 平衡。

一张重查空闲 SM120；CPU check 先行，0 CUDA/0 forward。GPU 全阶段共一个 1800 秒绝对 deadline，峰 allocated <60 GiB，新产物 ≤128 GiB。完整调用上限 17（8 BF16 校准+9 native 诊断）；校准最多 25,600 candidate native GEMM，另 8 shape-control GEMM，完整诊断 1,800 native GEMM。保存输入/源/导出绑定、逐候选读出与必要复核产物，不覆盖旧实验。独立 CPU 复核选择逻辑、首轮配对、输入身份和完整输出统计。不得在无有效结果时重置预算冒充原轮完成。

预算在首次 CPU/GPU 执行前经静态审计修正：原 80 GiB 估算遗漏两臂选中候选的完整局部输出约 49.9 GB；为独立重算校准 SSE 保留它们，总预计约 114 GB，故上限改为 128 GiB。DATA1 空闲约 1.1 TiB。实验样本、候选和时间预算不变；不是看结果后扩大实验。

CPU 检查修订（GPU 尚未启动）：v1 于 2.687 秒、0 CUDA/0 forward 拒绝了 p105 的合法零长 padding 段。原 cu_seqlens=[0,22336,22336]，其 50 个 blocks 只有一个正长 attention 段，合计应为 52 SDPA；其余六校准状态有两个正长段，应为 102。v2 仅将校准调用数检查绑定各输入的真实正长分段，不改数据、采样、候选或算法；原 v1 源与 check.json 保留，新检查写 check_v2.json。诊断四点仍为每次 102。不能把这次 CPU 合同错误记成方法阴性。
