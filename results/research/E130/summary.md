# E130: RTX PRO 5000 上的 Sage3/Sage2++ 剖析与严格数值预处理优化

## 实验口径

- GPU：RTX PRO 5000 72GB，110 SM；H3 真实 BF16 QKV，非因果，56 heads、head dim 128。
- 9,482-token microbenchmark 使用 E098 捕获输入；22,677-token 回归使用 E121 捕获输入。
- CUDA events：25 次 warmup、每轮 12×12 次调用，正反顺序各一轮。Sage3 的可弃用 K 在计时前恢复；core-only 不含预处理。Nsight Compute replay 不用于延迟。
- 修复版 Sage3 已含 K 尾块的 FP4 物理/逻辑索引掩码修复；未修复的快核不能作为可部署方案。

## 定位结果

论文对齐：SageAttention3 论文 Figure 4 的 RTX 5090 结果是 **attention kernel-only**，图中对照为 **SageAttention2**，不是 SageAttention2++；[作者仓库的 benchmark 说明](https://github.com/thu-ml/SageAttention/blob/main/bench/README.md)明确排除 QKV 量化和 smoothing。我们同时测了在线 API 与 core-only，并在短 H3 输入上分别使用 Sage2 FP32 和 Sage2++ 对照，因此不把论文的 ~1.8× 直接拿来要求本机在线 H3 路由达到。论文原文与 Figure 4：[SageAttention3](https://arxiv.org/html/2505.11594)。

设备规格也不同：[NVIDIA RTX PRO 5000 72GB 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf)列 2,064 AI TOPS、1,344 GB/s、300 W；[RTX 5090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)列 3,352 AI TOPS、1,792 GB/s、575 W。标称 AI TOPS 比例约 0.616，但前者数据表注明它是**带稀疏性的 FP4 峰值**，不能作为实际稠密 attention 或 Sage3/Sage2 加速比的直接预测。物理卡差异能解释绝对 TOPS 的一部分；同卡对比中 Sage3 Tensor pipe 只有约 49–51% 而 Sage2++ 约 74–75%，说明剩余问题仍在内核资源/流水线，而非单纯换算硬件峰值。没有 RTX 5090 实测时不能声称同代码在 5090 必然重现论文比例。

| 核心 | 在线时间 (ms) | 仅核心 (ms) | Tensor pipe | Shared/CTA | Spill requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sage2++ | 7.364 | 6.465 | 73.6% | 32.8 KiB | 0 |
| Sage3 修复版，M128/N128/stage3 | 7.729 | 4.839 | 48.8% | 100.4 KiB | 345,300 |
| Sage3 未修尾块，不能部署 | 7.580 | 4.696 | 51.1% | 100.4 KiB | 0 |

原 Sage3 核心比 Sage2++ 快，但在线预处理约 3.17 ms 对比 Sage2++ 约 1.06 ms，使在线时间反而略慢。100 KiB/SM 共享内存上，修复版 Sage3 的单 CTA 已占约 98 KiB，无法并驻第二个 CTA。相关原始数据：`baseline_maskfix_two_scope_gpu2.json`、`sage3_maskfix_stages_gpu2.json`、`sage2pp_stages_gpu2.json`、`ncu_core_summary.json`。

22,677-token 真实 H3 QKV 的在线四臂配对测试（同 GPU2，正反运行顺序）进一步确认差距并非只由短序列预处理造成：BF16 SDPA 70.42 ms / 209.4 TOPS，Sage2++ 38.72 ms / 380.8 TOPS，修复版 Sage3 34.97 ms / 421.7 TOPS，严格数值预处理版 Sage3 33.31 ms / 442.6 TOPS。后者相对 Sage2++ 是 1.16×，并非论文 RTX 5090 图中的约 2×；两种 Sage3 输出逐元素相等。相对 BF16 的输出 NMSE 分别为 Sage2++ `1.88e-4`、Sage3 `7.04e-3`。这些只针对该 H3 形状，不外推成通用 SageAttention 性能。见 `sage_long_four_arm_gpu2.json`。

同一次 22,677-token 双口径基准补齐了论文式 core-only 比较：Sage2 FP32 核心 35.623 ms / 413.9 TOPS，Sage2++ 核心 36.482 ms / 404.2 TOPS，Sage3 核心 27.001 ms / 546.1 TOPS；Sage3 相对两者分别为 1.32×、1.35×。在线 API 分别为 37.612、38.503、34.834 ms，Sage3 的在线减核心约 7.83 ms，Sage2 两臂约 1.99–2.02 ms。两个口径差距不能直接当成某个单独算子的独占时间，因为它们是分别计时；但足以表明预处理是剩余性能缺口的重要部分。见 `long_two_scope_gpu2.json`。

同形状 Sage3 分段 CUDA-event 计时显示：中心化、padding、`delta_s` 合计 7.025 ms；Q/K/V FP4 打包分别 0.419/0.412/0.418 ms；核心 25.117 ms；输出裁剪连续化 0.621 ms。前一阶段细分为 K mean 0.337、K 减均值 0.703、Q/K/V padding 0.940/0.938/0.937、Q group mean 0.640、`delta_s` BF16 GEMM 加 FP32 cast 2.538 ms。分段和独立核心计时不同，不把它们混为一个精确加法分解。见 `long_sage3_preprocess_stages_gpu2.json`。

将最后一项再次细分，BF16 `delta_s` GEMM 为 1.255 ms，单独的 BF16→FP32 cast 为 1.290 ms（重复测量与合并项一致，见 `long_sage3_preprocess_stages_split_gpu2.json`）。由于 GEMM 输出原本是 BF16，cast 不增加数值信息，却把这个约 0.9 GB 的补偿矩阵按 FP32 写出和传给核心。这给“核心直接 TMA 读取 BF16 补偿，并在寄存器中精确转 FP32”提供了可检验的目标；需要同时测核心变慢与在线净收益，不能直接把 1.29 ms 当作保证的加速。

现已实现该候选：TMA 直接读取 BF16 `delta_s`，在寄存器中逐个精确扩展成 FP32，再沿用原来的校正累加路径。1,024 与 9,482-token 输出相对修复版基线均逐元素相等；在 22,677-token 的双口径同形状基准中，公开 API 在线时间 34.834→34.201 ms，而核心 27.001→27.461 ms。省掉 cast 的净收益约 0.63 ms，不能把 1.29 ms 的 cast 时间全部算作加速。见 `bf16delta_vs_baseline_{1024,9482}.json`、`bf16delta_long_two_scope_gpu2.json`。

同一 BF16 内核上，严格数值的 Q-pad/mean 和 K/V 隐式 pad 快路径进一步节省预处理；再把 K 的中心化与 padded K 的构造合并成单个 Triton 写入，省掉单独的 K 原地减均值与 K pad copy。K 融合版在 127、128、129、255、256、257、1,024、1,025、9,482 和 22,677 token 的捕获输入上，均与候选公开 API **逐元素相等**，包括完整/尾端 K tile。22,677-token 同进程正反顺序交错 CUDA-event 测试：公开 BF16 `delta_s` API 34.278 ms；严格预处理 32.630 ms；叠加 K 融合 31.657 ms，即相对该公开 API 为 1.083×，相对不融合 K 的快路径为 1.031×。另外独立进程复测原 FP32 `delta_s` 公开 API 34.649 ms、严格预处理 33.026 ms；跨进程绝对时间只供参考，不将其比值称为严格配对收益。见 `bf16delta_fast_preprocess_validation_gpu2.json`、`bf16delta_fused_kpad_validation_gpu2.json`、`bf16delta_fusedk_threeway_long_gpu2.json`、`fp32delta_fast_long_ab_gpu2.json`。

9,482-token 非连续 HND 捕获输入的同进程三臂计时亦支持收益：公开 BF16 版 7.575 ms、严格预处理 6.771 ms、再融合 K 后 6.372 ms，后者对公开 BF16 版为 1.189×。短序列预处理占比更高，所以相对收益比 22,677-token 更大。见 `bf16delta_fusedk_threeway_short_gpu2.json`。

BF16 校正现作为 `third_party/sage3_h3_maskfix` 的**编译时可选**路径保存：独立复制源码后，用 `E130_BF16_DELTA=1 bash scripts/research/build_e130_sage3_maskbit.sh <source-copy>` 构建。未设置该变量仍为原 FP32 版本，不能在原目录直接编译 BF16 以免覆盖现有二进制。扩展通过 `delta_s_dtype` 属性告知 Python 预处理应保留 BF16 还是转为 FP32，C++ 入口会检查 dtype。源码的独立 BF16 构建已成功，扩展属性实测为 `bf16`；该构建的公开 API 相对原 FP32 基线 1,024-token 输出逐元素相等，融合 K 的 10 种长度也全与其公开 API 逐元素相等。原目录现有 FP32 二进制在源码改动后重新捕获 1,024-token 输出仍与旧基线逐元素相等。见 `conditional_bf16_vs_baseline_1024.json`、`conditional_bf16_fusedk_validation_gpu2.json`、`default_fp32_postedit_regression_1024.json`。自由生成、多 GPU/图捕获尚未回归，故此时仍是实验候选而非默认部署路径。

原 FP32 构建和可复现 BF16 构建在 22,677-token 长 H3 捕获输入上的完整 BF16 输出 SHA-256 也相同：`d612593d07e57c0ceebfe022f2ee8e9bbecdba3b35233dd3199c01c08d031bca`。此检查只保存两个小 JSON，不落盘约 325 MB 的输出张量；见 `fp32delta_long_output_sha256.json` 与 `bf16delta_long_output_sha256.json`。

H3 实际 NHD→HND 布局转换、attention 和 NHD 输出组装的 9,482-token 路由计时：公开 BF16 路由含不必要 K clone 为 8.884 ms；新 BF16+严格预处理+融合 K 且去掉 clone/crop 为 7.264 ms，约 1.223×；前者不应被解释为仅内核加速。该路由的输出与公开 API 相同，见 `bf16delta_h3_routed_gpu2.json`。

完整常驻 H3 DiT 的两个 teacher-forced cached steps 也完成回归；264 个 linear 权重全部留在 GPU，模型加载不计入，VAE/text encoder 不加载。公开 BF16 `delta_s` Sage3 与新的融合版的 video/audio 输出在两个 step 均**逐元素相等**：

| Cached step | BF16 DiT | 公开 Sage3 | BF16 `delta_s`＋融合 K | attention helper 合计（公开→融合） |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.222 s | 6.593 s | 6.448 s | 1.884→1.693 s |
| 10 | 8.411 s | 6.700 s | 6.533 s | 1.906→1.705 s |

因此相对公开 Sage3 的完整 DiT 净加速约 2.2–2.6%，attention helper 合计下降约 10%。这不是自由生成端到端耗时，且仅验证了两个 cached steps；完整记录见 `bf16delta_fusedk_h3_resident_steps0_10_gpu2.json`。原先的 FP32 严格预处理路径另有结果，表中“公开 Sage3”不是那条快路径。

22,677-token BF16 校正核心的 Nsight Compute 单 launch 回归说明**核心瓶颈尚未解决**：shared/CTA 从原修复版 100.352 KiB 降至 99.328 KiB，但仍只能单 CTA/SM；寄存器仍为 168/thread，spill requests 仍为 808,314。Tensor pipe 活跃度从 51.0% 变为 49.7%，short-scoreboard stall/issue 从 0.786 变为 0.808。它与 CUDA-event 的“核心变慢约 0.46 ms、在线整体反而变快”一致：BF16 `delta_s` 主要靠避免 FP32 cast 和降低校正矩阵的全局访存获益，不是提升 FP4 MMA 饱和度。见 `ncu_long_bf16delta_comparison.json`；原始候选报告保存在 `/data1/models/svdquant-wjq/research/20261025/E130/sage3_bf16delta_long_core_gpu2.ncu-rep`，Nsight replay 耗时不可当作正常延迟。

### 将尾端 padding mask 前移到 `delta_s`

进一步的核心实验把 `delta_s` 中对应 padding K token 的列静态设为 `-∞`，由 BF16 校正项随 QK MMA 一起使这些 score 变为 `-∞`；非因果核心不再对每个 score 计算 FP4 K 的物理→逻辑 token 索引和 mask。保留旧核心 mask 时，预先写 `-∞` 在 10 种长度均逐元素相等。独立编译移除核心 mask 后，129 和 22,677-token 输出 SHA-256 仍与原修复版一致；新核心的快路径在 10 种长度上也与其公开 API 逐元素相等。见 `delta_tailmask_preflight_gpu2.json`、`delta_tailmask_{baseline,candidate}_129.json`、`delta_tailmask_candidate_22677.json`、`delta_tailmask_candidate_fast_validation_gpu2.json`。

22,677-token CUDA-event：新公开 API 在线 34.007 ms、核心 27.177 ms；仅 BF16 校正但保留核心 mask 的对应数值是 34.201/27.461 ms（跨进程比较，未把差值夸大为精确收益）。同进程正反顺序三臂：新公开 API 33.899 ms、严格预处理 32.224 ms、再融合 K 为 31.232 ms；9,482 token 对应 7.474/6.708/6.305 ms。见 `delta_tailmask_long_two_scope_gpu2.json`、`delta_tailmask_fusedk_threeway_{long,short}_gpu2.json`。

22,677-token Nsight 对比确认核心 mask 的寄存器溢出被消除：spilling requests 从 808,314 降为 **0**，shared 99.328 KiB/CTA，寄存器仍 168/thread，Tensor pipe 活跃度约 51.1%。所以 mask 是 spill 的触发点，但 spill 清零后 Tensor pipe 未明显提高，说明单 CTA 驻留、同步/softmax 依赖仍限制核心。见 `ncu_long_delta_tailmask_comparison.json`，原始报告 `/data1/models/svdquant-wjq/research/20261025/E130/sage3_delta_tailmask_long_core_gpu2.ncu-rep`。

更细看调度器计数，不能把低 Tensor pipe 简单归因为“活跃 warp 数不够”：无 spill Sage3 的平均活跃 warps 占峰值 20.83%、每 scheduler 每活跃周期可发射候选 warps 0.599；Sage2++ 对应 16.59%、0.479。Sage3 的 short-scoreboard stall/issue 却为 0.799（Sage2++ 0.254），sleeping stall/issue 为 0.902（Sage2++ 0）。尾块优化后的 SourceCounters 仍把最显著的有源码热点归到 `softmax_fused.h:121` 的逐元素 exp2 及 `:122` 的行求和；`delta_s` BF16 展开次之。表明下一个更有价值的问题是每个 FP4 MMA tile 间的 softmax/概率量化与同步如何重排，而非仅追求更多驻留 CTA。见 `delta_tailmask_long_cuda_line_hotspots.json`；stall 指标存在重叠，不能相加当作时间比例。

在**已消除 spill** 的同一源码上又独立编译了奇偶两路 `row_sum` 累加器，以检验此前两路累加的负收益是否主要由 spill 造成。结果仍为负：22,677-token 同口径 CUDA-event 测试，核心由 27.177 增至 28.518 ms（慢 4.94%），公开在线 API 由 34.006 增至 35.310 ms（慢 3.83%）；同轮 BF16 与 Sage2++ 时间仅分别相差约 0.06% 与 0.07%，不支持明显时钟漂移解释。9,482-token 输出相对原修复版 NMSE 为 `4.94e-9`，因此也失去逐元素相等性。此候选仅存在于 `/tmp/e130-twoacc-spillfree.F4HBQu`，**未合入项目源码或默认二进制**。记录见 `twoacc_spillfree_long_two_scope_gpu2.json`、`twoacc_spillfree_vs_baseline_9482.json`。这一负结果把下一步重点从单纯拆分行和依赖，转向检查 `exp2`、行和与概率 FP4 量化的指令重叠及 tile 间同步。

另一个独立候选只把非首 tile 的 `exp2` 循环与 `row_sum` 循环分开，不增加第二个累加器。9,482-token 输出与原修复版逐元素相等，但 22,677-token 核心 27.177→27.223 ms、公开在线 API 34.006→34.009 ms；同轮 Sage2++ 38.578→38.539 ms，故无法确认任何正收益。候选仅存在于 `/tmp/e130-softmax-splitloop.bTpwxa`，**不合入默认源码或二进制**。见 `splitloop_vs_baseline_9482.json`、`splitloop_long_two_scope_gpu2.json`。两次调度尝试共同说明：把行和依赖简单拆开或延后，并不足以提高本形状下的 FP4 MMA 利用率。

### BF16 校正项的原生转换指令

`delta_s` 以 BF16 存放后，原核心用整数移位和位重解释展开为 FP32。RTX PRO 5000 所用的 CUDA 工具链还支持 `cvt.f32.bf16`；在**仅替换这段展开**、其余 tile/路由不变的独立候选上，9,482-token 输出逐元素等于原修复版，22,677-token 输出 SHA-256 亦与原修复版相同（`d612593d07e57c0ceebfe022f2ee8e9bbecdba3b35233dd3199c01c08d031bca`）。公开 API 与融合 K 快路径在 127、128、129、255、256、257、1,024、1,025、9,482、22,677 token 均逐元素相等。见 `bf16cvt_vs_baseline_9482.json`、`bf16cvt_long_output_sha256.json`、`bf16cvt_10shape_validation_gpu2.json`。

22,677-token 同卡 CUDA-event，候选核心 26.646 ms、在线 33.445 ms；此前无 spill 基线 27.177/34.006 ms，候选之后立即重测基线 27.232/34.060 ms。两次基线夹住候选，说明核心约快 2%、在线约快 1.7%，而 Sage2++ 同轮只在 38.53–38.66 ms 间波动。9,482-token 候选 4.763/7.389 ms，对应复测基线 4.847/7.456 ms，收益较小但方向相同。见 `bf16cvt_{long,short}_two_scope_gpu2.json`、`bf16cvt_{long_baseline_repeat,short_baseline}_two_scope_gpu2.json`。它仍远不足以把 Sage3/Sage2++ 的本机比例提升到论文 RTX 5090 的比例。

22,677-token Nsight Compute 单核心 launch：候选 shared 99.328 KiB/CTA、168 registers/thread、0 spill，均与无 spill 基线相同；Tensor pipe 51.06%→52.06%，short-scoreboard stall/issue 0.799→0.742，sleeping 0.902→0.857。指标方向与 CUDA-event 加速吻合，但 PC/pipe 百分比不是某段代码的独占耗时。原始报告 `/data1/models/svdquant-wjq/research/20261025/E130/sage3_bf16cvt_long_core_gpu2.ncu-rep`，摘要 `ncu_long_bf16cvt_comparison.json`。

常驻 H3 DiT 的第 0/10 个 cached step 中，候选融合路由直接时间 6.425/6.503 s，先前同配置位移展开版本为 6.447/6.533 s；attention helper 1.658/1.667 s 对比先前 1.681/1.693 s。两步 video/audio 与候选公开 Sage3 逐元素相等，且相对 BF16 DiT 的 NMSE 数值与先前运行完全一致；但两次完整 DiT 运行没有跨进程输出哈希，因此不把 NMSE 数值相等冒充跨构建逐位证明。单轮 DiT 净变化仅约 0.3–0.5%，尚需重复测量。见 `bf16cvt_h3_resident_steps0_10_gpu2.json`。

源码已将转换保存为默认关闭的 `E130_BF16_CVT=1` 构建选项，要求 `E130_BF16_DELTA=1`；实际推荐与 `E130_DELTA_TAIL_MASK=1` 和融合 K 预处理搭配。用三个开关从项目源码的干净副本重新编译成功，最终构建在 `/tmp/e130-bf16cvt-final.XGrFq0`；该构建在 10 种长度上继续通过逐元素的公开/融合路径检查，22,677-token 输出哈希仍等于原修复版，公开在线/核心时间为 33.424/26.656 ms，和先前的实验构建 33.445/26.646 ms 一致。对应 `bf16cvt_final_10shape_validation_gpu2.json`、`bf16cvt_final_22677.json`、`bf16cvt_final_long_two_scope_gpu2.json`。

最终扩展另存于 `/data1/models/svdquant-wjq/research/20261025/E130/optin_bf16_deltamask_cvt/`；其中 `fp4attn_cuda` SHA-256 为 `96f8f55c23b3d6f96f9b817eed4386c7546328d3bec42ff21ec0044c2f5ac639`。原目录默认 FP32 `fp4attn_cuda` SHA-256 仍为 `e4d51cce159a2537afba74c55096800777112f30809ed8f8062d1bc735b5cc7d`，**没有被覆盖**。复用时应把两个扩展同时放进独立源码副本，并令该副本的 Python 包优先加载；不要只替换默认 `.so`。

同一最终构建的完整精确快路径（BF16 校正、尾块 mask 前移、融合 K 中心化/padding）在 22,677-token 输入上为 30.739 ms；先前不含原生 BF16 转换的同一路径为 31.232 ms，净差约 0.49 ms，和公开 API/核心差值一致。候选构建内的公开、严格快路径、融合 K 快路径均逐元素相等；正反运行顺序的两轮数据见 `bf16cvt_fusedk_threeway_long_gpu2.json`。对照 Sage2++ 在线约 38.58 ms，这一路径为约 1.25×；仍未达到 RTX 5090 论文式比例，且不能把它称作同 GPU/形状的论文复现。

另比较了两个保持 padded `delta_s` GEMM 的精确 K 融合方案。中心化＋NVFP4 打包融合版本与公开 API 逐元素相等，但同进程正反顺序 22,677-token 平均 **31.662 ms**，慢于中心化＋padding 融合版 **30.768 ms**；所以不替换当前路径。虽然前者把中心化写入并入 K pack，后续为保持 BF16 GEMM 归约路径仍须另物化 padded K，省掉的 pass 没有转成净收益。见 `bf16cvt_kpack_vs_kpad_long_gpu2.json`。

进一步实验把互不依赖的 Q 块均值/打包、V 打包与 K 均值/中心化分配到不同 CUDA stream；`delta_s` GEMM 只等 Q 均值和 K 中心化，核心之前再等全部打包。加入了跨 stream 的事件依赖和张量 `record_stream`，输出先通过公开 API 的逐元素一致性检查。22,677-token 同进程正反顺序平均：异步 **30.726 ms**，当前串行融合版 **30.756 ms**；9,482-token 则是异步 **6.243 ms**、串行 **6.229 ms**。两个形状的差别都只有约 0.1–0.2%，方向不一致，不能确认异步有净收益。实验实现仅在 `scripts/research/e130_sage3_async_preprocess.py`，未接入默认 H3 路由；见 `bf16cvt_async_prepack_{long,short}_gpu2.json`。

据此可做一个口径清晰的性能上限估计：本形状当前 Sage3 精确融合在线约 30.74 ms，其中纯核心约 26.66 ms；即使**假设全部在线预处理零成本**，相对同卡 Sage2++ 在线约 38.55 ms，也仅是约 1.45×。要达到同卡在线 1.8×，总时延需约 21.4 ms；若其余约 4.1 ms 不变，核心需降到约 17.3 ms，比现在再快约 35%。这些是由测得时延推导的目标，不是可实现性承诺，也不能直接拿论文 RTX 5090 的不同硬件/基线结果当作同卡目标。下一阶段若追求显著突破，应优先重排核心的 QK→softmax→FP4-PV 依赖或减少每 tile 的同步/转码，而非继续期待单项预处理微调产生论文式加速比。

核心另试用源码已有的 `fma.rn.f32x2` 辅助指令，把 `rescale_o` 中相邻两个 FP32 输出的逐标量乘加打包；9,482-token 输出与 BF16-CVT 基线逐元素相等。但 22,677-token 核心 **26.678 ms** 对比基线 **26.656 ms**，在线 **33.456 ms** 对比 **33.424 ms**，同轮 Sage2++ 基本稳定，未见可辨认收益。该改写只保留在 `/tmp/e130-rescale-f32x2.KmX41e` 的独立候选中，**未合入项目源码或默认二进制**。见 `rescale_f32x2_vs_bf16cvt_9482.json`、`rescale_f32x2_long_two_scope_gpu2.json`。即使缩减某些标量指令，整体仍由 FP4 MMA 与 softmax/同步的跨 tile 关键路径支配。

又把每行 softmax 中最热的 `score * softmax_scale_log2 - max_scaled` 两两打包为 `fma.rn.f32x2`，保持 `exp2` 与 `row_sum` 的顺序。9,482-token 输出与 BF16-CVT 基线逐元素相等。22,677-token 核心 **26.564 ms**、在线 **33.391 ms**，随后反向复测基线为 **26.753/33.592 ms**；同轮 Sage2++ 也由 36.550/38.572 变为 36.679/38.740 ms，说明有整体时钟漂移。按 Sage2++ 同轮变化粗略归一化后，核心净改善仅约 0.35%、在线约 0.1–0.2%；9,482-token 候选核心/在线 4.722/7.355 ms，对应反向基线 4.773/7.387 ms，仍是小幅差值。见 `score_f32x2_{long,short}_two_scope_gpu2.json`、`score_f32x2_{long,short}_baseline_repeat_gpu2.json`。

同形状 Nsight：候选 168 registers/thread、99.328 KiB shared、0 spill，均未变；Tensor pipe 52.06%→53.36%，但执行的 FFMA 指令数两者**完全相同**（均 `1,981,957,376`），MUFU 与 OMMA 指令数也相同，SASS 导出没有可见的 FP32x2 FMA 指令。由此推断 PTX 打包未减少实际机器乘加条数，小幅计时变化主要是调度/采样差异，不足以作为部署优化保留。候选仅在 `/tmp/e130-score-f32x2.vUXGq7`，**未合入项目源码或默认二进制**。原始 Nsight 报告 `/data1/models/svdquant-wjq/research/20261025/E130/sage3_score_f32x2_long_core_gpu2.ncu-rep`，摘要 `ncu_long_score_f32x2_comparison.json`、`score_f32x2_long_sass_hotspots.json`。

该变体现作为 `E130_BF16_DELTA=1 E130_DELTA_TAIL_MASK=1` 的双重 opt-in 编译路径保存在源码里；扩展属性 `delta_tail_mask_in_delta=True` 使公开 API 与实验快路径自动把 padded `delta_s` 设为 `-∞`，避免调用者漏设。重新从保存的源码在独立目录构建成功，129 与 22,677-token 输出哈希均与原 FP32 修复版相同，10 种长度的自动快路径全逐元素相等，见 `delta_tailmask_final_{129,22677}.json`、`delta_tailmask_final_auto_validation_gpu2.json`。生产默认 FP32 二进制未被覆盖。

验证用的 opt-in 扩展已另存于 `/data1/models/svdquant-wjq/research/20261025/E130/optin_bf16_deltamask/`；其中 `fp4attn_cuda` SHA-256 为 `cdf6f0bd91f8cd449b32abbe141c6dc6a376f9652c3260720012e0ab3d7a259e`，默认目录原 FP32 `fp4attn_cuda` 仍为 `e4d51cce159a2537afba74c55096800777112f30809ed8f8062d1bc735b5cc7d`。使用时应把这两个扩展放进独立的源码副本并令其优先于默认路径加载，不能只替换默认 `.so`，因为 Python 预处理必须匹配编译变体。

这份最终源码的常驻 H3 DiT 第 0/10 步完整输出 video/audio 均与其公开 Sage3 路由逐元素相等；直接计时分别为公开 6.594/6.701 s、融合版 6.447/6.533 s，attention helper 合计约 1.872→1.681 s、1.893→1.693 s。模型级净收益约 2.2–2.5%，和先前仅 BF16 校正但保留核心 mask 的版本几乎一样；这与单算子额外收益不到 1% 一致。见 `delta_tailmask_final_h3_resident_steps0_10_gpu2.json`。仍未做自由生成、CUDA Graph、多卡/异构形状回归，故保持实验开关而不设为默认。

同一长序列的 Nsight Compute 核心采样：Sage2++ Tensor pipe 75.2%，0 spill；Sage3 Tensor pipe 51.0%，808,314 次 spill requests。Sage3 的 short-scoreboard stall/issue 为 0.786，Sage2++ 为 0.254；Sage3 还有 sleeping stall/issue 0.864。SourceCounters 把有源码行的最高采样热点定位到 `softmax_fused.h:121` 的指数与缩放、`:122` 的行求和；无源码行的 SASS 样本还显示同步/phase-check 热点。PC 采样不是各步骤的独占时间，不能按样本数直接估算某一优化的速度上限。见 `ncu_long_core_summary.json`、`sage3_long_cuda_line_hotspots.json`、`sage3_long_sass_hotspots.json`。

资源容量有另一层含义：本卡每 SM 65,536 registers、102,400 字节 shared。Sage2++ 的 128 threads × 255 registers = 32,640 registers/CTA，shared 32,768 字节/CTA，资源上可以双 CTA 驻留；Sage3 的 384 × 168 = 64,512 registers/CTA，shared 100,352 字节/CTA，两个资源都把它限制为单 CTA。Nsight 的 Tensor pipe 利用率差、同步等待及 spill 与这种流水线资源紧张相吻合，但不能仅凭占用上限断定全部 25 个百分点的利用率差都由它造成。

试过把 `AbsMaxP` 的除法改为预先求倒数再乘。旧/新内核分别在独立 Python 进程加载（避免同名扩展缓存），顺序旧→新→新→旧，两轮各 96 次 CUDA-event 调用。长序列均值 34.495→34.483 ms，差异只有 0.034%，低于可置信的实际收益；短序列 7.750→7.728 ms，约 0.28%。两个形状的输出 SHA-256 均完全一致。该改写与编译二进制都已撤回；这也说明当前源码中显式倒数不是主要瓶颈。原始记录见 `softmax_reciprocal_{baseline,reciprocal}_r{1,2}_gpu2.json`。

另试过将每行 softmax 概率和拆成奇偶两个独立累加器，以缩短串行加法依赖。9,482-token 输出相对原版 NMSE `9.50e-9`，但在线时间在 22,677-token 由约 34.50 ms 增至 36.54 ms（约慢 5.9%）；原版在新版本之后复测仍为 34.50 ms，排除了明显时钟漂移。9.5k-token 也由约 7.75 ms 增至 8.19 ms。此改写与编译二进制已撤回。对应输出比对 `softmax_twoacc_vs_baseline_9482.json`，计时 `softmax_reciprocal_twoacc_r1_gpu2.json` 与 `softmax_reciprocal_baseline_r3_gpu2.json`。

持久化 grid 扫描（22,677 token，正反顺序各 96 次 CUDA-event 调用，K 恢复不计时）显示 110 CTA 最快：55→57.57 ms、80→42.53 ms、96→37.40 ms、**110→34.87 ms**、128→50.32 ms、165→41.45 ms、220→34.97 ms。所有输出与 110 CTA 逐位相同。每 CTA 约 100.4 KiB shared，无法在 PRO 5000 每 SM 并驻两个 CTA；110 恰好一波铺满 110 SM。128/165 等非整波 grid 会留下只占部分 SM 的第二波持久化 CTA，因而明显变慢；220 是两整波，但没有胜过 110。可调 grid 的实验代码已撤回，默认 110 CTA 及其二进制均已恢复。原始数据见 `sage3_grid_sweep_long_gpu2.json`。

`M128/N64/stage1` tile 可编译且输出有限，但没达到双 CTA 驻留：Nsight Compute 测得动态 shared 53.248 KiB/CTA（双驻留上限约 50 KiB）、168 registers/thread，Tensor pipe 活跃度 44.5%。22,677-token 在线时间 38.01 ms，之后复测原 `M128/N128/stage3` 为 34.47 ms；因此慢约 10.3%。相对原 tile 的输出 NMSE 在 1,024 token 是 `8.32e-6`，在 9,482 token 是 `4.21e-6`，这种 tile 改动不保持逐位一致。该候选及其二进制已撤回。数值记录 `n64_stage1_vs_baseline_{1024,9482}.json`，时间记录 `softmax_reciprocal_n64_r1_gpu2.json` 与 `softmax_reciprocal_baseline_r4_gpu2.json`。

进一步读取同一 Nsight Compute 报告的设备属性：每 SM 只有 65,536 个 32-bit register、102,400 字节 shared memory。`M128/N64/stage1` 的 384 threads × 168 registers/thread = 64,512 registers/CTA，单 CTA 已占寄存器文件的 98.4%；即使通过对齐优化把 53,248 字节 shared 降到 51,200 字节以下，寄存器仍阻止两个 CTA 并驻。为此终止了尚未完成的“仅降低 shared buffer 对齐”编译，并确认它没有覆盖已验证二进制；源码也恢复。这将优化方向从单纯缩 shared 转向同时降低每 CTA 的寄存器需求和软最大值/同步开销。`M64/stage1` 旧候选约 232 registers/thread × 256 threads = 59,392 registers/CTA，也不能双 CTA 驻留；若维持 256 threads，双驻留需要每线程不超过约 128 registers，难度远高于只改 smem 对齐。

尾块掩码的位运算改写使 spill 增至 724,580；仅在尾块执行逆置换虽降至 310,320，但时间仍约 4.87 ms。stage2 使用 80.9 KiB/CTA，时间约 4.94 ms；stage1 使用 62.5 KiB/CTA，时间约 4.83 ms，仍不能双 CTA 驻留。M64/stage1 即使在无 padding 的 256 token 输入上也不保持数值一致（NMSE 0.612），且编译器每线程分配约 232 个寄存器，不能作为双 CTA 方案。以上负收益核心改动均已撤回，当前核心恢复为 M128/N128/stage3。

## 保留的严格数值预处理快路径

1. 一次 Triton kernel 同时做 Q 的 128-token group mean 与尾端 padding。
2. K/V FP4 packer 直接接收未 padding 的输入，并写入 padding 后的 code/scale 缓冲；不再物化 padded V。
3. **补偿矩阵乘仍使用 padded K**，保持原版 BF16 GEMM 归约路径。对未 padding K 先 GEMM 再补零虽只让单次 attention 输出改变约 `5.8e-9` NMSE，但在完整 50-block DiT 中可放大，故不作为严格数值快路径。
4. H3 路由已有独立的 contiguous HND K，不再 clone；attention 的 strided cropped 输出直接复制进最终 NHD 缓冲，避免额外 HND `.contiguous()`。

9,482-token 在线 attention：原修复版约 7.74 ms，严格数值预处理版约 6.97 ms，输出逐元素一致。包含 H3 NHD→HND 布局和最终 NHD 输出拷贝时：当前路由 9.055 ms，优化路由 7.857 ms（1.152×），输出逐元素一致。见 `preprocess_ab_exact_delta_gpu2.json`、`h3_routed_sage3_exact_gpu2.json`。Q/K/V packed code/scale 及补偿矩阵也与原版完全一致；9,482 和 22,677 token 的验证见 `kcenterpack_validation_gpu2.json` 与 `preprocess_shape_validation_gpu2.json`。

完整常驻 H3 DiT（264/264 linear 权重在 GPU，模型加载不计时）：

| Cached step | BF16 (s) | Sage3 原版 (s) | Sage3 优化 (s) | Attention helper 原版→优化 (s) | video/audio 与原 Sage3 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 8.284 | 6.659 | 6.565 | 1.933→1.792 | 逐元素相等 |
| 10 | 8.437 | 6.753 | 6.633 | 1.953→1.803 | 逐元素相等 |

见 `h3_resident_steps0_10_exact_gpu2.json`。完整 DiT 净增益约 1.4–1.8%，小于短序列单次注意力收益；长序列由核心计算主导，且 attention 只占完整 DiT 的一部分。没有推断自由生成质量或 RTX 5090 上的实测加速。

## 待继续

### N=256 / stage=1 候选：数值不成立，未进入性能比较

为检验将 K/V 每次迭代的 tile 从 128 扩到 256 是否能减少 softmax 更新与同步，独立复制已验证的 BF16-CVT＋尾端 mask 前移源码至 `/tmp/e130-n256-stage1.c1WY6y`，仅将 `launch.h` 的 `M128/N128/stage3` 改为 `M128/N256/stage1`，并将公开 API 的序列 padding 改为 256 倍数。编译完成，短尾输入也能运行且输出有限。但在相同捕获 Q/K/V 上，相对已验证的 N128 构建，129-token 输出 NMSE 为 **0.00555**，9,482-token 输出 NMSE 为 **0.08733**；见 `n256_stage1_vs_bf16cvt_{129,9482}.json`。这远大于正常 tile 顺序改变的微小浮点差异，**候选不具备数值正确性，不用它汇报吞吐或生成**。同时 ptxas 报告多个实例每线程 168 registers 且出现 320–1,204 bytes spill stores；减少循环次数并不保证缩短关键路径。尚未隔离出 N256 的具体布局/softmax/PV 失配位置，因此只把它定性为当前实现不可直接扩 tile，不据此断言 N256 算法本身不可行。项目默认源码和二进制未被覆盖。

补测恰好 256-token、完全**不需要尾端 padding** 的输入后，相对 N128 的输出 NMSE 仍高达 **0.26994**（`n256_stage1_vs_bf16cvt_256.json`），因此可以排除“只是 padding/mask 搞错了”的解释。失配位于 N256 tile 的内部算术或 FP4-PV/softmax 布局；需针对 tile 映射重做实现和独立单 tile 验证，不能仅改模板参数后使用。

诊断源码发现 `add_delta_s` 的 accumulator 初始化循环写死 4 次，只覆盖 N128；在独立 N256 候选上改为 `kBlockN / 32` 后，256-token NMSE 从 **0.26994** 降到 **0.22958**（`n256_deltafix_vs_bf16cvt_256.json`），证明它是**一个**错误来源，但远非全部。当前 N256 至少还有其他 QK/FP4-PV 映射假设不适配，继续逐项调模板已不是低成本优化，且编译时仍有明显 spill。该诊断只改 `/tmp` 候选，默认源码未变。

### N=128 / stage=2 候选：正确但没有可信净收益

另从相同源码独立编译 `M128/N128/stage2`（`/tmp/e130-n128-stage2.isnq1K`）。目标 H3 BF16、非因果变体使用 168 registers/thread，ptxas 报告 **0 spill**；在 129 和 9,482 token 相对已验证 stage3 输出均逐元素相等。22,677-token 的相邻进程同口径测量：stage2 在线/核心 **33.497/26.697 ms**，随后 stage3 **33.848/26.907 ms**。但同轮 Sage2++ 在线/核心也从 **38.577/36.582** 漂到 **39.051/36.879 ms**，BF16 SDPA 从 69.685 漂到 70.549 ms。按每轮 Sage2++ 归一化，stage2 对 stage3 在线约**慢 0.17%**、核心约**慢 0.03%**，小于值得部署的收益。不能把 stage2 更小的绝对时延解读为加速；保留已验证 stage3。原始数据为 `n128_stage2_vs_bf16cvt_{129,9482}.json`、`n128_stage2_long_two_scope_gpu2.json`、`n128_stage3_after_stage2_long_two_scope_gpu2.json`。

### 全局 Q 均值：预处理近似的收益太小

在相同 BF16-CVT 构建上，仅切换 Sage3 原生 `per_block_mean=True/False`，比较真实 H3 QKV 的在线 API（K 恢复排除、正反顺序各一轮 CUDA-event）。22,677 token 的分块/全局均值分别为 **33.261/32.278 ms**，全局约快 **3.05%**；但对同一 BF16 SDPA 输出的 NMSE 从 **0.007039** 升至 **0.008739**（误差增加约 24.2%）。9,482 token 分别 **7.394/7.386 ms**，仅快约 **0.10%**，NMSE 则从 **0.007484** 升至 **0.009014**。因此该近似不能作为不降精度的预处理优化，也不足以解释本机与论文的核心速度差距；不改默认均值策略。它是单个 attention 输出实验，不是完整 DiT 或视频质量结论。见 `global_mean_tradeoff_{long,short}_gpu2.json`，复现入口 `scripts/research/bench_e130_global_mean_tradeoff.py`。

### 真实 H3 QKV 的长度扫描：核心优势稳定，非对齐长度有预处理悬崖

从同一份 22,677-token H3 捕获 QKV 截取连续前缀并整理成各长度的 contiguous HND 输入；Sage3 均为已验证 BF16-CVT＋尾端 mask 版本。CUDA event 每臂 10 次预热、6×12 次正式调用，计时前恢复可修改 K，core-only 预先做相应 QKV 量化。每个长度的 Sage3 融合快路径与其公开 API 输出逐元素相等，Sage2++/Sage3 核心回放与各自在线输出的 NMSE 均不超过 `1e-8`。

| Token | Sage2++ 在线 | Sage3 在线 | Sage3 融合在线 | Sage2++ 核心 | Sage3 核心 | Sage3 核心加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.202 | 0.177 | 0.190 | 0.102 | 0.074 | 1.37× |
| 2,048 | 0.562 | 0.452 | 0.447 | 0.331 | 0.243 | 1.36× |
| 4,096 | 1.634 | 1.351 | 1.359 | 1.198 | 0.890 | 1.35× |
| 8,192 | 5.580 | 4.457 | 4.437 | 4.740 | 3.460 | 1.37× |
| 16,384 | 20.509 | 16.126 | 16.031 | 18.897 | 13.737 | 1.38× |
| 22,677 | 38.659 | 33.507 | 30.886 | 36.486 | 26.607 | 1.37× |

单位均为 ms。前五个长度恰为 128 的倍数，公开 Sage3 不必复制整个 Q/K/V 来补齐；22,677 仅差 43 token 却触发完整张量 padding，公开 API 与融合快路径相差 **2.62 ms**。这说明优化收益主要来自**避开非对齐序列的整张量复制**，不是改变 FP4 核心；对齐长度上的同一路径几乎不加速，1,024 token 还略慢。Sage3 核心相对 Sage2++ 的约 1.35–1.38× 在该长度范围相当稳定，所剩论文比例差距不能再主要归咎于特定 22k 长度或少数预处理调用。输入是同一 H3 样本的长度截断，不代表不同分辨率实际模型分布；绝对时间不能外推到 RTX 5090。见 `length_sweep_gpu2.json`，复现入口 `scripts/research/bench_e130_length_sweep.py`。

专门对 128-token 边界做相邻长度对照，进一步把这一点从长度趋势中隔离出来。16,383／16,384／16,385 token 的公开 Sage3 在线为 **18.296／16.191／18.838 ms**，融合快路径为 **16.434／16.081／16.927 ms**，Sage3 核心为 **13.716／13.799／14.032 ms**，Sage2++ 在线为 **20.497／20.590／20.860 ms**。例如从 16,384 增加仅 1 token，公开 Sage3 在线增加 **2.647 ms**，融合版增加 **0.846 ms**，核心只增加 **0.234 ms**；16,383 比 16,384 少 1 token 时公开 API 反而慢 **2.105 ms**。各臂 batch CV 不超过 0.2%，融合版在三种长度均与公开 Sage3 逐元素相等，故该现象不是内容误差或明显计时噪声。因果判断：公开 API 的 `F.pad` 会在任何非对齐长度复制整份 Q/K/V，形成随模 128 周期出现的在线性能悬崖；融合版省去大部分但非全部边界开销。见 `padding_cliff_gpu2.json`。

### 将精确预处理接入公共 API（默认关闭）

`third_party/sage3_h3_maskfix/sageattn3/api.py` 现支持显式 `sageattn3_blackwell(..., fast_preprocess=True)`。它只在非因果、BF16、相同 Q/K/V 形状、head dim 128、`per_block_mean=True` 下启用，否则报错；默认调用路径与二进制均不变。新路径用 Triton 合并 Q 的 group mean/pad、合并 K 的中心化/pad，V 在原生 packer 中隐式补零，同时保留与公开路径相同的**padded BF16 GEMM 归约顺序**计算 `delta_s`。它不原地修改传入的 K；这是有意的输入副作用差异，应由调用方知晓。

在独立副本中将新 Python API 与已验证 BF16-CVT 二进制组合（核心 SHA-256 `96f8f55c23b3d6f96f9b817eed4386c7546328d3bec42ff21ec0044c2f5ac639`），16,383／16,384／16,385／22,677-token 在线时间分别为公开 **18.320／16.193／18.850／33.775 ms**、新 opt-in **16.500／16.141／17.010／31.200 ms**。另在项目默认 FP32 二进制上，22,677-token 公开 **34.904 ms**、opt-in **32.422 ms**。上述每组 opt-in 输出均与对应公开路径逐元素相等，提升只来自预处理，不代表 FP4 MMA 核心改快。原始记录 `api_fast_validation_gpu2.json`、`api_fast_default_fp32_long_gpu2.json`。

新增 `scripts/research/validate_e130_api_fast.py`：两个编译版本均在 127、128、129、255、256、257、1,024、1,025、9,482、22,677 token 的真实 H3 捕获前缀上通过逐元素相等、有限值、形状和 K 不变检查；优化版本在真实 NHD→非连续 HND 的 9,482-token 输入上也通过。见 `api_fast_{bf16cvt,default_fp32}_10lengths_gpu2.json`、`api_fast_noncontiguous_9482_gpu2.json`。**尚未**完成自由生成回归，因此仍是 opt-in，不把它切为默认部署路径。

后续补了 CUDA Graph 检查：完整 opt-in 预处理＋FP4 核心在 9,482 和 22,677 token 均能捕获并两次重放，输出与 eager 逐元素相等；独立 30 次 CUDA-event 均值分别为 eager→Graph **6.345→6.218 ms**、**30.698→30.618 ms**。Graph 可用，但长序列几乎没有额外加速，不能靠 Graph 解决核心利用率问题。见 `api_fast_graph_{9482,22677}_gpu2.json`。

为 H3 的 NHD 输出组装另加 opt-in `fast_contiguous_output=False`（要求 `fast_preprocess=True`），让裁剪后的 HND view 直接被下游转置/拷贝，避免先做一次 HND `.contiguous()`；默认仍返回 contiguous。真实 9,482-token NHD→HND→NHD 路由，旧 clone 路由 **8.699 ms**，无 clone 公开 API **8.507 ms**，新 API 默认 contiguous **7.366 ms**，新 API 非 contiguous view **7.119 ms**；后者与研究脚本的无额外拷贝路径 **7.110 ms** 一致，所有输出相对旧路由的 NMSE 不超过 `1e-7`。非 contiguous 选项只适合马上装配到其他布局的调用方，不能作为普通 contiguous 返回值使用。见 `api_fast_h3_routed_nocrop_9482_gpu2.json`。

已将该 opt-in 路径接入 `scripts/research/e122_h3_helpers.py` 的独立 `sage3_api_fast` 测试臂。在真实 H3 两条 cached calls（step 0、10）上跑完整 50-block DiT；264/264 linear 权重常驻 GPU，200 个目标 linear 为 BF16，峰值分配约 42.81 GiB。公开 API 快路径的视频和音频输出均与同一 BF16-CVT 二进制的 `sage3_fixed` **逐元素相等**，也与旧研究融合脚本 `sage3_bf16_fused` 的输出逐元素相等。step 0：原 Sage3／新 API／旧融合脚本完整 DiT 直测分别 **6.568／6.469／6.431 s**，attention helper 合计 **1.851／1.667／1.660 s**；step 10：完整 DiT **6.697／6.529／6.518 s**，helper **1.871／1.674／1.672 s**。公开快路径使 helper 约快 9.9–10.5%，完整 DiT 约快 1.5–2.5%；单次测试仍有时钟/调用顺序噪声，不能把它解读为稳定的 2.5% 生成吞吐收益。报告 `h3_resident_api_fast_steps0_10_gpu2.json`，复现入口 `scripts/research/e122_profile_h3_attention_linear.py --resident --steps 0 10 --arms bf16_sdpa sage3_fixed sage3_bf16_fused sage3_api_fast`（需按 `launch_e121_h3_split_generation.sh` 设置本机模型环境变量）。

### 排除 H3 头数不足造成的核心吞吐差距

本机官方 SageAttention checkout 的 `bench/bench_qk_int8_pv_fp8_cuda.py` 默认 batch=4、heads=32；这只是基准脚本默认值，**不能证明论文 Figure 4 一定用同一配置**。为隔离 H3 B1H56 的并行度影响，把真实 H3 捕获前 32 个头复制成 B4H32（不创造新激活分布），在同一 GPU2 上用同一 Sage2++ 和 BF16-CVT Sage3 内核做配对计时。该输入只用于形状性能诊断，不代表真实 batch=4 生成的独立样本。

| Token | B4H32 Sage2++ 在线 | B4H32 Sage3 在线 | Sage3 opt-in 在线 | Sage2++ 核心 | Sage3 核心 | 核心加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.470 | 0.370 | 0.372 | 0.191 | 0.141 | 1.35× |
| 2,048 | 1.196 | 1.041 | 1.106 | 0.692 | 0.504 | 1.37× |
| 4,096 | 3.729 | 3.184 | 3.210 | 2.726 | 1.997 | 1.37× |
| 8,192 | 12.745 | 10.332 | 10.234 | 10.851 | 7.881 | 1.38× |
| 16,384 | 46.869 | 37.048 | 36.890 | 43.293 | 31.429 | 1.38× |
| 22,677 | 88.537 | 76.834 | 71.047 | 83.962 | 61.210 | 1.37× |

单位 ms；各长度核心回放与对应在线 API 输出一致，opt-in 与公开 Sage3 逐元素相等。增至 128 条 batch×head 序列后，核心比值仍在约 1.35–1.38×，与 B1H56 基本重合。见 `shape_b4h32_1024_preflight_gpu2.json`、`shape_b4h32_sweep_gpu2.json`、`shape_b4h32_22677_gpu2.json`。

再对 B4H32、22,677 token 做两臂相同指标的 Nsight Compute 单核心采样：Sage2++ Tensor pipe **75.40%**、32.768 KiB shared/CTA、0 spill；Sage3 BF16-CVT **52.37%**、99.328 KiB shared/CTA、0 spill。Sage3 sleeping stall/issue **0.8573**、short scoreboard **0.7424**，几乎等于 B1H56 的 **0.8571／0.7423**；Sage2++ 的 short scoreboard 为 **0.2543**。这强烈表明 Sage3 在此卡的核心利用率缺口不是单纯 H3 头数没铺满，仍在 CTA 内部的流水线／softmax／同步关键路径。原始 Nsight 报告位于 `/data1/models/svdquant-wjq/research/20261025/E130/sage{2pp,3}_b4h32_long_core_gpu2.ncu-rep`；摘要 `ncu_b4h32_long_core_gpu2.json`。Nsight replay 延迟不能当普通时延；上表只用 CUDA event。GPU2 运行中约 2.46 GHz、56°C，未见明显热降频，但功耗接近当时 350 W 上限，不能据此排除全部时钟影响。

解释 sleeping 计数时还需注意源码的 warp specialization：每个 Sage3 CTA 有 384 threads，其中 128-thread producer warpgroup 负责 TMA/epilogue，256-thread consumer groups 执行 MMA；producer group 中实际只有一 warp 发主循环 TMA、另一 warp 发 epilogue，其余两 warp 没有相同的计算工作。因此 aggregate sleeping stall 不等同于 256 条 MMA consumer threads 全在等待，也不能简单把 52% Tensor pipe 与 Sage2++ 的 75% 差距全部当成可通过 softmax 微调回收的空间。后续若要把等待定位为可优化的关键路径，应按角色/PC 分离 producer 与 consumer 的采样。

### 每 tile 独立行和：数值正常，但核心更慢

在独立源码和编译目录 `/tmp/e130-local-tile-sum.77uFUb` 中，将非首 K/V tile 的 `row_sum = row_sum * scores_scale + sum(exp2(scores))` 改为先用局部 `tile_sum` 汇总本 tile 概率，再加到历史 `row_sum`，尝试缩短每个 `exp2` 到跨 tile 状态的串行依赖。256／9,482-token 的候选相对 BF16-CVT Sage3 基线输出 NMSE 分别为 **1.33e-10／9.46e-9**，有限且非常接近；见 `localtilesum_vs_bf16cvt_{256,9482}.json`。目标 BF16、非因果、M128/N128/stage3 内核仍是 168 registers/thread、0 spill。

真实 H3 B1H56、22,677-token 同 GPU2、相邻独立进程各 25 次预热＋144 次 CUDA-event：候选 Sage3 **在线 34.507 ms、核心 27.730 ms**，随后复测基线 **在线 33.819 ms、核心 26.889 ms**；Sage2++ 同轮在线 **38.573→39.024 ms**、核心 **36.549→36.850 ms**，后测整体设备略慢，反而使候选的 Sage3 回退更可信。候选核心约慢 **3.1%**，在线约慢 **2.0%**，故不合入默认源码或二进制。原始记录 `localtilesum_{long,baseline_after_long}_two_scope_gpu2.json`。这与此前奇偶双累加器和 split-loop 的负结果一致：单纯拆开 softmax 行和依赖不够，且可能增加寄存器活跃区间或调度负担；具体 SASS 原因未证明。

### 将 PV 累加并入已有输出：核心明显加速，但误差放大，未合入

独立候选 `/tmp/e130-pv-accum.Hb88nz` 在每个非首 V tile 的 FP4 PV GEMM 前，先将历史 FP32 输出乘 `scores_scale`，再让 MMA 直接累加到同一输出 fragment，省去独立的 `tOrO` fragment 和逐 tile `rescale_o(old, new)`。非因果 H3 路径之外的 causal 迭代仍保持原逻辑；默认项目源码和二进制均未覆盖。目标 kernel ptxas 仍为 168 registers/thread，但报告 24 bytes spill stores/loads，故不能把性能收益解释为“消除了 spill”。

单次 attention 的 256／9,482-token 候选相对原 BF16-CVT Sage3 NMSE 仅 **1.78e-10／1.11e-8**；22,677-token 相对 BF16 SDPA 的 NMSE 原版 **0.00703871**、候选 **0.00703889**。CUDA-event 同 GPU2 相邻进程、25 次预热＋144 次正式调用：候选 Sage3 在线／核心 **31.781／25.004 ms**，随后复测原版 **33.822／26.908 ms**，即原始时延快约 **6.0%／7.1%**；同轮 Sage2++ 在线 38.594→39.000 ms，说明约 1% 的整体漂移不能解释这项收益。单次速度与误差见 `pv_accum_{long,baseline_after_long}_two_scope_gpu2.json`、`pv_accum_vs_bf16cvt_{256,9482}.json`。

但完整 50-block H3 cached step 10 暴露出不可忽略的累积差异。用两个独立构建、同一个 cached 输入分别保存 BF16 和 Sage3 的视频／音频最终张量，并由 `compare_e130_dit_crossbuild.py` 配对重算：**两个 BF16 teacher 逐元素相等**；候选相对原 Sage3 的视频／音频 NMSE 为 **0.011365／0.023111**。原 Sage3 相对 BF16 为 **0.018687／0.010690**，候选为 **0.019701／0.022709**，尤其音频误差超过翻倍。见 `pv_accum_dit_crossbuild_step10.json`；原始张量位于 data1 的 `research/20261025/E130/dit_crossbuild_{baseline,pv_accum}`。两步常驻 DiT 单独复测见 `h3_resident_pv_accum_steps0_10_gpu2.json`。虽然逐 attention 的重排误差很小，完整网络会放大它；因此**不将该候选作为“无误差提速”合入或用于默认生成**。若未来允许速度／质量权衡，须另做多 prompt 自由生成质量评测，不能仅凭单次 attention NMSE 放行。

### Consumer tile 循环两次展开：逐元素一致，但 spill 抵消调度收益

另从相同 BF16-CVT 快 API 基线复制 `/tmp/e130-consumer-unroll2.jcROgA`，仅将 `mainloop_tma_ws.h` 的非因果主循环 `#pragma unroll 1` 改成 `#pragma unroll 2`，保持 tile 算术与顺序。256／9,482-token 输出与原版**逐元素一致**，见 `unroll2_vs_bf16cvt_{256,9482}.json`。但 ptxas 对目标 M128/N128、BF16、非因果变体报告 168 registers/thread 且约 **176–228 bytes spill stores、236–280 bytes spill loads**，原 BF16-CVT 目标变体为 0 spill。22,677-token 同卡测试在线／核心 **37.751／30.924 ms**，明显慢于相邻已记录原版约 **33.82／26.91 ms**；该轮 Sage2++ 在线／核心 **38.572／36.560 ms**，与原版参考轮相近或更快，因此不是设备整体变慢。见 `unroll2_long_two_scope_gpu2.json`。结论是这个实现的 loop unroll 扩大活跃状态和局部内存流量，**不合入**；保持 `unroll 1`，不能将“更多指令级并行”当作无代价优化。

### 保留的 BF16 delta_s tile GEMM：逐元素一致的在线微优化

针对 opt-in 精确预处理剩余的 correction `delta_s = Q_mean @ K_centered^T`，新增 `sageattn3/api.py` 内的 Triton BF16 tile GEMM，并用 `sageattn3_blackwell(..., fast_preprocess=True, fast_delta_gemm=True)` **显式启用**；未更改默认 API 或项目 `.so`。真实 H3 22,677-token 的输入为 `Q_mean:[1,56,178,128]`、padded K `[1,56,22784,128]`，输出 `[1,56,178,22784]`。预分配输出的同口径 CUDA-event 单算子测量：PyTorch `matmul` **1.270 ms**，Triton M32/N128/K32、4 warps **0.750 ms**，两者 BF16 输出逐元素相等。旧阶段分解中的 `delta_s_matmul_cast=2.538 ms` 同时涉及原 FP32-correction/cast 配方与不同测法，**不可**拿它直接作为当前 BF16-CVT 路径的可省时延。见 `delta_gemm_probe_long_gpu2.json`，复现 `scripts/research/bench_e130_delta_gemm.py`。

API 级验证：B1H56 的 127、128、129、255、256、257、1,024、1,025、9,482、22,677 token，以及 B2H32 的 129、1,025、22,677 token，均与对应 BF16-CVT 原 Sage3 **逐元素相等**，K 输入保持不变；项目默认 FP32-correction 二进制也在 129、9,482、22,677 token 逐元素相等。见 `delta_tiled_api_b1h56_10lengths_gpu2.json`、`delta_tiled_api_b2h32_3lengths_gpu2.json`、`delta_tiled_default_fp32_3lengths_gpu2.json`。这覆盖了当前 H3 目标，但尚不构成对任意头维、因果模式和所有输入分布的保证，故仍保留 opt-in。

同进程 CUDA-event、每臂 10 次预热＋6×12 次测量，BF16-CVT 22,677-token 已有快 API／新 tile GEMM 在线 **30.770／30.248 ms**（约快 1.7%）；9,482-token **6.249／6.174 ms**。项目默认 FP32 二进制的对应 22,677-token **32.416／31.943 ms**，9,482-token **6.627／6.560 ms**。1,024-token BF16-CVT 是 **0.174／0.179 ms**，新路径略慢；2,048、4,096、8,192、16,384 token 则分别节省约 0.012、0.011、0.025、0.186 ms。原始记录 `delta_tiled_{length_sweep,crossover,default_fp32_length_sweep}_gpu2.json`。该候选加速的是预处理，不改变 FP4 MMA 核心吞吐，不能靠它弥合全部 Sage2++／Sage3 论文比值差距。

边界回归发现一项此前未覆盖的反例：在 E098 捕获的**非连续 HND 布局**、127-token 输入上，直接使用 tile GEMM 与 PyTorch BF16 `matmul` 的 correction 有少量舍入差异（correction NMSE `2.03e-8`、约 `0.014%` 元素不同），最终 attention 输出未能逐元素相等；同捕获的 129 和 9,482 token 通过。因此上文的 bitwise 结论只对列出的捕获形状成立，不能扩展为所有短序列/布局。现已把 opt-in 路由改为 padded token 数 **小于 2,048 时回退 PyTorch matmul**：这也避开了 1,024-token tile 路径略慢的问题。修改后，非连续 E098 捕获的 127、128、129、255、256、257、1,024、1,025、2,048、4,096、8,192、9,482 token，以及另一份 E121 捕获的 127、1,025、2,048、22,677 token，均与公开 API 逐元素一致、有限且不改 K；项目默认 FP32-correction 构建在 E098 的 127、1,025、2,048、9,482 token 也通过。见 `delta_tiled_noncontiguous_h3_gpu2.json`、`delta_gemm_probe_noncontig_127_gpu2.json`、`delta_tiled_noncontiguous_guarded_gpu2.json`、`delta_tiled_guarded_long_gpu2.json`、`delta_tiled_guarded_default_fp32_gpu2.json`。该回退不改变先前 9,482/22,677-token 的加速测量口径。

真实 H3 常驻 50-block DiT 的 step 0、10：新路径相对原 Sage3 的视频、音频输出均**逐元素相等**。相对已有 `sage3_api_fast`，attention helper 合计 step 0 **1.664→1.640 s**、step 10 **1.675→1.644 s**；完整 DiT 直测 **6.453→6.451 s**、**6.528→6.506 s**。可确认 helper 的小幅净收益，但不宜根据仅两步、固定调用顺序的差值声称稳定的完整生成吞吐提升。见 `h3_resident_delta_tiled_steps0_10_gpu2.json`；测试臂 `sage3_api_delta_tiled` 在 `scripts/research/e122_h3_helpers.py`。

- Spill 已通过把尾块 mask 前移到 `delta_s` 清零，但 Tensor pipe 仍约 51%；下一步聚焦单 CTA 驻留、softmax/FP4 PV 的依赖和 TMA/同步等待，寻找能实质提高核心吞吐的改进。
- 给 opt-in BF16＋尾块校正路由补更多 head/length/timestep、CUDA Graph、并发和自由生成回归；当前主要覆盖 H3 非因果、head dim 128 的捕获形状及两个 cached DiT steps。

### 概率缩放中的倒数复用：数值严格一致，但性能回退

Nsight SourceCounters 将 `softmax_fused.h` 中逐元素 `acc_conversion_flatten /= AbsMaxP` 列为显著热点，因此另在独立构建 `/tmp/e130-reciprocal.ChXVAz` 测试“每个 scale 计算一次 `1/AbsMaxP`、同组元素乘这个倒数”。它只改 `softmax_fused.h` 的这段缩放，基于 BF16-CVT、修复尾块的构建；项目默认源码与二进制未替换。真实 H3 9,482-token 的输出相对原 BF16-CVT 逐元素相等（NMSE 0），22,677-token 完整输出 SHA-256 也同为 `d612593d07e57c0ceebfe022f2ee8e9bbecdba3b35233dd3199c01c08d031bca`。见 `reciprocal_vs_bf16cvt_9482.json`、`reciprocal_22677_hash.json`。

22,677-token 同 GPU2、相邻运行、各 25 次预热＋144 次 CUDA-event：候选 Sage3 在线／核心 **34.088／27.295 ms**，随后原版 **33.863／26.924 ms**，即候选分别慢约 **0.7%／1.4%**；同时 BF16、Sage2++ 在候选轮分别比随后基线轮更快约 1%，故不能把这项回退解释为整体设备变慢。见 `reciprocal_long_two_scope_gpu2.json`、`reciprocal_baseline_after_long_gpu2.json`。`/=` 的源码热点不等于“重复执行了昂贵的除法”：编译器可能已复用/简化它，显式倒数反而改变指令调度或活跃寄存器。没有 SASS 逐指令对照，具体机制尚未证明。**不合入该候选**，下一步应优先针对 `exp2` 到行和/PV 量化的关键路径，而不是单独重写缩放除法。

### Q 分组中心化＋NVFP4 打包融合：保留为 opt-in

现有 H3 快预处理先用 Triton 按每 128 token 算 Q 均值并写出整块 BF16 `Q-centered`，随后 CUDA 的 NVFP4 打包核再次读取它。K 仍需 BF16 中心化张量用于校正 GEMM，但 Q 的该中间张量只供打包。因此在独立构建 `/tmp/e130-qpack-fused.1B2zVR` 中，新增仅写 Q 分组均值的 Triton kernel 和 `scaled_fp4_quant_group_center` CUDA 特化：它从原 Q 直接读取 16 元素向量、减去该组 BF16 均值并按原规则舍入，然后照旧计算局部 FP8 scale 与 E2M1 code；尾组缺失 token 先作零再减均值，与原 padded Q 一致。**默认 API 与项目现存二进制未改变**；源码以 `sageattn3_blackwell(..., fast_preprocess=True, fast_qpack=True)` 显式启用，并在缺少重新编译的扩展时明确报错。实验构建使用 `E130_BF16_DELTA=1 bash scripts/research/build_e130_sage3_maskbit.sh <独立源码副本>`；项目原二进制 SHA-256 仍为 attention `e4d51cce...`、quant `4ebd3b18...`。

E098 非连续 HND 捕获的 127、128、129、255、256、257、1,024、1,025、9,482 token，与 E121 捕获的 129、1,025、22,677 token：BF16 均值、Q 的 NVFP4 code、局部 FP8 scale 及最终 attention 输出均与原快路径**逐元素相等**，输出有限。见 `qpack_fused_{e098,e121}_validation_gpu2.json`。这覆盖当前 H3 目标及尾块边界，不声称任意布局、head dim 或模型均成立。

同进程交错顺序，每臂两轮、每轮 25 次预热＋144 次 CUDA-event，分开测 Q 中心化＋打包和完整在线 attention：

| H3 token | Q 预处理旧→融合 | 完整在线旧→融合 | 在线净收益 |
| ---: | ---: | ---: | ---: |
| 9,482 | 0.420→0.260 ms | 6.446→6.329 ms | 约 1.84% |
| 22,677 | 1.068→0.781 ms | 31.605→31.287 ms | 约 1.01% |

原始数据 `qpack_fused_{9482,22677}_bench_gpu2.json`，复现脚本 `scripts/research/bench_e130_qpack_fusion.py`。局部节省不能直接全额映射到在线时间，因为缓存、内存分配及其他组件会交互。

进一步用常驻全部 50 个 block、264/264 linear 权重在 GPU 的 H3 DiT cached steps 0 和 10 回归，video/audio 输出对原 Sage3 **逐元素相等**，峰值分配 42.81 GiB。现有快 API→融合 Q 的 attention helper 合计分别为 **1.700→1.691 s**、**1.716→1.699 s**；完整 DiT 单次直测分别为 **6.475→6.499 s**、**6.566→6.561 s**。因此可确认 attention 局部收益，但这两次固定顺序的完整 DiT 计时未能证明稳定端到端加速。见 `h3_resident_qpack_fused_steps0_10_gpu2.json`，测试臂 `sage3_api_qpack_fused`。该融合只改善在线预处理，不会提升约 52% 的 FP4 核心 Tensor pipe 利用率；核心优化仍是主要任务。

两个 opt-in 可同时启用：`fast_qpack=True, fast_delta_gemm=True`。E098 捕获的 127、129、9,482 token 和 E121 捕获的 1,025、22,677 token，组合版最终 attention 输出仍与未优化的快 API 逐元素相等；短序列的 tile GEMM 按上述 `p<2048` 规则回退。见 `qpack_delta_combined_{e098,e121}_validation_gpu2.json`。同进程交错顺序复测，9,482-token 完整在线 **6.462→6.263 ms**（1.032×），22,677-token **31.712→30.871 ms**（1.027×），参照均是相同 BF16-CVT 核心、未开启这两个额外 opt-in 的 `fast_preprocess`；见 `qpack_delta_combined_{9482,22677}_bench_gpu2.json`。

组合版也完成 50-block H3 DiT cached steps 0/10 回归：video/audio 对原 Sage3 逐元素相等，264/264 linear 权重常驻 GPU。现有快 API→组合版的 attention helper 分别 **1.702→1.665 s**、**1.717→1.670 s**；完整 DiT 直测 **6.482→6.477 s**、**6.574→6.536 s**。仅两步且固定运行顺序，不能据此保证自由生成稳定提速或质量；见 `h3_resident_qpack_delta_combined_steps0_10_gpu2.json`。下一步仍需继续优化 FP4 核心的 softmax／PV 调度，而非把这项预处理收益误报成论文级核心加速。

项目默认二进制未重编译/覆盖；源码改动后它在 E098 的 127、129、9,482 token 仍与自身公开 API 逐元素一致，见 `default_api_after_qpack_source_gpu2.json`。因当前默认 `fp4quant_cuda` 尚无新入口，显式 `fast_qpack=True` 会在调用前报出“requires rebuilt fp4quant_cuda extension”，不发生静默回退。

### Nsight 源码角色归因：热点在 consumer softmax，但同步归因仍有限

对已经消除 spill 的 22,677-token Sage3 核心报告做 `cuda,sass` 源码行归因。SASS 总采样 3,262,188 次；2,714,663 次（约 83.2%）可关联到源码行。在这些**可归因**采样中，`softmax_fused.h` 聚集 1,303,784 次，其中 `:121` 的逐元素 `exp2` 约 674,750 次、347,221 次 short-scoreboard stall，`:122` 的行求和约 167,470 次；`mainloop_tma_ws.h` 的 consumer 主循环另约 273,629 次。这个结果支持继续试 softmax 的指数计算/概率量化路径，而不是把新的预处理微优化误认为核心吞吐提升。可复现解析器 `scripts/research/analyze_e130_roles.py`，输出 `role_samples_gpu2.json`。

但不能把上述次数直接换算成独占耗时：源码行有内联重叠，例如源码行汇总的 short-scoreboard 次数略高于全 SASS 总量。SASS opcode 级别的 sleeping 采样主要落在 `SYNCS`（251,195）和 `WARPSYNC`（185,478），另有相当一部分映射至外部 CUTLASS 内联行或未知角色；因此它们**尚不能证明** consumer 在相同时间比例上等待 TMA。详细 PC/opcode 见 `delta_tailmask_sass_role_hotspots.json`。下一项核心候选须同时测寄存器/spill、CUDA-event 核心时间和严格输出一致性；仅凭 Nsight 热点行不能判断优化会成功。

### 从已算出的概率重算分组最大值：负结果

在独立源码副本 `/tmp/e130-softmax-probmax.c9f1fm` 上尝试省去每个 FP4 概率分组的额外 `exp2(max_logit - row_max)`：先照旧算逐元素概率，再对每组的概率做局部 `fmaxf` 和一次 warp shuffle，乘 `1/6` 得到量化 scale。该副本只在 `E130_SOFTMAX_PROBMAX=1` 时启用改动，与 BF16 `delta_s`、尾块 mask 前移、原生 BF16 转换一起编译；项目默认源码/二进制均未改。目标 H3 BF16、非因果、M128/N128/stage3 变体仍为 168 registers/thread、0 spill。

但它不是数值严格替换：256-token 输出相对 BF16-CVT 基线的 NMSE 为 `2.06e-8`（`probmax_vs_bf16cvt_256.json`）。更重要的是，22,677-token 同 GPU2、相邻进程、相同真实 H3 捕获输入和 25 次预热＋144 次 CUDA-event：候选核心/在线 **27.253/34.054 ms**，随后基线 **26.901/33.795 ms**，候选分别慢约 **1.31%/0.77%**。基线轮的 BF16 和 Sage2++ 也分别比候选轮慢约 0.8%/1.0%，故整体设备漂移不能解释候选回退。两轮原始报告为 `probmax_{long,baseline_after_long}_two_scope_gpu2.json`。这个改动用额外分组归约换掉少量指数指令，在当前布局/硬件上得不偿失，且引入数值变化；**不合入**，不再为它做完整 DiT 或自由生成回归。

### 直接用原 K 计算校正项＋只读融合 K 打包：保留为 opt-in

旧的精确快路径先写出整份中心化、padded BF16 K，再分别供 `delta_s` tile GEMM 与 FP4 K pack 读取。新候选在校正 GEMM 的 tile 加载阶段从原 K 读取、减去全局 K 均值并按 BF16 舍入，同时将尾端 padding 置零；随后只读融合 K pack 在寄存器中做相同的中心化与舍入，不写回 BF16 K。Q 和 V 仍用已有融合路径。这**不改变 FP4 attention 核心**，目标是减少预处理的大张量读写。短于 2,048 个 padded token 时保持旧路径，以规避此前发现的 PyTorch/Triton BF16 归约边界差异。

独立原型 `probe_e130_raw_k_delta.py` 在 E121 的 129／22,677-token 输入上，直接校正项与先中心化/pad 再 GEMM **逐元素相等**；22,677-token 的局部 CUDA-event 时间约 1.408→0.800 ms。但若复用原来会原地写回中心化 K 的融合 pack，完整 API 反而没有净收益：9,482-token 6.211→6.301 ms、22,677-token 30.678→30.786 ms。用 K clone 保留不变约定更慢。因此在独立构建里加只读 pack：它对 9,482 和 22,677 token 的 K FP4 code、局部 FP8 scale、最终输出都与旧路径逐元素相等，原 K 也保持不变；在线时间分别 **6.060→5.873 ms**（1.032×）和 **30.091→29.587 ms**（1.017×）。对应 `raw_k_delta_e121_{129,22677}_gpu2.json`、`raw_k_api_{e098_9482,mutable_e098_9482,mutable_e121_22677,readonly_e098_9482,readonly_e121_22677}_gpu2.json`。这些成对时延仅比较新旧预处理与**同一个** Sage3 FP4 核心，不能解释为核心 Tensor pipe 已提高。

实现已保存为项目源码中默认关闭的 `sageattn3_blackwell(..., fast_preprocess=True, fast_qpack=True, fast_delta_gemm=True, fast_raw_k_delta=True)`。它要求重编译的 BF16 `delta_s` 注意力扩展及含 `scaled_fp4_quant_permute_center_readonly` 的量化扩展；缺少时明确报错。项目默认 FP32 二进制未覆盖。由当前项目源码独立重编译出的候选在 E098 非连续 HND 输入的 127–9,482 token 共 12 个长度、E121 B1H56 和 B2H32 的 129／1,025／2,048／22,677 token 均与原 Sage3 逐元素相等、有限且 K 不变；见 `raw_k_project_{e098,e121,b2h32}_validation_gpu2.json`。同一项目构建的在线配对复测：9,482-token **6.024→5.838 ms**（1.032×），22,677-token **29.865→29.366 ms**（1.017×）；见 `raw_k_project_api_{e098_9482,e121_22677}_gpu2.json`。

常驻全部 50 blocks、264/264 linear 权重在 GPU 的 H3 DiT cached step 0／10 回归也已通过：新路径的 video/audio 输出对修复版 Sage3 均**逐元素相等**，峰值分配 42.81 GiB。现有 Q 融合＋tiled delta 快路径→新 raw-K 路径的 attention helper 合计分别 **1.619→1.596 s**、**1.630→1.602 s**；完整 DiT 单次直测分别 **6.405→6.420 s**、**6.496→6.480 s**，方向相反且差值很小，不能据此声称稳定的完整 DiT 或自由生成吞吐提升。原始记录 `h3_resident_raw_k_steps0_10_gpu2.json`，复现脚本 `scripts/research/launch_e130_raw_k_dit.sh`。opt-in 独立构建保存在 `/data1/models/svdquant-wjq/research/20261025/E130/optin_rawk_readonly/`；其中 attention／quant 扩展 SHA-256 分别为 `f8f4ef55b35716b4d5c3b39b3f0b47c5ae27ed4568d72501fb528a42aaeb38a1`／`c13e6b23f813700c80ded904053daf56177d4bc669316c13cd54a2b8ff5d6132`。项目默认二进制仍分别为 `e4d51cce...`／`4ebd3b18...`。两步完整 DiT 只验证已有 cached calls；自由生成、多并发和更广模型形状仍需单独验证，故保持 opt-in。

源码变更后还以**未重新编译**的项目默认 FP32 扩展复测 E098 的 127、129、1,025、9,482 token：公开 API 与原有 `fast_preprocess=True` 输出逐元素相等、有限且不修改 K，见 `raw_k_default_binary_regression_gpu2.json`。归档构建也已直接从 data1 导入，两个扩展加载路径、`delta_s_dtype=bf16`、tail-mask 标志及只读 K 入口均核实。调用新路径须让归档构建的 Python 包和两个扩展一起优先于默认版本加载；仅替换某一个 `.so` 不属于经过验证的组合。

### K/V shared-buffer 提前释放：逐元素一致，但没有稳定加速

在独立源码副本 `/tmp/e130-early-release.RQfgwF` 中，把 consumer 对 K/V TMA shared stage 的 `consumer_release` 从每个 tile 最后一个 MMA 后移到该 MMA 前；此时最后一段 K/V 已复制到寄存器，目的是让 producer 更早开始下一个 tile 的传输。该改动覆盖首 tile、causal middle 和非因果主循环，项目默认源码与二进制未覆盖。目标 H3 BF16 非因果 M128/N128/stage3 编译仍为 168 registers/thread、0 spill。129-token CUDA racecheck 报告 0 hazard；129、9,482、22,677-token 输出与原构建的 BF16 输出逐元素一致。后两种长度的完整输出 SHA-256 分别为 `ca7a64f68277b019f802989abe0b6d9e00d25ded896c58ccf890acf7da36e407`、`d612593d07e57c0ceebfe022f2ee8e9bbecdba3b35233dd3199c01c08d031bca`，见 `early_release_{candidate,baseline}_{9482,22677}_hash.json`。

22,677-token 真实 H3 QKV 做候选→基线及基线→候选两轮反序 CUDA-event 测速，每次 25 warmup、144 正式调用。第一轮 Sage3 核心 26.688→26.988 ms，但同期 Sage2++ 核心 36.578→36.979 ms；第二轮 Sage3 核心 26.950→27.095 ms，Sage2++ 同期 36.920→37.123 ms。把 Sage3 核心时间除以同轮 Sage2++ 核心时间，四次分别是候选 **0.72963**、基线 **0.72982**、基线 **0.72997**、候选 **0.72988**。约 0.02–0.03% 的相对差别远小于原始计时的整卡漂移；在线时间的正反顺序方向也不一致。因此**没有可信的核心收益，不合入该调度候选**，也不为它做更贵的完整 DiT 或长序列 racecheck。原始报告为 `early_release_candidate_long_two_scope_gpu2.json`、`early_release_baseline_after_long_two_scope_gpu2.json`、`early_release_baseline_reverse_long_two_scope_gpu2.json`、`early_release_candidate_reverse_long_two_scope_gpu2.json`。这项负结果说明单纯提前一个 MMA 释放 shared stage 不是当前约 52% Tensor pipe 利用率的主要瓶颈；下一轮仍应针对 consumer softmax/PV 或并驻资源，而非重复这项同步微调。

### 独立 CUDA stream 提前打包 V：逐元素一致，但不缩短在线预处理

另在独立副本 `/tmp/e130-vpack-overlap` 的 Python API 中，为已验证的 raw-K 快路径加实验性 `v_pack_stream` 参数：在 Q 均值、K 均值和校正计算前，先把 V 的 NVFP4 转置打包提交到另一 CUDA stream；在进入原 FP4 attention 核心前用 stream event 等待。输入 V 先等待调用方 stream，输出 buffer 记录使用 stream；这是调度实验，没有改动 FP4 数值算法或默认项目源码/二进制。真实 H3 9,482 和 22,677 token 的完整 API 输出均与串行版逐元素一致、有限。脚本为 `scripts/research/bench_e130_vpack_overlap.py`，两个形状都按串行→并行→并行→串行顺序，以 CUDA events、每轮 25 warmup＋144 calls 测量。

| H3 token | 仅预处理：串行→双流 | 完整在线：串行→双流 |
| ---: | ---: | ---: |
| 9,482 | 1.032→1.051 ms | 5.888→5.896 ms |
| 22,677 | 2.778→2.786 ms | 29.475→29.453 ms |

完整在线长序列的约 0.07% 表面加速远低于可信部署收益；仅预处理反而持平或变慢，不能把 V 打包的单独耗时当作可重叠节省。可能涉及 stream event／资源竞争／底层 launch 调度，现有计时不能区分三者，不作确定归因。原始记录为 `vpack_overlap_{9482,22677}_gpu2.json` 与 `vpack_overlap_preprocess_{9482,22677}_gpu2.json`。**不合入双流 API 候选**；这进一步把优先级压回核心 softmax/PV 和单 CTA 资源布局，而非 Python 层并发微调。

### 用 FP16x2 计算成对 softmax 指数：精度与速度双回退

从相同 BF16-correction／尾块修复基线复制独立源码 `/tmp/e130-half2-softmax`，仅把 `softmax_fused.h` 中每两个 score 的 FP32 `ex2.approx.ftz.f32` 改为先舍入到 FP16，再调用源码已有的 `ex2.approx.f16x2`，最后扩展回 FP32；行和仍按原顺序累加。目的是验证减半指数指令数量是否能缩短 consumer 关键路径。隔离构建成功，但目标 H3 BF16、非因果、M128/N128/stage3 变体由基线 0 spill 变为 ptxas 报告 **52 bytes spill stores、60 bytes spill loads/thread**（168 registers/thread 不变）。

真实 H3 256-token 单次 attention 输出相对原 Sage3 的 NMSE 为 **1.98e-5**（`half2_exp_vs_baseline_256.json`），已明显超过此前近似变体的数值差异，故不进入完整 DiT。22,677-token 同卡、相邻进程各 25 warmup＋144 CUDA-event：候选 Sage3 核心／在线 **27.960／34.745 ms**，随后原版 **27.009／33.976 ms**；同期 Sage2++ 核心由 36.580 漂到 36.987 ms，设备整体在候选轮更快。按同轮 Sage2++ 核心归一化，候选仍约慢 **4.7%**。见 `half2_exp_long_two_scope_gpu2.json`、`half2_exp_baseline_after_long_two_scope_gpu2.json`。因此 FP16x2 并没有在此实现中转换为吞吐优势，反而引入 spill 与量化前的额外转换；**不合入，不做自由生成**。这不否定所有 vectorized softmax，但排除当前“逐对 FP32→FP16→指数→FP32”的局部替换。

### M64 小 Q tile 的校正行索引：确认一处 bug，但仍无法形成有效内核

先前直接把 M128/N128/stage3 改成 M64/N128/stage1 时，256-token 无 padding 输入相对原版 NMSE 高达 0.612。检查发现 `delta_s` 的 Q 均值每 **128 token** 一行，而 consumer 的 `gDS` 原先直接用 64-token `m_block` 作行索引；从第二个 M64 tile 开始读错校正行，后续甚至可能越界。独立副本 `/tmp/e130-m64-dsfix` 将索引改为 `m_block / (128 / kBlockM)`，保留其余 M64 算法。实测 256-token NMSE 降到 **0.59486**，但首个 64-token 区段仍是原来的 **0.60975**，第二个区段由 **0.64341→0.61004**。因此分组索引确是一个错误来源，却**不能解释首 tile 的大误差**；还存在与 QK/PV MMA fragment 或概率布局有关的独立问题，不能把该候选视为数值正确。见 `m64_dsfix_vs_baseline_256.json` 与旧 `m64s1_vs_original_n256.json`。

目标 BF16、非因果 M64/N128/stage1 编译使用 **232 registers/thread、316 bytes spill stores、348 bytes spill loads/thread**；256 threads/CTA 的寄存器用量约 59,392，仍不能在每 SM 65,536-register 的 PRO 5000 上并驻两个 CTA。由于既未达正确性，又未达到双驻留的资源目标，**不做吞吐或 DiT 评测，不合入**。要真正走 M64/M32 路线，需要重做 fragment 布局与校正广播并先过单 tile 逐分量验证；简单改 tile 模板参数并不能带来可信的 Sage3 加速。

M64 的后续受控输入进一步定位了严重误差的性质。保持真实 Q/K、将 V 全部设为 1 时，M64 与 M128 的 256-token 输出 NMSE 为 **0.50077**；M64 的 56 heads 中前 28 个输出约为 1、后 28 个**逐元素为 0**。仅截取原后 28 heads、重新编号为 0–27，仍是前 14 个约为 1、后 14 个全 0（NMSE **0.49807**）；再把 Q/K 都缩到 0.1 倍，后 14 个仍全 0。Q/K 全 0 时两版输出一致。这些结果排除“某些原始 head 激活尺度特殊”和单纯极端 logits 下溢，但 Q/K 全 0 的一致性本身不能证明所有输出位置都经过有效计算。捕获脚本 `capture_e130_sage3_output.py` 新增 `--probe`、`--head-start`、`--heads`、`--qk-scale`；批量复现入口 `launch_e130_m64_probes.sh`，数值 JSON 为 `m64_dsfix_{zero_qk,ones_v,zero_qk_ones_v}_256.json`、`m64_dsfix_ones_v_last28_256.json`。

CUDA memcheck 对 M64 的这一短例报 **0 个越界错误**。initcheck 报告未初始化读取，但同一受控输入的**正确 M128 基线也触发同类报告**，因此不能把它当成 M64 漏写的证据；两份日志为 `results/logs/e130_m64_{memcheck,initcheck}_gpu2.log`、`results/logs/e130_m128_initcheck_gpu2.log`。为直接检验写回，在独立 M64 C++ API 将 `out` 预填 42 后重编：异常的后半 heads 仍逐元素为 0，而不是保留 42，说明这些位置至少被某条写入路径覆盖，不能把零简单解释为 `torch.empty` 的旧内容。另只在该隔离版 Python 预处理中给 `delta_s` 预留双倍行跨度，后半 heads 仍为零；“head stride 重叠”尚未被证实为根因。当前更精确的结论是 **M64 与非零 QK 相互作用时产生 head-count 相关的零输出，来源仍需对 QK→softmax→PV 分阶段内核插桩**；在找到根因前不运行其性能基准。

逐 head 均值、严格全零 head 索引、哨兵覆盖检查另存 `m64_controlled_head_summary.json`，由 `scripts/research/summarize_e130_m64_head_probe.py` 从捕获张量重算并校验预期模式。默认 Sage3 二进制 SHA-256 仍为 `e4d51cce159a2537afba74c55096800777112f30809ed8f8062d1bc735b5cc7d`。

### 跳过运行最大值未变化时的 softmax rescale 指数：微小正收益，保留 opt-in

在非首个 K tile，Sage3 每次计算 `exp2((old_max-new_max)*scale)` 来重缩放历史 softmax 状态；当运行最大值未变化时，它严格等于 1。独立候选只在这一条件下返回 1，其余情况保持原表达式。9,482/22,677-token 真实 H3 B1H56D128 非因果输入的完整输出 SHA-256 均与原构建相同，分别为 `4631837679c39accfca905c57589924be8a9c6c4b11c63744bea4259dba15ac7` 和 `d612593d07e57c0ceebfe022f2ee8e9bbecdba3b35233dd3199c01c08d031bca`。见 `scorescale_skip_{9482,22677}_gpu2.json`。

两组独立进程 AB/BA 配对、各臂内再做正反顺序 CUDA-event 计时，并以同轮 Sage2++ 核心/在线时间控制设备漂移：9,482-token 的 Sage3 **核心**归一化加速分别为 `1.00857×/1.00942×`，**在线**为 `1.00832×/1.00840×`；22,677-token 核心为 `1.00389×/1.00561×`，在线为 `1.00542×/1.00470×`。原始记录为 `scorescale_{skip,baseline}_two_scope_gpu2.json` 与 `scorescale_{skip,baseline}_reverse_gpu2.json`。这只是约 0.4–0.9% 的改进，不能当作论文级核心提速或稳定完整 DiT 收益。

代价是目标 BF16 非因果特化从原来的 0 spill 变为 ptxas 报告约 72-byte spill stores/80-byte spill loads/thread，168 registers/thread 不变；需要进一步 Nsight/更广形状和完整 DiT 回归。另试将 C++ 条件改为 PTX 谓词，spill 降至约 68/76 bytes，但同轮归一化的核心与在线时间反而比 C++ 版慢约 1.5–2%；逐位输出仍相同，故**淘汰谓词版**。见 `scorescale_predicated_lengths_gpu2.json`、`scorescale_predicated_two_scope_gpu2.json` 与 `scorescale_branch_after_predicated_gpu2.json`。

项目源码将 C++ 版保存为默认关闭的 `E130_SKIP_UNCHANGED_MAX_EXP=1` 编译选项；必须在独立目录构建，勿覆盖项目默认二进制。推荐仍与 `E130_BF16_DELTA=1 E130_DELTA_TAIL_MASK=1 E130_BF16_CVT=1` 配套。当前只是 H3 目标形状上的 opt-in 候选，不作为通用 Sage3 默认路径。

最终已从项目源码独立重编上述四开关组合，9,482/22,677-token 输出哈希继续与原修复版一致；129、257、1,025、2,048、8,192、9,482、22,677 token 的公开/组合 API 逐元素一致、有限且不修改 K，见 `scorescale_final_lengths_gpu2.json` 和 `scorescale_final_7shape_validation_gpu2.json`。可加载的最小源码＋扩展归档在 `/data1/models/svdquant-wjq/research/20261025/E130/optin_scorescale_skip/`，直接从归档加载的 9,482-token 输出哈希再次一致（`scorescale_archived_9482_gpu2.json`）。attention/quant 扩展 SHA-256 分别为 `fd6ca9e6b3ccc734e6e1b85c7d3b88e54f1a45564c4d90b6728260b96ceed670` 和 `1c0427f0bd82265107a81f4b0c9833a6db96f35e9eeee75621d524287c1f8d8e`；默认扩展未覆盖。归档只含源码和约 17 MB 扩展，不保留大体积中间编译文件。
