# E039 后的有限接口与贡献判断

2026-10-03。只读本地 FlashInfer/Nunchaku；定向重读下面三篇 primary sources。无 GPU、安装或新实验代码。

**支持完成已冻结的 E040；不建议据当前结果直接扩成完整 SP 系统。** E039 是有效工程信号：同一实际 O、原 native SVD 投影下，pair-max wall 中位数 BF16-return / FP8-return / row-parallel / FP4-side 为 6.34766 / 11.09036 / 8.76230 / 4.89735 ms；后三者对 BF16-return 的 valid NMSE 分别为 .00233519 / 7.55112e-6 / 5.93241e-8。row/side 保存的 down 完全相同，故两者输出误差差距不能归为 side 用了更准确的 down。这是单个投影边界对既定 native 配方的结果，不是 BF16 模型或视频质量结论。[独立汇总](../../results/research/E039/independent_summary.json)

## 已有 API 与公平复用边界

- **FlashInfer 已有外部 packet＋down 的 SM120 consumer。** [gemm_svdquant.py:751](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/gemm/gemm_svdquant.py:751) 的 `mm_nvfp4_svdquant(a,b,a_sf,b_sf,alpha,d,l1,backend='cute-dsl')` 直接接受 row-major E2M1 codes、128×4 swizzled scales、BF16 `d[M,32]`；检查见同文件 :667，SM120 fused 分支见 :256。E039 的表示形状可接，**本轮未在 GPU 调用**。但它计算 `alpha*(main+d*l1ᵀ)`，要求 BF16 `l1=B/alpha`，并将 correction 加入 FP32 epilogue；这改变原 BF16 main/up/add 舍入，动态 activation global 还使该 B 变换不能一概算作离线成本。公平复核应让 BF16-return 与 side 同用此 consumer，并计入动态准备，不能只优化候选。
- **完整 FI producer 不是原配方的无损替换。** 同文件 [nvfp4_quantize_smooth:915](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/gemm/gemm_svdquant.py:915) 消除 BF16 smooth 中间态；[svdquant_linear:986](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/gemm/gemm_svdquant.py:986) 将 smooth 折进 down 权重，再独立 BF16 GEMM。其格式虽相容，舍入不等于 `BF16(O/smooth)` 后的原 down。
- **本地 Nunchaku 是旧 INT4 实现。** [ops.h:9](/home/wjq/workspace/svdquant-exp/nunchaku/csrc/ops.h:9) 的 `gemm_w4a4` 已接受外部 act/scales/lora_act；[融合 producer](/home/wjq/workspace/svdquant-exp/src/Linear.cpp:321) 已联合 quantize/down。但是版本 0.0.2beta1 使用 integer s4/u4 MMA、group64，down 是 FP32 专用 packed 布局、M 补齐256；并非当前 E2M1/group16/E4M3 packet 的直接 consumer。[布局](/home/wjq/workspace/svdquant-exp/src/kernels/zgemm/gemm_w4a4.cuh:746)、[构建 SM86/89](/home/wjq/workspace/svdquant-exp/setup.py:56)。有限本地查找未定位更新的 Nunchaku 源码，不外推新版不支持。
- **E039 的 FP8 不能称成熟融合上界。** 本机 FI 的 [per_token_group_quant_8bit:596](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/quantization/fp8_quantization.py:596) 是 cuTile 分组量化；同文件 :172 的 MXFP8 固定 group32，:551 的还原是 host API；[quantized_all_reduce:478](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/comm/quantized_allreduce.py:478) 是 symmetric-memory 分组 FP8 AllReduce。它们均不是当前逐 source/destination 单 scalar、标准 A2A、GPU 还原的直接同合同替换。所见 [per_tensor_fp8_quantize:173](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/trace/templates/_init_helpers.py:173) 仍为测试用 PyTorch 链。不能据 11.09 ms 排除优化 FP8；也无需为本轮把这些不同算法全部接成网格。

## 最近覆盖与未覆盖

1. [SVDQuant §4.3 / Figure 6](https://arxiv.org/html/2411.05007v3)：quantize/down 共享原高精度输入，main/up 共享输出，联合融合减少搬运。**packet＋低秩状态作为消费表示，以及保留未量化输入给 LR 的必要性，本来就是其设计。** 所读论文与本地入口没有给出该表示跨 head→token SP 的传输实现。
2. [LongLive-2.0 Appendix D，式11](https://arxiv.org/html/2605.18739v2#A4)：明确交换集合为 Q/K/V，使用预 attention NVFP4 A2A，复用压缩 KV。所读段落未给出 O 回传的既有 SVD-down 侧状态；不能把这个宽泛近邻自动当完整同合同覆盖。
3. [CompactFusion §3.3–3.4、Appendix C](https://arxiv.org/html/2507.17511v1)：拟合通信 activation/residual 的低秩表示、量化因子、误差反馈及压缩/通信重叠。E039 的 down 来自固定模型 A，而非在线拟合重建激活；这是真实接口区别，但不是“低位＋小矩阵通信”的新概念。

**本次未核实完整同合同跨 SP 实现，不等于新颖性证明。** 若最终只剩“把成熟 packet＋down ABI 放进 A2A，并按既有列分片恒等式合并 down”，我判断不足独立 paper unit：实测收益有用，但新的表示、信息来源与消费原理都尚未建立。不能用 SM120、视频或物理 scale 重组单独填补这个缺口，也不应因所有成分已有就否认工程结果。

**最小下一决定就是 E040 已定的 same-packet/main、原 down / source-partial down / decoded-X down 控制，暂不追加性能臂。** 它能判断廉价普通 FP4 回传是否已足够，或既有 LR 需要保留原输入信息；若后者成立，也首先是在通信边界确认 SVDQuant 已知的信息合同，并不会自动变成新方法。完成后若没有超出成熟 ABI 调度的具体新约束/机制，停止将此候选作为独立论文主线，保留代码和可测收益供后续部署复用。若以后确需扩大性能证据，先做同 consumer 的公平库替换，而非直接投入完整多卡生成；当前不建议安排这一步、不把尚未执行的优化成本当下界。
