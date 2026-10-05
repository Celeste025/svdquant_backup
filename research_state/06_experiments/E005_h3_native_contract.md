# E005：真实H3线性层的NVFP4量化契约审计

状态：complete（2026-10-02）；baseline-correctness，不是优化/novelty实验。

- 数据：同E003的prompt1 step0/19完整packed raw calls。显式torch SDPA，BF16上游；只运行block0并用pre-hooks捕获qkv_proj与mlp.fc2输入，不跑50blocks、不生成视频。
- 权重：plain BF16 W；旧rank32/g10 state产生的residual `W*s - B@A`，相应输入`X/s`。低秩branch使用未QDQ的`X/s`，与corrected E003一致。所有方法固定同一输入/权重/FP32 tensor global/E4M3 scale；旧state不重校准。
- 比较：历史common.nvfp4_qdq；独立software pack/unpack的legacy tie rule；明确E2M1 nearest-even；分别以同一legacy/RNE codes执行真实F.scaled_mm两级global。软件BF16 QDQ GEMM与FP32 unpack/dequant reference分开，禁止将其差异归因成量化方法收益。
- 统计：精确FP32归一化midpoints与near-midpoints分开；有符号ties及changed-code频率；E4M3非零subnormal、zero scale分组统计，并单列all-zero原始groups/非零group underflow；排除swizzle padding。全量及video/audio/text/pad输出NMSE和native-vs-QDQ偏差。原生接口不包含CUDA cvt量化，不能声称已验证hardware conversion。
- 单独unpack uint8 nibble解码，用软件重建对照common QDQ；common不修改。至少一次profiler捕获SM120 blockscaled FP4 kernel。RNE和legacy共享rounded scales，z不提前转BF16。
- 三项数值误差分开：方法相对原BF16 linear；ties造成的软件输出差；固定codes的native相对BF16 QDQ/FP32数学reference差。SVD输出表包含相同高精度branch，native main-only差另报。
- 容错：legacy pack/unpack与历史QDQ必须逐元素一致（zero sign不区别）；分区唯一且互斥；非finite输出停止并保存failed；native API/kernel失败只记录，不用BF16假fallback。出现非零group scale-underflow时标记风险，不通过epsilon静默“修复”。GPU峰值接近60GiB则逐层释放或减chunk，不能截token替代完整shape。
- 止损：2step×2linear×2weight recipe完成即停止；不增加prompt/block，不做吞吐扫参。native全block是否值得复验由误差尺度及ties比例决定，不预设QDQ已失效。
- 预计：GPU5单卡，block0/临时tensor预计<30GiB，chunk128；5–10 GPU分钟保守预算（多数为多路张量量化与校验，不是模型吞吐测量）。预pack GEMM无端到端加速结论。
- 产物：`scripts/research/probe_h3_native_contract.py`、`results/research/E005_h3_native_contract.json`、`results/logs/research_E005_h3_native_contract.log`；tmux `research-E005-h3-native-contract`。输入/state/source完整SHA，基础模型使用E003已算完整SHA并校验文件大小/mtime与记录；若元数据不足以确认则重新hash。GPU5启动前检查。

## 执行与校验

完成预注册的 2 steps × 2 linears × 2 weight recipes，共 8 组，未增加样本。GPU5 峰值分配 11.34 GiB，tmux 已退出。H3 recovered 环境为 torch 2.11.0+cu128；显式 torch SDPA，实际审计到 8 次入口调用且 Q/K/V 全部 BF16。完整 packed 22400 tokens，无截断。

全部 activation/weight 的独立 legacy pack→unpack 与历史 `nvfp4_qdq` **逐元素完全一致**。真实 GEMM profiler 捕获：

```text
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

这证明 packed operands 进入 SM120 FP4 GEMM；软件产生 FP4 codes，未验证 CUDA FP4 conversion 指令的舍入实现。profile 同时包含 scale/swizzle/copy 等操作，未做任何速度结论。脚本对基础模型、state、raw inputs、依赖源码全部重新计算完整 SHA256，记录大小和 mtime；运行结果直接保存在 JSON。

## 舍入与 scale 统计

以下 ties 为给定 FP32 global/block-scale division 配方中精确 E2M1 midpoint 的元素比例；near-midpoints 单独保存在 JSON。A/W 分别指输入和权重，SVD 为旧 state 生成的 residual，未重新校准。SF 为每行每 16 K 元素一个 E4M3 scale，不含 swizzle padding。

| step | linear | recipe | A ties % | W ties % | A subnormal SF % | 非零 A group 的 zero SF 数 |
|---|---|---|---:|---:|---:|---:|
| 0 | qkv_proj | plain | 0.392971 | 0.000000 | 0.000000 | 0 |
| 0 | qkv_proj | SVD residual | 0.182729 | 0.165575 | 0.000000 | 0 |
| 0 | fc2 | plain | 0.106273 | 0.372282 | 1.522655 | 0 |
| 0 | fc2 | SVD residual | 0.050626 | 0.117663 | 0.218162 | 0 |
| 19 | qkv_proj | plain | 0.000000 | 0.000000 | 0.000000 | 0 |
| 19 | qkv_proj | SVD residual | 0.000000 | 0.165575 | 0.000000 | 0 |
| 19 | fc2 | plain | 0.164620 | 0.372282 | 1.824019 | 234 |
| 19 | fc2 | SVD residual | 0.142980 | 0.117663 | 1.350925 | 0 |

全部 weight SF 与 qkv activation SF 没有 subnormal/zero。fc2 step19 plain 的 234 个非零 input groups 下溢，占 20,070,400 groups 的 0.001166%；各组不是全零输入。实现没有通过 epsilon 隐藏该现象，但尚无证据表明这些极小 groups 造成显著质量损失，也没有测量其模态归属。subnormal 不等于 zero；不能从存在 subnormal 推断量化失效或 batch 耦合灾难。

## 输出差异

相同 RNE codes 下，native 相对 BF16 operand-QDQ 主支输出的全量 NMSE 为 **2.43e-6～6.35e-6**；相对独立 nibble 解码的 FP32 GEMM/global 数学 reference 为 **2.71e-6～2.77e-6**。后者直接比较 BF16 native output 与 FP32 reference，包含正常输出舍入，不能要求逐元素 bitwise equality。前者还包含 BF16 重建 operands 与 native block/global 应用次序的差异。

RNE 与 legacy 软件主支差异的全量 NMSE 为 0～1.041e-4，不能称代码或输出完全相同。但比较相对原 BF16 linear 的误差时，所有 8 组 × all/video/audio/text 的 NMSE 相对变化最大为 **0.1621%（RNE QDQ）/0.1746%（native RNE）**。这是误差大小变化，不是输出逐元素偏差的上界。

以下 video NMSE 相对相同 BF16 upstream 的原 BF16 linear；SVD 输出含同一高精度低秩 branch，不是只评估 residual GEMM。

| step | linear | recipe | legacy QDQ | RNE QDQ | native RNE |
|---|---|---|---:|---:|---:|
| 0 | qkv_proj | plain | 0.00369638 | 0.00369677 | 0.00369650 |
| 0 | qkv_proj | SVD residual + branch | 0.00305687 | 0.00305501 | 0.00305158 |
| 0 | fc2 | plain | 0.01006247 | 0.01006306 | 0.01006039 |
| 0 | fc2 | SVD residual + branch | 0.03843375 | 0.03842783 | 0.03842062 |
| 19 | qkv_proj | plain | 0.00234192 | 0.00234192 | 0.00233921 |
| 19 | qkv_proj | SVD residual + branch | 0.00271155 | 0.00270821 | 0.00271409 |
| 19 | fc2 | plain | 0.00269118 | 0.00269156 | 0.00269484 |
| 19 | fc2 | SVD residual + branch | 0.01570766 | 0.01569868 | 0.01569348 |

## 对 E001/E003 的影响与下一步边界

在已测真实线性层上，legacy tie rule 与明确 RNE 的差别存在，但没有改变 plain/SVD 的 video NMSE 排序。尤其 fc2 的 SVD video 误差大幅变坏在原生 GEMM 中仍存在：step0 为 0.0384206 vs plain 0.0100604，step19 为 0.0156935 vs plain 0.00269484。因此，没有证据把已有模态误差冲突主要解释为 fake-QDQ 舍入失真。

这只支持保留 E001/E003 作为探索性诊断；不能直接证明 E003 的最终 video denoiser 误差上升 13.7%/8.5% 在 native 完整 block 和后续 49 blocks 后仍保持。E005 使用 BF16 upstream，只覆盖 qkv/fc2 两个线性层，不包含 block 内多个量化层的串联。E001 的 SageAttention 默认后端标签问题独立存在，不能由 E005 修复；E003 在显式 BF16 torch SDPA 下独立重算 teacher/donors。

**需要 native 全 block 复验，才能声称该结论适用于真实原生 W4A4。** 最短下一步是在同一 p1 step0/19、同一 teacher 和 state 上，为 block0 四个目标 linears 实现明确 RNE QDQ 与 packed native donor，比较 donor 局部误差及同样的 BF16 continuation pulses；先做 2 cases，不扩 prompt，不生成视频。该工作尚未启动。当前结果没有触发“现有 E003 全部作废并重跑”的止损条件，也不构成 novel insight、生成质量或端到端速度证据。

结果完整性检查：8 个预期组合唯一；各 modality 的 err2/ref2/numel 求和与 all 一致；全部数值有限；所有 legacy independent roundtrip NMSE=0；小型源码/input provenance SHA 与当前文件一致。`results/research/E005_artifact_sha256.json` 记录脚本、报告、JSON、日志的最终 SHA。
