# E115 — Triton FP4 与官方 SageAttention3 数值对齐

输入为 E098 保存的真实 MiniMax-H3 block-25 Q/K/V；只做 attention 输出比较，
不重新生成视频。官方参照为安装的 SageAttention3 Blackwell FP4 core；
原生混合版参照为 E114 的 CUTLASS/CUDA mixed core。所有 NMSE 均按配对输出
`sum((candidate-reference)^2)/sum(reference^2)` 计算。

## 找到的差异和修正

1. E113 Triton 的 E2M1 精确中点判断方向与硬件
   `cvt.rn.satfinite.e2m1x2.f32` 相反。修正七个中点比较后，
   1024-token、4-head 的 Triton 全 FP4 vs 官方 NMSE 从
   `0.00275778` 降至 `0.00003529`。同步修正 MixAttention 的 PyTorch
   量化参考和 BF16-matmul Triton 仿真代码。1024-token Q/K/V 官方 pack
   与修正后的 Triton probe 在 E2M1 code 和 E4M3 scale 上全部逐字节一致：
   各有 524,288 个数据元素，三者 mismatch 均为零。
2. 官方 Sage3 的在线 softmax 从最后一个 128-key tile 向前遍历，
   E113 Triton 原先正向遍历。改为反向后，1024-token 全 FP4 vs 官方
   NMSE 降至 `6.20e-9`；混合版 vs E114 原生混合版降至 `5.32e-9`。
3. 官方 Sage3 的尾块 padding mask 按 K 的物理排列列号判断。
   K pack 存在 within-32 重排，故 54-token 例子会错误屏蔽 4 个有效 key，
   并放入 4 个 padding key。对同一个短序列，仅把 PyTorch 模拟的 mask
   改成 `k_permutation(128)[54:]` 后，其 BF16 输出与官方逐元素相同。
   Triton 添加显式 `match_sage3_padding_mask=True` 选项用于匹配官方；
   默认仍采用逻辑上正确的 mask，不隐式复制这个边界行为。

## 最终配对结果

| 输入 | Triton 全 FP4 vs 官方 FP4：默认 mask | 官方兼容 mask | Triton 混合 vs 原生混合：官方兼容 mask |
| --- | ---: | ---: | ---: |
| 1024 tokens × 4 heads | `6.20e-9` | `6.20e-9` | `5.32e-9` |
| 9482 tokens × 56 heads | `1.74e-5` | `1.15e-8` | `1.13e-8` |
| 54 tokens × 4 heads | `0.006468` | `0` | `0` |

全主序列在官方兼容口径下，两版的混合精度增量 NMSE 分别为
`2.6168e-5`（原生）和 `2.6168e-5`（Triton），增量方向余弦相似度
`0.99982`。因此旧 E114 的 `0.00147649` 跨实现差异不能用来评价
混合策略；它主要来自当时的 E2M1 舍入和在线遍历顺序不一致。

同一个 9482-token × 56-head 输入，以 BF16 SDPA 为分母重新核算的
**全 FP4** 输出 NMSE：历史 E113 Triton `0.00741296`；修正后 Triton
默认 mask `0.00748442`；修正后 Triton 官方兼容 mask `0.00750234`；
官方 Sage3 `0.00750226`。配对数值见 `fp4_vs_bf16.json`。

## 工件与限制

- `fp4_alignment_reverse_kv.json`：最终逐 case 结果，包括 Q/K/V code/scale
  校验、正确 mask 与官方兼容 mask 的双口径。此前正向遍历结果保留在
  `fp4_alignment.json`，供 A/B 复核。
- `native_vs_pytorch_group128_reverse_full.json`：反向遍历 Triton 与
  FP32 PyTorch group-128 参考实现的回归。
- `triton_emulator_validation.json`：修正中点后，另一份 BF16-matmul
  Triton 仿真与其 PyTorch 参考的回归。
- `fp4_vs_bf16.json`：旧版、新版和官方 Sage3 的同输入 BF16 SDPA NMSE。
- 复现主审计：`python scripts/research/diagnose_e115_sage3_fp4_alignment.py --gpu 2`
  （需 E098 Sage3 环境和 Q/K/V capture）。

本结果针对一个真实 H3 Q/K/V 捕获；不证明任意 shape/输入都逐元素一致，
也不是生成画质或速度结果。E113/E114 历史 JSON 未覆盖，其旧数值需与
E115 的修正前后版本区分。官方 mask 的物理/逻辑错位是本地版本的代码
与数值对照结果；本研究没有修改官方 Sage3。
