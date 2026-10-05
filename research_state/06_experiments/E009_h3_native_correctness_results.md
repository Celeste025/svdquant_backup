# E009 H3 完整原生正确性基线

2026-10-02。**已完成：同一真实输入上的完整 resident BF16、旧公式 QDQ、原生 NVFP4，以及 slow/fast 打包全图一致性闭环。** 本轮只有原 PTQ 校准集 p1 step0，不能推断生成质量、泛化或速度。

## 固定合同与已通过门槛

- 原本地 `MiniMaxH3DiTComfyPruned`；50 主块、200 目标线性层，另 8 个 token-refiner 线性层保持 BF16。真实 22,400 packed tokens，不裁 token；原 QK norm、RoPE、位置/分区保持，attention 使用实际 BF16 torch SDPA。
- 474 个磁盘包装模块完成常驻迁移；全部 **50 block 与 video/audio endpoint SHA** 相同。`rope.inv_freq` FP32[16] 从原 checkpoint 显式物化，`adaln_t_table` 保持 FP32，其余非包装 tensor dtype 保留；resident 前向零磁盘 load。
- 导出 **200/200 legacy 权重**按原 GPU BF16 算术重建，并独立回解对旧 QDQ 逐元素 exact；旧 rank32 state 未重新校准。原位转换前后及完整前向后非目标 tensor 哈希不变。
- 四个预指定真实 block0/N512 输入，独立 `nn.Linear + common hooks` 与 legacy 公式完整输出 SHA 相同。相同 packet 的 native 主支 NMSE：qkv `4.39283e-6`、out `5.13403e-6`、fc1 `7.07859e-6`、fc2 `8.25129e-6`，均低于 `1e-4`；有实际 SM120 E2M1 kernel 证据。
- 完整 slow-native 与 fast-native 各 **200 次实际 FP4 GEMM、102 次实际 BF16 SDPA、0 次磁盘 load**，全部输出有限。fast 的 **50 block 与两个 endpoint SHA 全等于 slow**。200 个 fast flags 在 collector 退出后验证，invalid=0。

旧 QDQ 全图 reference 使用原 `common.nvfp4_qdq(x/s)`、已验证导出的 BF16 回解权重和 corrected 高精度低秩分支；不是另驻留的一整份 BF16 QDQ 模型，也不是把已量化激活送入低秩分支的旧 bug。

## 完整 endpoint 误差

NMSE 的分母为列中所写 reference 的能量。以下是固定输入的张量诊断，不是视频质量指标。

| 比较 | Video NMSE | Audio NMSE |
|---|---:|---:|
| 旧公式 QDQ / BF16 | 2.648866% | 6.878320% |
| Native / BF16 | 2.333776% | 4.168893% |
| Native / 旧公式 QDQ | 0.371760% | 1.342494% |
| Native fast / Native slow | **逐字节一致** | **逐字节一致** |

这一个输入上 native 更接近 BF16，**不解释为方法改进**。相同量化 packet 的 FP4 GEMM 与 BF16 QDQ GEMM并非相同算术；完整图中的后续动态量化可以使差异传播。本轮没有因果实验区分这些因素，也没有要求原生图与 QDQ 图逐字节一致。

部分 hidden-state NMSE 如下；valid 只含 video/audio/text，排除 16 个 pad/unassigned token。完整 50 行和各模态能量在 raw JSON 中。

| Block | QDQ / BF16 | Native / BF16 | Native / QDQ |
|---|---:|---:|---:|
| 0 | 0.104493% | 0.105273% | 0.024772% |
| 1 | 0.246607% | 0.247721% | 0.065147% |
| 9 | 0.186863% | 0.187000% | 0.060443% |
| 24 | 0.693903% | 0.657575% | 0.449761% |
| 49 | 5.473291% | 5.341501% | 3.356110% |

## 保留的失败与显式兼容域扩展

初次 resident 运行在未物化的 `rope.inv_freq` 处停止；报告、日志、源码与计划快照保留。修复读取原 checkpoint 后，完整迁移 exact 门槛通过。

原 strict packer 随后在 **block1.fc2** 遇到非零输入组的 E4M3 scale 下溢到 0 而停止；原失败报告仍为 `failed_stop`，未覆盖或改写为成功。新最小诊断只重放前两个 block，先确认 block0 SHA 复现原失败轨迹，再保存真实 raw 与 x_s。

该输入 `[22400,14336]` 有 **95 个 SF0 组、1520 个值，均在 audio**。旧 common QDQ 本来就把这些值量化为零。E005 canonical-zero 独立回解与 common 的 **321,126,400 个元素数值 exact、err2=0**；87,840,962 个零符号差包含普通 round-to-zero，不能全部归因于这 95 组；其他 byte 差=0。95 组的原输入能量为 `2.94696e-4`，占该层 audio 输入能量 `6.88708e-6`、valid 全体能量 `1.49312e-12`。实际 fullshape native 主支 NMSE=`4.56099e-6` 且有限。

因此新独立 adapter 明确允许并记录这种数值等价域，冻结的原 packer/export 保持不动。先在保存的失败输入上验证 compat fast vs E005 slow 的 codes/global/含 padding SF **逐 byte 差=0**，flags checked1/affected1/invalid0，再恢复完整图。完整 slow 统计与 fast call-level flags 一致：仅 index7=`blocks.1.mlp.fc2` 受影响，95 组。fast flags 本身记录受影响调用数，不能称作组数。

恢复运行继承原报告已通过的 BF16、resident、legacy reference 及全部 source 哈希，没有重跑三个 reference forward；只重跑新的 slow native 与同进程 fast SHA 回放。没有据诊断耗时或每层采样显存作性能主张。

## 可复核产物与后续

- 主完成证据：`results/research/E009_h3_native_resume.json`。
- 保留的 strict 失败及三个完整参考：`results/research/E009_h3_full.json`；其 SHA 为 `2caef6a433089207b62bb8ef2d508f06f975af8dc70d109f18f1ab2171fc82e0`。
- 真实 SF0 诊断：`results/research/E009_h3_sf0_domain_v2.json`。早先 import 启动失败也保留在不带 v2 的文件中。
- 权重导出与四类 fastpack parity：`E009_h3_export.json`、`E009_fastpack_legacy_four.json`，均在 `results/research/`。
- root 独立汇总：`research_state/06_experiments/results/E009_native_summary.json`；本报告未覆盖该文件。
- 原始参考、endpoint、全长 xs：`/data1/models/svdquant-wjq/research/20261002/E009/`。同一输入的两个 native endpoint SHA：video `bdc0e53288ef5dba47548f64afb90f8c30b8a34b1f7f879b30390ec9cab1e463`，audio `ac64a78752e26cb1d5b1f46bd265673c93c91275b74eda9c655c405c978f5c88`。

下一步由独立 runner 做 resident BF16 / 常驻旧 QDQ / nativefast 三臂性能测量，首个输出必须匹配上述已保存参考。此处不新增校准样本、生成网格或质量评估。
