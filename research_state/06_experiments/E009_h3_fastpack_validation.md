# E009 H3 legacy activation packer 验证

状态：2026-10-02，独立 Triton packer 已完成 synthetic、四种全长形状随机输入、BF16 upstream 的真实 block0 qkv/fc2，以及完整 legacy 轨迹 p1 / step0 的 block0 四类 linear 激活验证。这里只给出 packing correctness；没有完整模型、速度或质量结论。

## 源码与接口

- `scripts/research/h3_nvfp4_fastpack.py`：3 kernels，partial amax → FP32 global → group16 E4M3/E2M1 编码和 SF swizzle。未修改 E005/E007 冻结源。
- helper SHA256：`f5a22a9e3f23d7903403017575e1a63c1cb42e04b0230256a5808902bbc198b7`。
- `pack_legacy_h3(x_s)`：接收 contiguous CUDA BF16 `[..., K]`、K 整除16。返回 `FastPacked(codes, scales, global_scale, domain_flags)`，分别为 uint8 `[M,K/2]`、含 padding 的一维 swizzled E4M3、FP32 `[1]` 和 int32 `[2]`。低层函数不做 host sync，调用方必须检查 flags。
- `pack_activation_fast(x_s, *, chunk_rows=128)`：与 `NativeH3Linear` 一致的 `PackedNVFP4` adapter，K 整除32；默认立即检查 flags。`collect_fastpack_checks()` 可把多个调用的检查合并到 context exit；计时结束的同步应放在 scope 内，计时区间之后再读取 flags。不得忽略非零 flag。

输入已经过旧 BF16 smoothing；packer 不重复 smooth，不计算 LR，不 materialize activation QDQ，不改变 GEMM。

## 精确合同

FP32 global 为 `(amax.clamp_min(1e-12) / 2688).clamp_min(1e-12)`；group ideal 为 `(amax_group / 6).clamp_min(1e-12)`；除 global 后 clamp FP32 tiny，再做 E4M3FN RNE。E2M1 midpoint 取较小 signed 值：正数阈值 `>`，负数阈值 `>=`。常数除法按照 PyTorch CUDA 的 FP32 reciprocal multiply，tensor/tensor 除法使用 `tl.div_rn`，关闭 FMA fusion。

字节 oracle 是冻结 E005 `quantize_pack()['legacy']`，其函数 AST 原样编译执行，避免导入历史 pipeline；另从 common 原样执行 `nvfp4_qdq` 比较 decode 的数值。测试记录三个冻结源 SHA256 和 helper/test SHA256。

零的表示必须分清：

- 真实 `+0/-0` 输入均编码 nibble0。
- 负的非零值舍入为零时，E005 保留 nibble8；common signed15 QDQ 则给 `+0`。两者数值相等，解码 BF16 的 sign bit 可以不同。
- SF0 的全零组，E005 canonical nibble0；common 原始 `0/0` 后 argmin 路径可能得到 `-0`。这也是数值相等而 decode 字节不同；测试 JSON 的通用 zero-sign note 主要描述上一种情况，本节补充此边界。
- 非零输入组的 SF0 明确 flag / 拒绝；没有用 epsilon 把它静默改成另一种量化配方。
- 本 packer 与 E005 本身没有任何 canonical-zero 字节偏差；对比的是完整 codes/global/SF（包括 swizzle padding）。

## 已完成检查

| 输入 | 与 E005 codes/global/SF bytes | decode 与 common | flags |
|---|---|---|---|
| E2M1 正负 exact ties、真 signed zero、负小数舍入零 | 全相等 | 数值全相等 | 0,0 |
| E4M3 midpoint，含非零 subnormal SF | 全相等 | 数值全相等 | 0,0 |
| 全零（含负零） | 全相等 | 数值全相等 | 0,0 |
| SF0 全零组 + subnormal SF + 列尾部 `[129,48]` | 全相等 | 数值全相等 | 0,0 |
| BF16 最小 normal / subnormal | 全相等 | 数值全相等 | 0,0 |
| 随机尾部 `[257,48]`、batch `[3,17,32]` | 全相等 | 数值全相等 | 0,0 |
| 随机 full qkv / out / fc1 / fc2：M22400，K5376/7168/5376/14336 | 全相等 | 数值全相等 | 0,0 |
| 真实 p1 step0 qkv `[22400,5376]` | 全相等 | 数值全相等 | 0,0 |
| 真实 p1 step0 fc2 `[22400,14336]` | 全相等 | 数值全相等 | 0,0 |
| 完整 legacy 轨迹 p1 step0 block0 qkv/out/fc1/fc2，M22400 | 全相等 | 数值全相等 | 0,0 |

拒绝测试：nonzero-SF0、E4M3 zero/min-subnormal midpoint、NaN、+Inf、-Inf，全部正确抛错；deferred flag 检查及异常后的 context 恢复均通过。

第一次 attempt 将 E4M3 zero/min-subnormal midpoint 错放进有效域 case，触发了预期的拒绝，三个 byte mismatch 仍均为0。保留失败 JSON/log；修正测试归类后 attempt2 完成。helper 自第一次实现后未修改。

## 证据与范围

- `results/research/E009_fastpack_parity_attempt1.json`、`results/logs/E009_fastpack_parity_attempt1.log`：保留的测试归类失败。
- `results/research/E009_fastpack_parity_attempt2.json`：11 有效用例 + 7 拒绝/上下文检查；全部通过。
- `results/research/E009_fastpack_real_qkv.json`、`results/research/E009_fastpack_real_fc2.json`：两个真实输入全部通过。
- `results/research/E009_fastpack_legacy_four.json`、`results/logs/E009_fastpack_legacy_four.log`：audit 捕获的完整 legacy 轨迹四类激活全部通过；输入 `/data1/models/svdquant-wjq/research/20261002/E009/pack_inputs.pt`，schema `inputs{name:tensor}`。四层均没有 zero SF；fc2 有41,973个非零 subnormal SF，仍全字节一致。helper SHA 与 test SHA (`46721598d83114e83a6fa084c2b615b470953ad3765623ea576f008f085a01bd`) 保持不变。
- 真实输入：`/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/fused_inputs/blocks.0.attn.qkv_proj.pt` 与 `blocks.0.mlp.fc2.pt`，字段 `x_s`。由 systems 在完整 BF16 block0 upstream 捕获，施加旧 smoothing；不改变 prompt/step、不补充新样本。
- GPU0 只运行 packing 验证；每次实际使用前检查空闲，并以命名 tmux 启动、日志落 `results/logs/`。GPU3 曾见他人瞬时占用，未使用；GPU1/5 留给 audit/systems。

这建立了当前输入域上的 fast/slow execution parity。有限测试不等于对任意 BF16 输入的形式证明；后续完整模型仍须保留 domain flags，并检查实际被调用的 fast packer、分支顺序和 native kernel。

完整 E009 在另一个实际激活（block1 fc2）触发 SF0 guard 的事件不被本报告的 block0 通过结果覆盖。严格 helper 按预先规定停止，不等于证明该 H3 数值不能用零重建表示；真实 SF0 的 common/E005 语义由 audit 独立核查，域扩展 adapter 由 systems 独立实现。本验证没有修改 guard 或扩展许可域。
