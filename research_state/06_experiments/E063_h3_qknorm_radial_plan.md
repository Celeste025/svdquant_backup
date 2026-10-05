# E063 — QK 径向收益错配的必要条件筛查

2026-10-04，exploration，执行前冻结。H3 主模型与 CFG1 合同不变。它是架构假设的最小判别，不是方法或 attention 功能验证；E062 阴性不提供本假设的阳性动机。

## 动机、假设、决策

H3 在 Q/K 投影之后使用逐 head RMSNorm。纯径向、正比例、忽略 epsilon 的扰动在归一化后消失，因此原始输出 MSE 的改善不一定代表后续表示改善。**待验证问题**是现有低秩分解相较同尺度无低秩量化，是否将主要重构收益放在这类方向上。

不能把数学上的径向不变性认领为创新，也不能把混合残余的径向部分当作严格零空间：`qhat=(1+alpha)q+e_perp` 中，alpha 会改变归一化后的切向误差比例，负缩放还会翻号。实际 learned gamma、epsilon 与 BF16 norm 必须直接重放。

- H1 必要条件：同一预选层的两个状态都出现明确 raw QK 改善，改善主要为径向，而实际 norm 后改善很小。才值得另做 joint QK/teacher V 的真实 attention 诊断。
- H0a：raw 收益本身不足；优先解释为当前 legacy 配方弱，不建立几何机制。
- H0b：收益主要是切向，或 norm 后同样改善；停止本次径向预算错配解释。
- 输入/布局/数值控制失败：保留产物，报告不可判定，不作为机制阴性。

近邻 [QuantMLA](https://arxiv.org/html/2609.36760v1)、[HeadQ](https://arxiv.org/html/2605.03562v1)、[SlimDiff](https://arxiv.org/html/2509.21498v1)已覆盖功能校准、可见空间补偿或联合 QK 低秩；见[定向筛选](../02_problems/h3_qknorm_screen_20261004.md)。不能将泛化 norm 后损失/联合 QK loss 作为新贡献。这里尚无方法 claim。

## 最小有效设置

固定 E014 teacher 状态 p30/s5、p36/s14，预选 blocks 0/24/49。两次完整 BF16 forward 的 actual 输入/raw/velocity 必须逐字节复现历史。hook 捕获三处完整 qkv_proj 输入、teacher 输出、实际 q_norm/k_norm 输入与输出、RoPE 参数。

**布局控制：实际为 MiniMaxH3DiTComfyPruned。** 核对其 attention.forward 绑定 `minimax_h3_dit_comfy._comfy_attention_forward`，使用 `[rows,3,heads,head_dim]`。必须与真实 q_norm/k_norm 输入逐字节验证，不能采用基础类 `[rows,heads,3,head_dim]`。所有源文件 hash 绑定。

每个完整输入先算原 BF16 `xs=x/s`，量化一次；两臂严格共用同一个 activation packet（packed codes、block scales、global）。

1. **rank0_same_smooth**：从原 BF16 W 计算 BF16(W*s)，重新按旧配方打包；通过 PlainNativeH3Linear.main_from_packet 执行真实 W4A4，无 LR。新权重解码与独立旧 QDQ 的 BF16 roundtrip 必须一致。
2. **legacy_rank32**：原 E009 packed residual + 原 BF16 两次 LR GEMM 的加法顺序，使用同一 activation packet。原导出文件与 W 身份核对，不重做初始化。

两臂权重 global 可以不同，必须记录；这个比较改变整个分解，不把全部差异归于 LR 支路。直接删除旧 residual 的 LR 不能作为 rank0。

在完整输入上获得 QKV，再沿实际 img_pos_info 选择固定 512 个 video 行：`floor(i*(Nvideo-1)/511), i=0..511`。Q/K 的真实 norm 与 RoPE 在完整形状上重放再采样，teacher norm 对实际 hook 精确。Q/K/V 都保存采样值，V 仅旁证；不增加 attention 调用。完整输入仅暂存 CPU，不写全量张量。

## 主读出与固定门槛

以 teacher 的每行每 head Q、K 为共同方向，FP64 分解两臂误差 `e=radial+tangent`。分别对 Q/K、每 head 及合并 QK 报告 SSE；验证正交分解和带符号收益恒等式：

`G_raw=E0_raw−E32_raw=G_rad+G_tan`。

几何投影采用 FP64 `dot(e,q)/dot(q,q)`，不把 RMSNorm epsilon 加入几何投影分母。q 恰为零时 radial=0、tangent=e 并单列计数；近零但非零仍按原值计算。epsilon 只用于实际 norm 与诊断。`G_rad/G_raw` 可为负或大于 1，不截断，也不称独立收益份额。

另报告实际 BF16 norm 后 Q/K 合并 SSE、RoPE 后 SSE、V raw SSE、teacher 近零范数/epsilon 与两臂径向翻号计数。norm/RoPE SSE 仅表示误差，不等于 logits、attention 输出或视频质量。

同一预选层在两个状态同时满足以下三项，才允许提出下一次联合 attention 诊断（本次不执行）：

1. `G_raw/E0_raw >= 0.10`；
2. `G_rad/G_raw >= 0.75`；
3. `(E0_norm−E32_norm)/E0_norm <= 0.10`。

分母为零不晋级；负收益照实报告。10%/75% 是预设研究投入门槛，不是统计显著性或科学不可能性定理。六格全部报告，不换层/提示找阳性。

判别范围只限“主要 raw 收益在径向且 norm 收益有限”这个具体候选。若不过，停止该解释；若过，仍需固定 teacher V、实际完整 key 集的联合 Q/K attention 读出和匹配相关结构的普通控制，才能谈功能错配。再后才有同 rank 的容量重分配及普通 Q/K/V 重加权/block reconstruction 强对照。没有预先假设正结果，也不把本次局部 proxy 当成功方法。

## 执行及产物

最多 2 个完整 BF16 DiT forward、12 个局部 native qkv GEMM；6 次完整输入激活量化。无新完整 native DiT、无新增 attention 反事实、无训练/视频/MJ。单张空闲 SM120；GPU 60 GiB、张量文件 2 GiB、绝对 900 秒截止，启动前重查。需要先 CPU check（0 CUDA）和代码审查，再唯一 GPU 执行。

原 v1 源/结果一经执行不覆盖，路径/工程失败如需修复另版本且保留失败；不因观察超时另开任务。原脚本和 E062 不修改。复用 E062 启动时将已有 recovered site-packages 追加 sys.path 末尾供 av，不安装或替换原 native 包。

小报告 `results/research/E063/`；张量 `/data1/models/svdquant-wjq/research/20261004/E063/`；新源码 `scripts/research/probe_h3_qknorm_radial.py`。保存采样 QKV、actual norm/RoPE 读出、gamma/epsilon、选择索引、共享 packet signature/weight global 与输入身份，供独立 CPU/NumPy 复算。不把一般误差几何或基线工程计为新贡献。
