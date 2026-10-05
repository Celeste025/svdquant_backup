# E011：实际 native W4A4 投影之后的官方 mean-sim dense guard

2026-10-02，**NOT_READY / 未执行，停止实现扩展**。仅完成 CPU 静态审计和官方 utils 文件导入；没有启动 GPU、安装 consumer 或新增 probe 脚本。下文保留的是未执行设计草案，不能视为完成实验或冻结运行合同。

停止原因：本地没有已验证可用的 SM120 sparse consumer；官方 fused mean-sim 尾块存在 masked-zero 归一化 `0/0 → NaN → False`，而真实 H3 两种长度都有 tail。输出 bool 有限不足以证明 guard 有效。目前不能可靠回答全长度工作量或质量问题，因此不值得为此启动一套 12-case 重捕获。**这不是机制阴性，也不是测得的 GPU domain 失败。** 父 agent 已明确不写/修新 kernel，不做 GPU smoke。完整有效块 guard 只能作为未来的限定设计选择。

## 固定输入和对照

- H3 同一 Comfy-pruned checkpoint；prompt 1/20、step 0/19、block 0/24/48，12 case，全是原 PTQ 校准数据。保持 packed 模态顺序、原 cu_seqlens、QK norm 和 RoPE。
- 一份 E009 已验证 resident BF16 模型。4 个完整 BF16 teacher 调用捕获三个 block 输入；每个 block 的 BF16 重放输出须逐元素等于 teacher 对应 endpoint。
- T0 为原 BF16 qkv linear，T1 只替换该 block 的 qkv linear 为 E009 导出 NativeH3Linear：legacy NVFP4 codes/scales/global、BF16 smooth/LR、zero-SF compatibility fast adapter。其余三个 linear/所有上下游均 BF16。E006 的 RNE 投影不复用；它的代码仅提供真实 attention 入口捕获位置，且没有持久化实际 QKV。
- 记录 T0/T1 post-norm/RoPE QKV SHA 和真实 shape；主路由只需 Q/K。变换严格沿模型原函数，不手写 norm/RoPE。

## 官方实现与固定路由配置

`thu-ml/SpargeAttn@ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` 的 `spas_sage_attn/utils.py`，SHA256 `eecf9109d12bbf34e853e327c4fa8f1b60e66dca74ec1a6ca355236bbbd06325`。按文件载入；不安装 C++ consumer、不写新 kernel、不修改该文件。

调用 `get_block_map_meansim_fuse_quant`，同一 BF16 selector、BLKQ=128、BLKK=64、is_causal=False、attention_sink=False、return_lut=False，`km=k.mean(-2,keepdim=True)` 用全部真实有效 token。配置固定：

1. 主配置：官方 CDF 入口默认 `simthreshd1=0.6, cdfthreshd=0.98, topk=None`。
2. 简单对照：官方推荐 top-k 入口默认 `simthreshd1=-0.1, cdfthreshd=None, topk=0.5`。

不扫阈值或换配置找现象。router 的 INT8 输出不是稀疏选择的输入；不能把其混为额外 projection 量化。

## 前置域门槛

真实 main sequence 长度 p1=22384、p20=22432，均有部分 Q/K block。独立 padding 序列 16/32 token 小于官方 core>=128 范围，保持 dense 并单独报告，不能并入 main。

先短随机整块/相同余数 smoke，再对真实全长度 Q/K 检查官方 pooled/scales 有限、有效 token 范数非零、两次重复 guard/mask 完全一致。不能仅检查 bool guard 的 finite：官方 similarity 内部 NaN 经 `>threshold` 会成为 False，必须显式检查其零范数/未定义 masked-load 来源。

已知静态风险：fused K tail 的 masked 行被置零，随后 normalize 形成 0/0；Q masked load 没有 `other`。**任何真实长度下的未定义值/NaN→guard、非有限 pool/scale 或不稳定 map，状态必须为 `domain_stop`，不解释为 W4A4 引起的 guard 变化。** 本轮不改 kernel、不丢真实 tail、不人工 tail-dense 补丁。可保存完整有效块的限定诊断，但无合法全长度 map 就不能通过下列 10% work 门槛。

## 统计和 STOP

每头分别记录 Q dense-row 和 K dense-column 数量、T0→T1 双向翻转、强制 dense 行列的去重并集、最终 active blocks。按真实 Q/K block 长度计算 token-pair 加权工作 `W=sum(mask_ij * nq_i * nk_j)`，并同时保留 tile 数；主汇总先对各 case 的所有 heads 求和，再对 12 case 汇总，不能挑一个 head 或只用 mask Jaccard 声称成本。

只有合法全长度 CDF 路由完成全部 12 case，才计算 `sum(W_T1)/sum(W_T0)-1`。实际额外工作低于 10%、符号抵消，或域检查失败，停止该投入方向；10% 仅是投入门槛，不是显著性。top-k 对照原样报告，不允许主配置失败后改以该配置作为通过标准。

即使达到 10%，本实验也只能报告实际块数变化；不能称额外块无必要。只有下一阶段在固定 T1 QKV/logits 下交换 guard 位并测精度、再与普通阈值重调比较，才讨论因果或方法。当前不运行 sparse consumer、attention 性能、自由生成或视频评价。

## 运行与工件

目标新脚本 `scripts/research/probe_h3_sparse_router.py`；结果 JSON 和独立日志在 `results/research/` 与 `results/logs/`，失败保留。全 12 case、4 teacher，单 GPU **600 秒硬 timeout**；先检查 GPU 空闲并使用命名 tmux。GPU 由 root 后续分配，未获启动指令前仅 CPU 检查。

结果绑定本计划、新脚本、官方 utils、E009 manifest/3 个 qkv payload/相关 frozen native/fast/compat/loader、模型与 4 个输入、torch/Triton 版本；保存每 case 状态。全量 post-RoPE QKV 不默认写盘；只保存失败输入和足够小的 mask/guard，以防约 21.5 GiB 的无必要双臂 QKV 缓存。
