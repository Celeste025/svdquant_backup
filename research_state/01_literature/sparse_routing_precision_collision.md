# N2 收敛核查：W4A4 投影与稀疏路由

2026-10-02；仅 CPU/网络静态审计，未运行 GPU、安装后端或登记 claim。承接 [系统先验 N2](native_systems_frontier.md)。结论：**泛化的“量化使 top-k 不稳定 / 因而要高精度 selector、误差反馈或 SP 重平衡”已高度拥挤，不升级为研究题目。只留一个有明确实现位置的统计门槛：真实 W4A4 QKV 是否显著改变 SpargeAttn 的整行/整列 dense guard。尚不能称额外保留的块“无必要”。**

## 已覆盖的执行合同

| 最近邻 / 核实范围 | 已覆盖什么；不能据此声称什么 |
|---|---|
| [QuantSparse v4](https://arxiv.org/html/2509.23681v4)，§3.2–3.4、Algorithm 1；[作者仓库](https://github.com/wlfeng0509/QuantSparse) | 明确建模量化 X/W 导致的 Q/K 投影误差与稀疏 attention 联合偏差；通过多尺度 attention 蒸馏、跨步缓存残差修正。不能说前人只量化 attention 操作数。论文未给同一 QKV 下仅交换离散 selector 的因果分解；所查仓库只有 README/图片等，没有可审计实现。这是证据范围，不是 novelty 证明。 |
| [SLA2](https://arxiv.org/html/2602.12675v1)，§4–6 | 当前 Q/K mean pooling 后经可学习投影选 top-k，训练路由与低位 sparse attention，最终硬路由微调。低位前向描述是 FP16 Q/K/V→SageAttention2++ 风格量化；不能等同于原生 NVFP4 W4A4 投影，但“联合训练 router 与量化 attention”已覆盖。 |
| [FVAttn](https://arxiv.org/html/2607.16190v1)，§3；[SparSP](https://arxiv.org/html/2609.32197v1)，§4–5 | 前者使用实际生成的 mask 做 head/rank 负载修复；后者先交换当前 pooled Q/K 摘要、得到需求，再传对应 KV，几何 placement 固定但需求动态。路由变化不会自动成为“过期调度”；普通工作量重平衡、按实际需求通信已经是强基线。 |
| [Runtime-Certified Quantized Attention](https://arxiv.org/html/2605.20868v1)，§4.4、§6.1、§9.9 | 已明确讨论量化排序翻转，检查已选块 FP16/INT8 排序及未选块上界，并触发精度回退。它是 KV-cache / LLM 场景，参考未量化 KV，不覆盖前置 W4A4 projection；泛化的“认证 selector + fallback”仍直接碰撞。未复核证明或性能；论文自报现实缓存下慢于 dense，作者仓库链接本次返回 404。 |

## 唯一更具体的入口：相似度保护导致整行/整列保留

官方 [SpargeAttn utils.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L371)，pin `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`：

- `get_block_map_meansim_fuse_quant(q,k,km,...,return_lut=False)` 返回 mask 与 INT8 Q/K packet。pool/guard 来自当前浮点 Q/K，**不是** INT8 decode：均值 FP32 reduction 后存回输入 dtype；guard 把归一化向量转 FP16 做块内平均余弦相似度，与 `simthreshd1` 比较。
- L407–410 将低相似度 K block 对应整列、Q block 对应整行强制置 1，再并入 CDF/top-k 选块。因此即使名义 top-k 不变，实际工作量也会变。guard 是启发式保护，不是已证明的误差证书。
- [core.py](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py#L40) 默认先算 `km=k.mean(-2,keepdim=True)`；SM90 用块 64×128，其他已有分支用 128×64。独立非 fused 入口没有 km 参数，不能直接替代合同。
- utils 本身仅依赖 torch/einops/Triton，可按文件载入避开包初始化中的 CUDA 扩展；统计 mask 可复用原 kernel，无需实现 sparse attention kernel。**这不证明本机已有可运行的 sparse consumer，尤其不能从 dense FlashInfer/Sage 的可用性推断。** systems 静态审计：6 个环境未安装 Sparge/SLA；recovered 环境有 torch 2.11、Triton 3.6、einops 0.8.2，按文件 CPU import 已成功，但没有运行 GPU。审计源文件 SHA256 `eecf9109d12bbf34e853e327c4fa8f1b60e66dca74ec1a6ca355236bbbd06325`。

剩余未证实的问题仅是：**投影量化是否使保护规则发生足够大的工作量变化，并且这种变化与实际近似需求脱节，简单重调阈值仍无法消除？** 这里只确认代码有这种放大入口，未观察到现象。selector 提高到 FP32 不能恢复已经丢失的投影信息。

## 最小决定性流程与 STOP

**先统计，不测 sparse GPU 性能。** 少量预指定 teacher 输入，分别做 BF16 / E009 legacy native W4A4 投影（使用当前 zero-SF adapter），均通过模型真实 norm/RoPE，保持 cu_seqlens 分段。E006 RNE 是另一合同，不混用；E006 未持久化 QKV 本体，只有 shape/hash，不能假定可直接重放。两边调用同一 pin 的官方 router、相同 smoothing/块形状/阈值/selector dtype；不引入另一套 quantization recipe。先做官方全长度输出 finite/重复稳定 gate：真实长度 22384/22432 有尾块，官方 masked load、零范数路径存在 NaN 风险。失败则 STOP，不把数值异常解释成 guard signal。完整有效块的 guard 可独立诊断，但截尾 mask 不能冒充全长工作量，且 km 必须用全部有效 token；若未来明确采用尾块统一 dense 的适配合同，须双方相同、预注册，并说明不是原函数全长原样执行。

记录每头 Q/K guard 位的双向变化、强制保留行列的去重并集、最终 active blocks 与 rank/head 负载；可按 SparSP 的实际 KV 需求并集计字节。用真实块数/预指定 placement 下最大 rank 工作量评估，不以 Jaccard、guard flip 数或一项极端 head 作为成本结论。**若预指定样本汇总后有效工作变化不足 10%，或只有翻转而增减抵消，立即 STOP；不启动 sparse 性能/视频。** 不在看到结果后换 threshold 找现象；10% 是投入门槛，不是显著性标准。

**只有过门槛才做一次固定 QKV 干预。** 固定 W4A4 的 Q/K/V、pooled logits、smoothing 和 attention 算术，只把其 guard 位替换为 BF16 同输入 guard，按原规则重建 mask；与 W4A4 自己的 guard 比较。BF16 guard 只是干预，不是 oracle。分别对当前 QKV 的 dense attention 与 BF16 dense teacher 报误差，确认节省的块是否确实没有提供相应精度收益。不要同时换整套 logits、QKV 或 kernel。

**最强普通对照 / 最终 kill：** 在独立校准样本上重调 `simthreshd1` 与 CDF/top-k（包含关闭 guard 的端点），并比较实际块数匹配的误差曲线；当前 QKV 的高精度 pooled selector、FVAttn 式实际负载平衡也是基线。若普通阈值重调即可重获同等收益、额外 dense 块确实必要、或收益只来自预算不同，kill。只有 held-out 上仍有 ≥10% 有效工作节省、相同预声明误差容限，且能指出普通调参失败的结构原因，才值得重新查新；在此之前不构造方法、不声称可发表。

## 最终执行决定（E011：not_ready，未执行）

root 决定暂不运行 E011、不继续扩展 12-case probe。H3 真实尾块的 `0/0 → guard=False` 可能是保守保留行为，不能立即称为 bug；当前完整长度消费者及语义未验证，本地也没有已验证的 SM120 sparse consumer。缺少合法完整路径时，继续路由统计无法决定全长度成本/质量收益。此状态不是机制阴性结果或新发现；成熟 H3 sparse backend 到位后再复议。当前推进的是独立 E010 生成验证。
