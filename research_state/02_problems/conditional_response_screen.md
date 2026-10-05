# 条件响应差异：一个可否证的观察量，尚不是研究方向

2026-10-02，CPU-only 概念/源码审计；未运行实验、未登记 claim、未作新颖性判断。**结论：值得区分“响应衰减”与普通误差，但不能仅换成差分 NMSE 就重启 CFG 或模态加权路线。当前没有弱条件响应被 W4A4 抹掉的证据。**

**它测什么，为什么不等于单样本 MSE。** 固定同一 `x=(video_latents,audio_latents)`、两个真实 timestep、attention 后端，令 `D_B=F_B(x,t,c₁)−F_B(x,t,c₀)`，`D_Q=F_Q(x,t,c₁)−F_Q(x,t,c₀)`，F 为 scheduler 消费的 velocity。记 `e_i=F_Q(c_i)−F_B(c_i)`，则差分误差 `d=e₁−e₀`，`||d||²=||e₁||²+||e₀||²−2⟨e₁,e₀⟩`。对 `m=(e₁+e₀)/2`，普通重构损失已有 `||e₁||²+||e₀||²=2||m||²+0.5||d||²`，不能说它完全忽略响应。单个总 MSE 值不决定两项的分配；只测 d 又会完全漏掉公共漂移。该结构与 cond/uncond CFG gap 相同，也能直接写成配对蒸馏；换损失权重本身不是研究方向。

不能把大差分 NMSE 称为“塌缩”。同时报告绝对 `||D_B||²,||D_Q||²`、两端边际误差，以及 `g=⟨D_Q,D_B⟩/||D_B||²` 和 `r²=||D_Q−gD_B||²/||D_B||²`。差分 NMSE 正好是 `(g−1)²+r²`：低 g 且低响应范数才支持衰减；g≈1、r² 大只是额外偏转/噪声。video/audio 分开，避免高能模态掩盖另一模态。分母很小或 teacher 没有可靠响应时停止解释。

**已有证据不支持跳步推论。** [E002](../06_experiments/results/E002_summary.json) 否决的是 CFG6 下跨时间相干误差：conditional 去偏相邻 cosine 0.603/0.654，CFG 后仅 0.102/0.184；它没有测语义最小对照，也不证明有条件响应塌缩。[E003](../06_experiments/results/E003_summary.json) 的大 text 局部能量不等于最终影响；[E004](../06_experiments/E004_modality_reweight_results.md) 普通 refit 的局部 video 改善 39–47% 未变成稳定 endpoint 收益，video4 相对 pooled 仅 −6.74%，且 E003 的取舍方向跨 prompt 反转。因此不能预设“少量 text/audio token 是被忽视的控制信号”，更不能直接用重加权校准当答案。

**实际条件路径决定什么能保持不变。** [H3 `_embed`](</home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:301>) 把 BF16 condition_proj/refiner 的文本，与 video/audio embedding 合入同一序列；后续 50 个联合 attention 的 200 个主干 linear 使用 native W4A4，文本也被更新，没有独立量化 cross-KV 支路。[Wan](</home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/diffusers/models/transformers/transformer_wan.py:491>) 则是 video self-attention 后接固定文本 context 的 cross-attention，K/V 来自条件；因此更易定位条件支路，但 cross-KV 高精度回退只是普通对照。H3 改文本长度会改变 [packed 位置、RoPE 起点和分段](</home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:647>)：先选同词语集合的角色/顺序交换，并确认实际 tokenizer token 数、tags、packed metadata 完全相同，不能截文本强行对齐。文本里的 Audio Description 仍走文本条件；真正 reference audio 是额外 anchor/layout/时间处理，E010 没有启用，不混为同一实验。

**最强反论。** 任意弱方向都容易被固定大小的误差淹没；teacher 可能本来不理解否定/绑定，或差分主要反映措辞。必须与 **BF16 响应幅度匹配的普通条件对照** 比较，才能问是否特定控制关系额外受损；只把目标差分分母做小不成立。BF16 舍入、噪声敏感性、晚步粗 Euler 变化都可能大于 prompt 效应。即使有限 D 保留，也不保证最终遵循条件，反之亦然。E010 两条自由轨迹不能用于同状态差分；真正控制能力需多 common-noise seed 的双条件 rollout 及独立语义评分。

**若仍需一个廉价决策测试（仅建议，未授权运行）。** 只预选一对清晰的正向角色交换，避免用 teacher 未证明理解的否定句；例如同词语集合的“red cube moves / blue sphere stays”与交换角色，实际 text encoder 产生两份 BF16 embedding。固定 E010 BF16 p36 的 step5、step14 两个既有联合 latent/time，两个 arm 在每一点接收逐 byte 相同的 x 和 packed metadata，CFG=1，原 model_fn 的负号/unpack 不变。每点四次调用得到 `B₀,B₁,Q₀,Q₁`；另在相同确定性 ±1-BF16-ULP latent 扰动上，各算两个 BF16 条件，共 16 次完整 DiT，再做 2 次原点 BF16 重放确认数值底噪。记录实际扰动能量/变动坐标，比较 teacher 的 `D_B` 稳定性及单条件噪声敏感度，不用名义 epsilon。约 2–3 GPU 分钟纯 DiT，加一次原 TE 编码/加载预留 10 分钟，无 VAE/视频；这些点不是独立 seed。

先要求两点 teacher 差分能量均高于重复与上述扰动背景至少 10 倍，否则仅记“观察量信噪比不足”。两点均有 `g<0.5` 且 `||D_Q||²/||D_B||²<0.5` 才记候选衰减；只有大 r² 或相对误差就停止“塌缩”表述。这是投入门槛，不是显著性。阳性也只能要求下一次加入预先确定、teacher 响应幅度可比的普通条件对照，不能直接进入方法或生成预算；若它同样受损，解释退回一般差分精度，而非特定控制失效。1-ULP 对照只排除数值不稳定，不能替代真实噪声种子的语义稳定性检验。现阶段不建议以新损失/新校准命名此对象。
