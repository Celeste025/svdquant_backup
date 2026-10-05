# Query-only 布局与 128-token Qmean 相位：待测诊断，不登记方法 claim

2026-10-02。只读源码与一手文献；本次仅新增此记录，未运行 GPU、未修改旧实验。结论：H3 的分组几何已核实，但**分组改变量化结果本身是平凡事实，不是贡献**。目前没有分组相位导致实际视频闪烁的证据，也不将 E017 的最终视频评分差归因于它。

## 已确认的真实数据路径

- [latent 形状生成](/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:241)：124 输出帧、576×1024 分辨率产生 video latent `[1,24,37,36,64]`；audio latent `[2,32,207]`。
- [video patchify](/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:15)：patch=(1,2,2)，`nctrhpwq -> nthwcrpq` 后 flatten；因此视频是 T-major、W-fast 的 `[37,18,32]` token 网格，每 latent 帧 576 query，共 21312。这里的“帧”均指 latent 帧。
- [fl2va packing](/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:647) 明确为 `[text | cond | audio | video | pad]`。E017 [调用 builder](/home/wjq/workspace/svdquant-exp/scripts/research/run_h3_fp4_attention_video.py:168) 没有传 cond/ref，所以 audio 的 414 token 在 video 之前。不能只用文本长度计算视频起点。
- [attention 入口](/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:17) 在 norm/RoPE 后保留 packed row 顺序。[adapter](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_fp4_attention.py:194) 对整个有效段 `[:n]` 一次调用官方量化，再执行 noncausal attention；不会在 audio→video 或帧边界重置分组。
- [官方 preprocessing](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/nvfp4_attention_sm120.py:123)：K 全序列去均值，然后 Q/K/V pad 到 128；Q 从 packed row 0 起 reshape 成连续 128-token 组并取均值。global 模式取整个 padded Q 的均值。[correction](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/nvfp4_attention_sm120.py:160) 为 FP32 `Qmean @ K_centered.T`；它使用量化前 K。[Q/K/V pack](/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/nvfp4_attention_sm120.py:234) 才生成 E2M1 codes/E4M3 scales。

由 [E017 manifest](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E017_h3_fp4_video_manifest.json:42) 的文本长和 cu 验算：

| 样例 | text | audio | video 起点 b | 有效总长 | 第 t 个 latent 帧起点相位 `(b+576t) mod 128` |
|---|---:|---:|---:|---:|---|
| p30 | 501 | 414 | 915 | 22227 | 19（偶数 t）↔83（奇数 t） |
| p36 | 813 | 414 | 1227 | 22539 | 75（偶数 t）↔11（奇数 t） |

这是 `128/gcd(576,128)=2` 的**分组成员关系周期**。两组 phase 都非零，因此每个内部帧边界都横跨一个 Qmean 组，跨界比例交替；并非“只有隔帧才跨界”，也非 0↔64。末尾 Qmean 还混入 zero padding，是应单独排除的边缘效应。模型 [temporal grid](/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:588) 有 `(1,4,4,4,4)` 的非均匀映射，不能把上述几何周期写成两个输出帧周期。

## 最近碰撞与本次定向复核

| 一手来源 | 已覆盖内容 | 未在所核材料中看到的精确内容 |
|---|---|---|
| [PAROAttention §3、4.1、4.3](https://arxiv.org/html/2506.16054v1) | 三维展平改变邻接；按 head 选择 F/H/W 六种排列，改善 P 块的量化分布与稀疏；分组/重排的重要性已明确 | 固定 K/V、只移动 post-RoPE Q 时，128-token Q-centering 与 latent 帧步长的相位作用，以及是否导致相位随动的时序误差 |
| [DeltaQuant 官方项目](https://hanlab.mit.edu/projects/deltaquant) | 线性层激活按三维 cube 分组，以均值/core 加低位 delta 利用时空相似性 | Q-centering/attention correction 的帧相位诊断；但“改用三维均值组”不能据此直接称新方法 |
| [RotateAttention §3、4](https://arxiv.org/html/2607.02584v2) | 3D RoPE 的 frame/height/width **通道**分区影响 outlier，做通道旋转与 P 范围优化 | token 轴连续 Qmean 组的内存相位；其 spatiotemporal partition 不能与这里的分组混同 |
| [SageAttention3 §3.1、Table 1–2](https://arxiv.org/html/2505.11594v1) | 沿用 SageAttention2 Q/K smoothing；消融 NVFP4/MXFP4、P 两级 scale；报告 CogVideoX/HunyuanVideo/Mochi 整体视频指标 | 未见 Qmean 的 token-group 大小或布局与 temporal flicker 的受控消融。论文的 1×16/1×32 是 microscale granularity，不是 128-query mean group |
| [SageAttention3 官方 README](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell)、[setup.py](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/setup.py) | README 建议其他视频模型必要时按层/步回退 Sage2++；核查时 setup 行106–107 固定 QBLKSIZE/KBLKSIZE=128，行156 package version=1.0.0 | 没有给出 Qmean layout→flicker 实验。本轮没有核实一个独立官方“3.1”论文/实现合同；不能把用户报告的版本号视作已覆盖的独立方法研究 |
| [Delta Attention §3、4.1、Fig.6](https://arxiv.org/html/2505.11254v1)、[作者源码](https://github.com/jeffwillette/delta-attention) | Willette 等在 LLM sparse attention 中每 γ 行抽一个 query，向邻域广播 dense−sparse output correction；确实测 γ=8…256 的 PPL/latency 与邻域 cosine | 这是 query **采样步长**，没有测 Qmean 分组量化、视频或 flicker；不能因为存在 stride ablation 就当作同一 claim |

这轮补查限于 SageAttention3/官方 Blackwell 包与 Willette 等的 Delta Attention，未扩展模型或 baseline。另有 [KJNodes 用户 issue #629](https://github.com/kijai/ComfyUI-KJNodes/issues/629) 报告 SageAttention 3.1.0、Wan2.2、5070Ti 下闪烁，切回 SDPA 消失；该报告没有 query 分组、相位或周期测量，只是一手症状报告。对“尚未见”的结论应限定在上述材料，不声称文献全集不存在。

## 严格残余：什么仍值得一次最小反证

已知的分组依赖可以直接从量化非线性推出。只证明更换组成员会改变 Q codes、均值或输出，没有论文贡献；“对齐到帧/tile 后更准”也可能只是 PARO/局部均值的自然应用。几何周期不意味着误差强度有周期：Q/K 统计和 native arithmetic 可以掩盖、削弱或放大它，必须测。

可保留的问题仅为：**在固定内容、post-RoPE Q/K/V 与 KV 执行路径下，packed-memory 分组相位是否产生可归因、随干预移动的 latent 时空误差结构；且这个结构是否有不能由普通组选择/现成 global mean 解决的实际成本或质量约束？** 第一问尚未测，第二问更未成立。E017 的局部/最终 latent 与视频评分不一致只说明代理指标不足，不能作为此机制的正证据。

需要保留真实 correction 合同。若记 `Kc` 为中心化的高精度 K、`Khat=Kc+εK` 为固定解码包、`Qhat_g=Q−μg+εQ_g`，忽略额外浮点舍入时：

`S_g = Qhat_g Khatᵀ + μg Kcᵀ = Q Kcᵀ + (Q−μg) εKᵀ + εQ_g Khatᵀ`。

因此固定 K packets 后，改变 μg 仍会通过 K 量化误差项改变 logits；不能把所有变化都解释成 Q4 reconstruction error，也不能用 `(Qhat_g+μg) Khatᵀ` 替代真实 native 合同。

## 最小可证伪测试定义（本次不执行）

1. 只取一份实际轨迹的单层/单步 post-RoPE Q/K/V。对相同视频 Q 做两种 row layout：原序与循环移位 64 token；text/audio 行不动、长度不动、K/V packets 与 KV 顺序不动。必须在 RoPE 之后移动整行，输出逆置换，避免改变位置语义。输入与 packet 可记录 hash。
2. 测交换残差 `C = Π⁻¹ Â(ΠQ,K,V) − Â(Q,K,V)`，重算 block Qmean 和其真实 FP32 correction。BF16 做同样置换作为数值控制；global-mean 控制复用同一均值/correction，避免 reduction 顺序成为额外变量。先在量化前中心化与 correction 恒等重构中检查代数，再看 native 输出，避免把错误模拟当机制。
3. 将误差映回 `[latent_t,18,32]`，比较均方误差、空间带状图与相邻 latent 帧残差差分；剔除视频首末 Qmean 组和循环移位接缝涉及的组。要检验的是误差结构是否跟随人为 Q 分组相位移动，而非仅观察到 C 非零。可在 CPU 算术重放中额外冻结/搬运原 Qmean group label，以区别重新分组和纯 row relocation；它不是新 baseline 或 kernel 开发要求。
4. 若相位随动结构不存在、只在接缝/末尾 padding 出现，或变化可由 BF16/global 控制的常规数值差解释，停止这个具体机制；不扩大 prompt 或 tile 网格。若成立，也仅升级为一个 operator-level 原因候选，仍需独立因果干预才能讨论视频后果。新方法必须留下已知 global mean、普通 frame/cube grouping、PARO 重排之外的具体残余成本/约束，不能仅靠这一次观察立项。

与此分开的 [P4 partition composability 记录](/home/wjq/workspace/svdquant-exp/research_state/02_problems/fp4_attention_partition_composability.md) 继续保留，不合并成方法 claim。

## 后续实测记录（保持以上初始只读判断的时间范围）

E018现已完成1DiT真实状态重放及27次原生attention干预、142文件独立CPU核验。确有可干预的分组相位能量结构，BF16/global排列控制与block128内区完全不变；但同相位伪边界也交替，追加固定栅格向量投影没有整体奇偶翻号。当前只保留有限算子观察，方向park，不外推视频闪烁。见[报告017](../reports/017_20261002_query_group_phase.md)及其中完整原始证据；没有将近邻未覆盖的精确诊断自动认作新方法。
