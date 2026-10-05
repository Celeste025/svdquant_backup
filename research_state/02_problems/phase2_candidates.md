# Phase 2：两个有限诊断问题，不是已成立的论文方向

日期：2026-10-02。已读 E002 汇总、阶段报告 003、E003 完整结果与审计。按 problem-formulation skill 分离 signal / problem / claim / method；本次只更新本文件，不改主 agent 的 current_state。

**判断：当前没有足够可信的剩余方法贡献可以直接立项。** 以下两项值得各做一个有限实验，优先 P2-B。阳性只允许继续查新与提出机制，不能直接进入大规模训练。

证据约束：E002 在真实 CFG6 输出下否决了强跨步相干误差假设；W4A16 与 activation-increment 的分解含交叉项，不是独立方差。E003 不是全阴性：确实看到全 block 局部误差改善、最终 video endpoint 恶化而 audio 改善；但只有一 prompt、block 0、两步的实际幅值 pulse，普通模态权重和输出几何先例已很密集。E001 默认 Sage 与 E003 显式 BF16 torch attention 不可混用。

## P2-A：CFG 差分精度是否被 NVFP4 的离散尺度切换主导？

**来源与问题。** E002 的分支内 activation error 仍有弱时间相关，但 CFG 后相干性几乎消失。问题不是再证明 CFG 放大误差，而是：相近 cond/uncond 激活通过原生 E2M1 + E4M3 两级量化时，差分失真主要来自各自的连续重建误差，还是少量微块尺度/码元边界切换？后一机制是否在保持 guidance 不变时形成一个不能由分支二阶统计预测的精度下限？受益方是视频 PTQ 与部署系统设计者；类型是机制/测量。

**致命已知工作。**

- [GCBT，2026-10-01，§3.3 与附录 C](https://arxiv.org/html/2610.00930v1) 已把配对分支作为二维编码空间，结合 guidance 和 second moments 得到旋转。其 surrogate 忽略 transformed-coordinate error cross terms；这是已知适用边界，不是本项目发现。共享尺度、差分编码、分支旋转本身不能算新方法。
- [DSAQuant，2026-09-03，§4.2 与附录 C](https://arxiv.org/html/2609.04031v1) 已直接说明分支 activation grids 不同、误差弱对齐、CFG 放大，提出晚步关闭 CFG。E002 的定性现象已被覆盖。[GAMP，2026-07-09](https://arxiv.org/html/2607.08241v1) 还说明仅保护 guidance gap 会漏掉共同的 branch drift。
- [ScaleSearch，2026-05-12](https://arxiv.org/html/2605.12464v1) 已在可表示 E4M3 scale 邻域搜索，改善 NVFP4 与视频 attention。因此“搜更好的尺度”也不能成为剩余贡献。

**尚未核实的机制。** 上述原文没有使本轮能够确认：native E4M3 scale 切换及跨分支量化误差的精确协方差，是否足以系统性推翻 GCBT 的 surrogate 排序。这里需要的是有量级的失配及可部署的解决空间，而不是发现 surrogate 不总精确。现有每 token×16 通道已有独立 scale；不能误写成整个模态共用微块 scale。全局 scale 的二次幂变化常被指数吸收，不能预设共享 global scale 是主因。

**唯一关键实验（一天内；只用 2 prompts × 早/中/晚 3 steps × 3 个预注册线性位置）。**

从同一 BF16 teacher 获取配对激活，固定原生 W4 权重，在同一 native GEMM 契约下比较：独立 NVFP4 scale、配对微块共享 scale、GCBT + 原独立量化、GCBT + 配对 scale。共享 scale 只是因果干预，暂不命名方法。记录 local guided output 的绝对误差、两 branch bias/energy/cross term、E4M3 scale 不同率与 E2M1 code change；同时做独立 ScaleSearch 小范围校正作为强廉价对照。对同一 pair 令 m=(x_c+x_u)/2、d=x_c−x_u，附带 d→ηd（η=1,1/2,1/4）的局部响应曲线；η=1 的真实激活结果决定是否保留，人工缩差只用于定位边界。选择固定一个位置做 BF16 剩余网络 continuation，避免拿局部 guided activation 当最终 velocity。所有 arms 使用完整序列，固定 scale 计算域；reference unpack 与 native pack 必须先一致。

**可能形成的 claim 方向，仅待验证。** ① second-moment 相似但离散网格状态不同的配对，其 CFG 误差排序不同且可解释；② 在固定指导强度、位宽和真实执行预算下，控制网格耦合比改善各自 MSE 更有效。不能仅凭这两句话声称 novelty。

**Kill 条件。** 若 native GCBT 或独立 ScaleSearch 已解释/消除绝大多数失真；或干预在真实 η=1 上收益 <10%、只有人工缩差才显著；或 native GEMM 局部收益经 dense continuation 不保留，则停止。10% 是本轮资源分配门槛，不是统计显著性标准。若仅发现改成共享 scale 略好，记录工程结论，不能包装为 CFG 新机制。不要关闭 CFG 来“证明”量化改进；不要重启时间 correction 路线。

**状态：needs measurement，较低优先级。** 与 GCBT/DSAQuant/ScaleSearch 三面重叠，阳性也很可能只剩工程优化。

## P2-B：低位 projection 是否通过改变 attention 量化网格，产生非加性的误差？

**来源与问题。** E003 中 error energy 与功能影响错配，加上旧环境默认 Sage 的审计发现，说明必须区分 projection 和 attention 的真实组合。本问题是：在 W4A4 projection 后运行低位 attention 时，上游 Q/K/V 的小扰动是否改变下游微块 scale/码元，使 attention 量化额外误差强依赖上游量化，而非由两段各自误差独立预测？这关系到 PTQ 与 attention kernel 能否模块化拼接，尤其是少步视频剩余纠错余地小的部署；类型是机制/算子接口。

**致命已知工作。**

- [SageAttention3，2025-05-16，§3](https://arxiv.org/html/2505.11594v1) 已做原生 FP4 QK/PV、P 的两级 scaling；[ScaleSearch](https://arxiv.org/html/2605.12464v1) 已改进其微块尺度。单独优化 attention quantizer 不新。
- [VC-Attention，2026-09-14，§3.1–3.2](https://arxiv.org/html/2609.15810v1) 已分解 P/V 误差并处理 V token outlier，指出 QK smoothing 不解决 V 主导误差。若组合问题仅是 V outlier，直接撞此工作。
- [DSAQuant 附录 J](https://arxiv.org/html/2609.04031v1) 已组合 W4A4 与 SageAttention 并报速度；[QuantSparse](https://arxiv.org/html/2509.23681v4) 已讨论量化和稀疏的 attention shift。故“两个近似共同使用会更差”与“联合校准更好”都不足以成文。

**尚未核实的机制。** 本轮未从这些原文确认，是否有针对“上游 W4A4 扰动→下游 E4M3 scale/code-cell 切换→额外 attention error”的受控因果分解。要识别的是这个可干预的边界，而非普通 softmax 对输入敏感、普通误差传播或 AMP 精度叠加。H3 有 QK RMSNorm，raw QKV 幅值与 attention 真正看到的方向不能混为一谈。

**唯一关键实验（一天内；native FP4 attention 已可调用才执行）。**

固定完整 packed 输入、一个 Wan 模型和 3 个预注册 attention blocks、2 prompts × 2 steps，做同输入 2×2 因子实验：BF16 / native W4A4 projection × BF16 / native FP4 attention。严格保持 RMSNorm、RoPE、mask 与 token 顺序相同，Q/K/V 从对应真实 projection 产生。记 O00、O10、O01、O11，分析交互 I=O11−O10−O01+O00，而不是把四个 NMSE 相减。第五个 counterfactual：在 W4A4-QKV 上复用 BF16-QKV 时下游 attention quantizer 的 scales，仍对实际扰动后的值按这些 scales 编码；记录新增 clipping，避免把 clip 变化当作 scale 因果收益。若该 arm 无法在 native kernel 合法表达，则只作明确标注的 reference 诊断，不作 native 性能结论。记录 I/总误差的能量及符号交叉项，定位是否由 scale switching 主导；再对固定一个 block 的输出进行相同 BF16 continuation。不开 full rollout，不调模型。

**可能形成的 claim 方向，仅待验证。** ① 原生低位算子的精度契约需要包含上游扰动，而不仅是固定 BF16 输入上的 kernel cosine；② 一类可预测的网格切换主导组合误差，并存在不增加位宽的接口处理。若最终处理只是联合 QAT 或普通 scale search，则贡献门槛仍未越过。

**Kill 条件。** 若 ||I||² / ||O11−O00||² 的中位数 <0.1，或量级虽大但正负抵消、没有更差的功能误差；或固定尺度不改变 I、普通 ScaleSearch/V smoothing 就消除差异；或 BF16 continuation 不保留效应，则停止。若原生 FP4 attention 尚不能可靠调用，不为本诊断投入一天以上搭 kernel，直接 parked。只有模拟量化阳性不算硬件机制证据。

**状态：needs measurement，优先于 P2-A。** 原因是可以用一次固定 factorial 实验否定整个组合机制，且与已做 E002 的同一失败叙事距离更远；目前没有阳性结果，更没有方法贡献。

## 资源与决策边界

两项均不授权自身运行 GPU；执行排期由主 agent 统一安排。不要把已否决的时间相干假设重新命名后续做。若两项被 kill，诚实记录本轮没有可信剩余贡献，转向新的实测 signal；不要靠增加层数、模型数或调门槛挽救。联合音视频 token 的普通重加权留作 E003 的廉价 baseline，不并入以上两题扩大范围。
