# Signal Bank

| Signal ID | Source | Abnormal phenomenon | Why it matters | Linked problem | Candidate move | Status |
|---|---|---|---|---|---|---|

## 2026-10-02：基于已完成原生路径的信号重筛

| Signal ID | Source | Observed phenomenon | Why it is uncomfortable | Candidate move | Strength / uncertainty | Next check |
|---|---|---|---|---|---|---|
| S006 | E007/E009原生profile；phase4_systems_signals.md | Wan的LR+smooth约322.5ms，大于主支dense→FP4省下约215ms；完整native慢11.6%，H3却快1.217× | 额外配方成本是否有对应完整网络收益从未确认，不能只归因硬件或模型尺寸 | 强基线/运行区间分析 | 成本为实测；质量价值未知，模型形状不同不是纯scale实验，已知fusion尚未整模比较 | E014从原始W独立量化完整plain，与SVD共同输入/计时 |
| S007 | 已安装FlashInfer SM120官方源码+E006真实QKV形状；phase4_systems_signals.md | per_block_mean=True预先物化FP32[B,H,M/128,N]修正；实际H3主段0.818GiB，4倍N公式13.08GiB | 低位attention API预处理仍可能包含部分二次显存，但这不是硬件下界 | 系统合同/容量分析 | 当前shape与分配为实证，长序列数值仅推算；已有global-mean模式容量线性，尚未测试其取舍 | 先比较现成per_block_mean=False；目前不立新kernel或论文、不GPU |
| S008 | LongLive-2.0 Table3；phase4_deployment_constraints.md | 固定4step NVFP4+KV的64s视频E2E99.5s，另加VAE GPU后57.6s | 单GPU完整实时性与多GPU宣传FPS不是同一服务合同 | 系统边界/资源预算 | 论文自身已明确多GPU；未证明常规同卡双stream不能解决，非新的缺口证据 | 若后续有真实单卡streaming需求，先同模型普通colocation强对照；当前不移植后端 |

S006是本轮优先的基础配方核查；S007/S008保留为待验证线索，不能从“有源码限制/有研究兴趣”直接跳到新方法。E013失败不提升任一信号的新颖性。前三轮已停止的问题与其原始证据保留在claim ledger及实验索引，不因本表换名复活。

## 2026-10-03：E022/E024闭环后的证据更新

| Signal | 来源与观察 | 不确定性 | 当前动作 |
|---|---|---|---|
| S009 | E022 native开发NMSE−32.22%，但16条独立轨迹的QAD−plain MJ均差+.000868、仅2/8文本提高 | 开发仅4文本；共同BF16失真；局部MSE与生成质量错位已有JustQuant等直接近邻，不构成新颖性 | 停止普通QAD扩训练和运动收缩故事；保留完整阴性结果 |
| S010 | E024官方成熟QAD产品AMT.99442/DINO.97902而RAFT仅2/16，部分语义偏离；如港口接近画笔水杯 | 8题中4题仅场景名；低运动不都意味着失败。无同权重teacher；TAEHV、采样和蒸馏混杂未分离 | 官方UniPC合同已支持；E025固定全部16 final latents、完整WanVAE FP32零DiT对照一次；这只是定位问题阶段 |

没有为达到skill形式数量要求新增第三条假信号，也未把信号升级成方法claim。E022不支持E021的QAD运动收缩解释；E024不同产品的静止不能将其复活。下一研究问题须由受控结果产生，不能由想得到顶会成果倒推。


## 2026-10-03：E038/E041的受控动作读出

| Signal | 来源与观察 | 不确定性及近邻 | 当前动作 |
|---|---|---|---|
| S011 | 同native线性权重、只换attention的E038，16/16 FP4配对RAFT均值下降；E041两seed全帧却仍有快速开合，重影使后段不可读 | 光流含构图/尺度/纹理，事件下界不能代表实际总数；Attn-QAT已经覆盖attention-only动态指标下降。没有QK/P/V/decoder因果归因 | 停止统一变慢解释；关联P006（论文问题parked），只查具体解码上下文混杂与未读出的音画事件入口。当前无方法claim，无新GPU计划 |

[完整帧结果](../reports/037_20261003_clapping_readability.md)。不将S011扩成新的运动loss，也不复活E022已被数据否决的QAD运动收缩故事。VAE敏感方向/拼接放大目前只是待排除解释，未登记已成立signal或新颖claim。


E042更新（报告038）：六份既有PCM可提取完整包络，但未建立基线音频峰与视觉接触的一一对应，不登记“AV同步损伤”新signal。P006论文问题parked；0新模型/GPU，停止从当前拍手组继续派生方法。非时间blend帧还已否定“全部重影来自最后时间拼接”，广义decoder根因仍未定位。

## 2026-10-04：完整 H3 配方差的来源定位

S012：E014/independent_summary、E015/independent_summary及E061/summary的实际完整输出。p30 plain/legacy视频SSE在s5=.849813、s6=.886179，去通道均值后仍为.855658/.909404；p36 s14几乎等能量(.998654)，两臂误差cos却仅.466780（p30约.520）。不同量化方案本来就可以产生不同误差方向，因此这不是量化特殊机制或创新证据。它提示固定局部MSE无法单独解释完整配方差，值得定位真实输入偏移是否改变了量化块的新增误差。

关联P009；初始强度仅“受控完整输出差异”，无深度来源证据。E062/E063阴性不提高本信号强度。可能研究动作只限measurement/机制区分；成熟cross-block校准和精确递推已有直接先例。下一E064在两完整量化臂各自实际路径上做预选五块四角，用带符号净作用区分固定teacher残余、BF16传播和输入依赖差，不立方法claim。

S012/E064更新：[报告058](../reports/058_20261004_h3_depth_four_corner.md)。完整来源诊断及独立核验通过，但16非零格输入依赖项的净作用都为抵消，两臂均不满足稳定有害放大动机。停止该解释，完整配方差异本身保留未被充分解释；不由此声称量化更稳定，亦不把源分解当创新。后缀相干草案未运行。


## 2026-10-04：同teacher输入逐层Pareto优势与整模排序相反

S013：E065b/E066及独立复核。固定same-smooth/rank32/native、四既定teacher状态的每个200层video SSE均carry更低，M512/full-M同样本/全有效行三种排序一致，整模video SSE却全更高。挑战的是“这些teacher逐层SSE足以对两个部署候选排序”，不是发现量化误差会累积。固定非负层标量权重不能翻转两个候选，未否定加权训练第三候选。Owner为部署受限量化设计者；潜在动作仅measurement、可干预机制定位。强度：窄范围已复核观察；尚无新颖机制/质量证据。近邻SVDQuant/ARHQ/QDrop已经覆盖最自然分解、残差保护和扰动重建出口。关联P010；下一E067仅用保存样本核查径向近似，任何方向统计不自动推因果或新GPU。

## 2026-10-04：科研投入纠偏后的有界筛查

D097/报告067停止未执行SVD工程。没有新实测signal；RoPE融合与AdaLN shift/Jensen偏置通用出口已有直接近邻。帧间弱变化是否被选择性削弱仍是未观察假设，不能由大相对差分误差、E067方向变化或E014含噪velocity推导。保留反例和数据域判定，避免为数量要求构造新signal。


S011/D098补记：报告068以既有r0媒体检查窄“接近/遮挡后身份混淆”版本；初始分离已有缺陷，未追踪到身份转折，但缺无遮挡控制。此为候选证据不足与工程投入筛选，不是新signal或全局对应机制阴性。暂不开发P重排干预；同一例不继续派生工具。


S013/D099：对E068公开匿名媒体作六条×六帧模型辅助筛查，未识别稳定A/B功能损失，不能报等质量或取代人评；不增强S013为画质机制。pow2-global与低秩胞元秩边界均未产生新实测signal。报告069。
