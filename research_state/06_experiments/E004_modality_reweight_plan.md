# E004 — 普通模态重加权校准能否消除 E003 的视频/音频取舍

状态：2026-10-02 运行前预注册；已完成，结果另见 `E004_modality_reweight_results.md`。原门槛未变。模式：exploration / 强简单对照，不作为新方法。

## 问题与决策

E003 在 p1、step0/19 发现：修复后的旧 SVDQuant 使 block0 全局误差降低，但在其后49个BF16 block的传播后，video output NMSE相对直接W4A4恶化13.7%/8.5%，audio改善。普通 pooled error 被文本巨大能量主导，与最终视频目标不一致。

假设：固定量化结构、rank和计算预算，仅用常规模态归一化或视频加权的块重构即可明显改变或消除取舍。若成立，将该现象归入既有多模态校准目标问题，停止把它作为新颖机制。若失败，只能否定本次固定子空间的简单对照，不能据此获得新颖性；MBQ/MixDQ/PulseQuant碰撞风险保持。

## 最小有效设置

- H3 pruned BF16、完整124帧/576×1024 packed text/audio/video输入；torch SDPA，实际调用计数和BF16 Q/K/V校验。
- 与E003同一rank32旧state、修复后高精度低秩输入语义。旧state曾受hook bug影响，本实验不宣称标准重新校准的SVDQuant结果。
- 仅更新block0 `mlp.fc2` 的低秩上投影B（5376×32），固定其A、qweight、smooth与其余三个投影。未增加rank或推理分支。所有变化是普通块输出重构。
- 用全token真实量化上游activation；不抽样token，不把text-only注入当成text-only量化。
- 训练：p20/p25 × step0/19（4个样本）。本轮未用于拟合的评估：p11/p46 × step0/19（4个样本）。**这些都是原PTQ校准集；只能称本轮refit隔离，不能称原PTQ-heldout。** p11/p46未用于本轮拟合及E003探索，不据评估结果调整权重。

## 闭式简单对照

设corrected SVDQuant的块输出为y0，BF16目标为y*。fc2高精度低秩输入经固定A得到特征phi，fc2后的AdaLN gate为g。只更新B+=DeltaB，FP32线性近似为 `y_hat = y0 + g * (phi @ DeltaB.T)`。

对每个 `combined_indices` 群组（相同AdaLN gate）累计完整token的 `phi.T @ phi` 和 `phi.T @ (y*-y0)`；对每个输出通道乘入其g²/g，形成32×32 ridge正常方程。使用FP64累计/求解，关闭TF32。拟合DeltaB后转换回BF16，**最终指标必须重新执行真实block与模型，不能用线性近似损失代替**。

三个同容量/同数据/同求解器的固定对照：

1. `pooled_refit`：所有有效模态的原始平方误差之和（全局归一化仅缩放目标）。
2. `normalized_refit`：text/audio/video各自总误差除以各自训练teacher总能量，三者等权。
3. `video4_refit`：上述三项权重 video:audio:text=4:1:1；4为预先固定，不扫参。

各对照均排除padding作为拟合目标；共享B可能改变padding输出，完整评估仍保留padding。ridge取每输出通道Gram平均对角的1e-3（下限全通道平均的1e-6），惩罚DeltaB而非B本身。仅一次求解，不搜索lambda/权重/epoch/样本。数值线性近似存在BF16舍入误差，报告真实训练目标与拟合预测的差异。

必要基线：直接W4A4、修复后旧SVDQuant、pooled_refit。两个重加权refit均要报告，不能择优后隐藏另一项。正结果不能从增加可训练参数获益，因为三个refit完全同容量。

## 分阶段门槛与止损

A. 数值/训练有效性：真实BF16块重放err2=0；分组gate逐token一致；固定qweight/smooth/A指纹不变；真实refit对应训练目标不能比原corrected恶化超过2%。若全部refit训练目标无至少2%改善，标记“固定A/fc2-B控制无优化能力”，不做新调参；仅允许下述最多2例完整传播阴性核查。

B. 本轮隔离块级评估：至少一个重加权方案在4例中至少3例降低video block NMSE，且总误差/总参考能量比原corrected改善至少10%，方进入完整模型传播。若仅训练改善或块级不改善，最多在预定p11×step0/19进行完整传播阴性核查，然后结束；无权声称视频质量改善或加权原理失败。

C. 49 BF16 block完整传播：对teacher、plain、corrected及全部三个预注册refit（训练有效性单独标记，不隐藏失败对照），在固定4例执行。zero-pulse重放必须逐元素一致；实际torch SDPA调用数应与teacher一致。主要指标是video模型输出NMSE，audio模型输出NMSE为取舍指标；不解码，不运行VBench，不做自由rollout。

预注册决策：若重加权相比pooled_refit的aggregate video NMSE改善≥10%、至少3/4例改善，且相对plain不再恶化（aggregate ratio≤1），则本次取舍可被普通重加权校准消除，停止将其单列新贡献。若只改善video但audio相对corrected恶化>10%，记为质量取舍移动，不称同时修复。若未过门槛，保留本次阴性结果，停止在本对照上继续调参；不通过增加层数/rank/目标权重追逐结果。

## 预算与证据边界

- 单GPU1，启动前`nvidia-smi`；tmux命名 `research-e004-h3-reweight`，日志在results/logs/E004_h3_modality_reweight.log。
- 预算上限120分钟GPU墙钟；先A/B门槛，再C；任何OOM先结束并保留partial，不扩卡/降分辨率掩盖。
- 预计训练仅4次上游/块teacher捕获；局部评估4例×5个量化方案；C最多4例×7次完整forward（teacher+zero+5个量化方案）。复用同一次teacher及block输入，GPU计时仅用于预算，不用于性能论断。
- 主产物 `results/research/E004_h3_modality_reweight.json`，拟合B单独保存，源/模型/state/cache哈希，训练/评估ID与完整分项指标入JSON；不覆盖旧state。
- 主要贡献类型为“简单竞争解释的检验”；可发表性不会由这一个小样本实验成立。纯局部改善、训练集改善、teacher-state指标都不是最终生成质量证据。
