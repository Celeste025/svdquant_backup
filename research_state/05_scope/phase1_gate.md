# 第一阶段范围门槛

C001/C002：当前路线停止。不能在真实CFG反证之后仅靠换更弱场景维持方法主张。

C004：当前证据仅是有用诊断，**不足以独立成为顶会论文**。同一模型一个block、两个timepoint的局部/最终denoiser排序错配，属于已有modality-aware PTQ和propagation-aware calibration的覆盖区。

允许的下一阶段：最多一个泛化探针和一个强简单校准对照。必要证据是heldout可重复、机制可解释且能预测干预效果；之后才考虑自由rollout、native部署和完整评测。若简单双模态权重即可解决且无额外机制，记为基线修复并换研究问题。

暂不做：全模型长时间QAT、几十组rank/bitwidth网格、多模型大benchmark、论文起标题。不能靠增加scope补弱新颖性。
