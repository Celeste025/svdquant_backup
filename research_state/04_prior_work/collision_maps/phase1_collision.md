# 第一阶段候选碰撞与残余贡献

更新：2026-10-02。先前工作的具体核实深度见 `../../01_literature/`，不声称检索穷尽。

## C001/C002：跨步误差与互补量化

已覆盖：De-biasing Diffusion的低bias/随机舍入，TCEC时间补偿，QuAKE输出历史估计，PulseQuant传播风险。只把这些从FP8/INT4迁到NVFP4或从图像迁到视频，不足以区分。
潜在剩余：在等存储/计算预算下，改变实际solver传播后的跨步相关并取得不可由已知bias correction解释的收益；目前无此证据。
实验证据：E002两个prompt真实CFG6输出未满足强相关门槛，因此当前方法路线停止。
状态：heavily overlapped；abandon current method route，不做双权重的大实验。

## C004：H3模态端点误差与生成响应错配

已覆盖：[MBQ](https://arxiv.org/html/2412.19509v2)的模态加权校准；[MixDQ](https://arxiv.org/html/2405.17873v2)的条件极值/指标错配；[RGSQ](https://arxiv.org/html/2609.25492v1)的模态几何度量；[PulseQuant](https://arxiv.org/html/2609.33384v1)的pulse加dense continuation。

因此“text能量大”“按模态重加权”“用梯度/Fisher衡量重要性”“注入误差测最终损伤”均不能单独算新贡献。E003使用的是已有诊断范式。

尚待解释部分：在joint音视频网络的同一个共享权重块中，减少整体重构误差是否系统性造成video/audio目标间的反向变化；它来自校准数据/能量权重的偶然偏差，还是存在可泛化的共享表示约束？目前只有block0少量teacher输入，不足以回答。

残余贡献状态：unverified/partially verified，尚不能判断paper-worthiness。若简单相对MSE或MBQ式重加权解决，即作为基线工程记录，不包装新方法；若跨heldout/block出现稳定且现有方法不能预测的目标冲突，才重新提出更具体claim。

下一证据：E003完成；随后必须对新提示词/不同block以及简单模态校准对照进行小规模验证。需要区分自然量化误差归因与等能量方向敏感性；不把局部RMSNorm不变性外推为全网络gauge。
