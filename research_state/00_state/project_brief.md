# 研究任务与约束

更新时间：2026-10-02；仓库基点 `586456b`，已有未提交修改保留。

研究对象：视频生成模型（首批 Wan2.1/rCM-Wan、MiniMax-H3）的 NVFP4 W4A4；允许算法、真实算子、attention、稀疏与系统协同，但研究问题必须统一、可证伪，不以拼接模块作为贡献。

目标：经过 prior-work collision 核查、足以支撑顶会投稿的独立研究贡献。当前没有已验证的新贡献，不承诺录用。

已有材料：历史PTQ代码、真实模型/量化状态、校准输入、旧生成视频及评测。历史结果视为待审计证据。

资源：8×RTX PRO 5000 72GB Blackwell；初查GPU0–5空闲，6/7为其他用户任务。首轮最多用0/5验证，后续按有效实验需求扩展。数据盘 `/data1/models/svdquant-wjq` 约1.4TB空闲；系统盘约4GB空闲。大张量与媒体写数据盘，小JSON、代码、报告留仓库。昂贵任务在命名tmux，日志写 `results/logs/`。不更改旧环境、不覆盖旧结果、不推送远端。

人员：主agent统筹与实验；literature建立文献地图；audit审计历史证据；systems检查真实FP4可行性。首批结束收敛为单一主线，不让agent无限发散。

评估原则：同模型/提示词/种子/调度器/步数/帧数/分辨率；校准与测试分开；记录真实kernel或QDQ；数学相似度、视频质量、实测速度分开；负结果保留。首轮诊断不能作为发表证据。

当前阶段：entry-source-intake 已完成，进入 representative-literature-bootstrap 与历史证据审计。先有signal和problem，再生成claim和实验。


2026-10-04 用户更新：冻结现有可用 smooth＋native NVFP4 W4A4＋BF16 rank32 配方为后续 SVDQuant baseline；不再为对齐官方默认配置投入昂贵完整校准。实现来源沿用既有记录，后续研究以该用户指定基线比较。
