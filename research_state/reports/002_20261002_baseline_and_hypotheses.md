# 阶段报告 002：修复基线，筛选两个机制问题

日期：2026-10-02。研究仍处探索期，没有新方法有效或可发表性的结论。

**后续更正**：本报告原始E001数据在恢复环境默认SageAttention下取得，并非纯BF16 attention；各arm使用相同后端。显式torch BF16 attention复验也已完成：修复后的video NMSE相对旧hook为+0.19%（无稳定改善）；corrected SVD相对plain的全block NMSE−9.99%，video-token NMSE+76.16%。因此模态取舍signal保留，修复收益不成立。原数据保留，下文数字专指原Sage配对。复验数据见 `../../results/research/E001_h3_branch_precision_torch.json`。

## 已完成及实际结论

- **H3基线修复**：低秩分支改为读取高精度平滑输入。10项CPU回归通过；原始576×1024/124帧packed输入，2个prompt×首/末步，4组真实block0配对完成，BF16重复输出逐元素一致。
- **修复影响有限**：固定旧校准state，video-token聚合NMSE从0.00263581变为0.00261522（−0.78%）；不能把此前生成效果差主要归因于hook bug。旧state仍需在正确目标下重校准后才是正式基线。
- **出现模态取舍信号**：corrected SVD相对plain W4A4，全block聚合NMSE 0.00126151→0.00113658（−9.90%），video-token却0.00149335→0.00261522（+75.12%）。text约3%tokens，占BF16输出能量88.1%–99.7%，占SVD误差能量79.7%–97.6%。这是4个block case的局部现象，不是最终视频变差证明。
- **原生FP4可运行**：当前PyTorch/SM120已验证真实packed NVFP4 GEMM与两级scale。没有端到端速度结论。
- **停止batch-scale故事扩展**：合成控制显示正常区间2的整数次幂global scale能被FP8指数吸收。非2倍率可改变量化格点但并不自动恶化误差；没有真实serving失效证据，暂挂起。

## 当前实验与查新决策

1. **E002：跨denoising步误差相关**。普通Wan、2 prompts、连续6步、完整两CFG分支；对照W4A4与W4A16、去channel bias前后。脚本已审查，首次因PATH缺Ninja退出；已复用现有Ninja/扩展目录重启。只验证teacher-state相关，不能推断自由rollout。
2. **E003：H3不同模态误差是否真的决定最终输出**。固定block0误差，分别只注入text/video/audio，再由剩余BF16网络继续，比较最终video/audio denoiser。用于判断模态取舍是否值得继续；不先做普通loss重加权。
3. **查新收窄**：De-biasing Diffusion已做低bias/随机舍入；TCEC/QuAKE已做时间历史纠错。MBQ已做模态加权校准，MixDQ涉及条件极值及指标失配，RGSQ涉及模态分区度量。不能把这些上位想法当新意。残余贡献仍未证明。

## 下一步及停止规则

E002若去bias后相关弱或不稳定，终止互补误差路线；若强，先排除输出比例bias并测闭环影响。E003若text的大局部误差也主导最终video，就否定“校准被无关模态带偏”的解释；若排序逆转，进一步测多个block与heldout样本，仍需区别已有modality-aware PTQ。

代码、逐case数据与实验边界：`../06_experiments/`；真实结果在 `../../results/research/E001_h3_branch_precision.json`。本轮不生成论文宣传性结论，不覆盖历史结果。
