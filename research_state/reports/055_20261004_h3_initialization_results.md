# 055 — 改低秩初始化有局部收益，但不足以推进该路线

2026-10-04。E061完整执行与CPU汇总完成。它是既有基线的有界对照，不是新方法；没有新增视频、MJ或质量结论。

## 动机 / 观察

E060确认了现有H3导出与legacy state的身份，但没有证明其等价于完整官方SVDQuant校准。一个具体可判定的问题是：在同样smooth和rank预算下，直接把平滑权重的主方向放入高精度支路，是否比把低秩容量用于量化残差更有效？若一致有效，应先补强普通基线，避免围绕legacy配方的缺点发明机制。

## 实验

固定H3 CFG1、所有smooth、rank32、BF16残差构造、NVFP4/native执行、attention及输入。两套新权重逐层使用同一seed及相同FP32随机SVD近似：`error_init`分解`Ws−Q(Ws)`，`weight_init`分解`Ws`；都随后量化`Ws−BA`。全部200层参与，而非单层误差代理。

两套导出共22.281GB、400层packet解码与旧QDQ逐元素相同。先做一次legacy完整复现，再分别对四个teacher输入执行两新臂，共9次完整DiT；输入身份、200 native/pack及102 SDPA每次核对。原BF16/legacy/plain同输入参考复用。预设门槛是四点video velocity SSE逐点同时优于legacy及error_init至少20%；不是统计显著性或论文质量标准。

## 结果 / 结论

下表是对同输入BF16的video velocity SSE，按每行legacy归一化，越低越好；不是画质分数。

| 固定输入 | legacy | 已有plain | error_init | weight_init |
|---|---:|---:|---:|---:|
| p30 / step5 | 1.000 | 0.850 | 0.962 | 0.883 |
| p30 / step6 | 1.000 | 0.886 | 1.025 | 0.925 |
| p36 / step14 | 1.000 | 0.999 | 1.004 | 0.998 |
| p36 / step15 | 1.000 | 1.020 | 1.054 | 1.059 |

weight_init相对legacy分别为−11.72%、−7.49%、−0.19%、+5.87%；相对error_init为−8.22%、−9.78%、−0.62%、+0.47%。**四点均未通过预设投入门槛，且最后一点变差。** 去通道均值后结论也未转为一致改善；audio读出另保留在原始汇总，不替代video主判据。weight_init相对plain有三点更差，另一点仅低0.056%，不据此称低秩方案稳定优于plain，更不据此选择最终质量最优配方。

执行合计218.452秒（含两套导出及模型装载），峰allocated37.611GiB。旧基线raw/velocity逐元素复现；CPU汇总0.561秒，0CUDA。主门槛不通过，按计划没有追加4次shifted输入、视频或完整PTQ。原worker/tmux已退出，GPU0释放。

## 下一决定 / 解释边界

停止当前固定smooth初始化替换路线，不扩rank/seed/提示网格。两个窗口的prompt和timestep共同变化，不能把p30改善、p36不改善解释成纯粹早晚步机制。

本实验没有测试官方carry-Q交替优化、含LR的smooth搜索、FP64 full SVD或独立质量泛化，因此**不否定完整官方SVDQuant可能更好，也不恢复“legacy已是最强基线”的口径**。其价值是完成一个直接的基线投入判别；普通实现或校准修正依然不属于创新。后续新研究须有独立的具体预测，且不能由这些旧基线结果声称已排除成熟方法。

## 可追溯结果

- [计划](../06_experiments/E061_h3_lowrank_initialization_plan.md)；[执行器](../../scripts/research/probe_h3_lowrank_initialization.py)，SHA `334ab1747c8aad44f3f309d0bbcaa873f65c2bf382da6a495811447f12d35d65`
- [CPU check](../../results/research/E061/check.json)，SHA `40cb0a3d1c086f4fca32d510cc074bbed2bf5f457cb0cd37e82ff446ac02cf61`
- [完整执行](../../results/research/E061/evaluate.json)，SHA `101a5c0b718c5020947ebd54c6214e64fbd2903473e05e1b51a8db415503afa0`
- [独立CPU汇总](../../results/research/E061/summary.json)，SHA `5aed4e2cc5f325d2c2588cc3118190985688eeac1cbadcf277b4219f9fa7c0a4`

独立NumPy复核已完成：[independent_check.json](../../results/research/E061/independent_check.json)，SHA `beb1640f025c4cea6fbfdb31714eb38793124633e1bfd0ba01752c25a98fed91`。重hash21份实际PT，9个实际输入及raw/velocity绑定通过，fresh legacy四个输出数组逐byte一致；BF16位解码后直接FP64归约，与执行器最大相对差3.75e−16。CPU 0.823秒、0CUDA，独立确认0/4过门槛及9次/0shifted调用。
