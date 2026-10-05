# 阶段报告081：V中心化的机制与完整视频检验

2026-10-05，MiniMax-H3，E081/E082均完成。**局部误差改善已保留到原生算子，但两个拍手视频尚未看到明确、稳定的手部细节修复。** 不把中心化已有思路或实现接线认领为创新；主基线仍SVDQuant＋官方Sage3。

## 动机 → 实验

E080中心化有明显局部收益。为判断来源，E081复用两seed、s14/b0,b24的真实QKV，在同一QK/P上拆开“V表示改变”和“概率质量补偿”，固定56heads、36对相邻query、完整keys，g32/128，不重捕获。

设实际BF16块均值为μ、R=V−μ，P为当前QK的softmax、P̂为FP4重构。四臂是 `P̂Q(V)`、`P̂Q(R)+(P̂1)μ`、`P̂Q(V)+((P−P̂)1)μ`、`P̂Q(R)+(P1)μ`。残差先BF16舍入包含在V表示误差内；不能把收益全归因动态范围，两个单项的SSE改善也不能直接相加。

## 机制结果

block24、g32、实际量化QK，相对同QK的精确PV，输出SSE下降：

| 干预 | seed0 | seed1 |
|---|---:|---:|
| 仅改变V表示 | 22.32% | 25.31% |
| 仅补偿概率质量 | 23.02% | 21.93% |
| 两者合用 | 40.24% | 42.07% |

质量补偿只解释组合改善量的57.22%/52.14%，未达预设80%；移除补偿损失44.53%/39.83%的组合改善。**两者都有贡献。** g128、完整attention参考和未量化QK条件结论一致。block0则质量补偿可解释约84%–85%，仅提示两个预选位置不同，不推广为一般层规律。

E081逐tensor复现E080；CPU独立重算856项，归一差≤4.44e−16，两因素闭合相对能量≤1.78e−10。GPU13.414秒/峰10.30GiB，CPU1.335秒。

## 原生实现 → 完整视频

E082采用g128：保留大多数g32局部收益，均值数量和恢复计算更少。私有Sage3副本复用原QK、P量化与online softmax，在最终BF16转换前加入量化前概率质量×BF16均值。只中心化完整视频key块，全部query照常获得补偿；SVDQuant、refiner、padding、原VAE保持原合同。

四真实capture的零补偿路径逐byte复现官方及历史输出；center与E081模拟的差能量仅占center误差0.214%–0.680%，低于预设1%；常值V控制精确恢复。原生center相对完整attention参考，block24总SSE下降14.42%/15.04%，block0下降36.92%/36.21%，四组差分SSE均下降。CPU独立核验通过。以上仍为固定输入局部结果。

随后完成两seed所有20步、每步50层的原生center生成，共40DiT，复用真实noise/embedding及原设置1024×576、124帧、CFG1，并用原video/audio VAE解码。原Sage3＋SVD基线与center对BF16的全124帧距离：

| 样例 | LPIPS 原 → center | L1/MAE 原 → center | L2/RMSE 原 → center |
|---|---:|---:|---:|
| 拍手seed0 | 0.27126 → 0.26532 | 0.05654 → 0.05716 | 0.09434 → 0.09540 |
| 拍手seed1 | 0.36829 → 0.42654 | 0.11803 → 0.16491 | 0.20864 → 0.23654 |

这些是配对距离，不是画质百分比。seed0仅LPIPS小幅降低，其余略增；seed1构图/姿态变化较大，不能把距离增大直接等同质量恶化。

已查看两例全部8页/每例32固定抽帧；seed1另由agent在未读取距离前审阅。seed0仍有明显手指重复轮廓与模糊，未见稳定修复；seed1手部重影持续，姿态、手势朝向与构图改变，也未见明确细节收益。两者仍可辨拍手。此为有标签抽帧观察，非实时完整播放/独立人评，未评价音质。

## 结论与后续

V中心化的局部阳性不是假象，已完成从模拟、原生到完整视频的检验；但本轮没有建立“降低局部SSE即可恢复手指细节”的效果链条。保留它作为已知方法的有效局部对照，暂不替换主基线，不扩大kernel优化或参数/层步扫描。下一需要能解释残留重复轮廓、区分数值距离与细节损伤的具体假设，再决定干预；不能把此次有限结果推广成所有中心化或完整VC无效。

## 资源与复现

两次私有编译，官方源不改。保留两次验收失败：缺Ninja PATH（0.022秒，未执行张量）、CUTE按值helper使均值补偿写入副本（2.078秒）。仅修两处引用后，通过正式验收7.704秒；失败源码/so在DATA1 E082/failed_v1_adae6f16a006fe24。修正so SHA256 `5a7c9d31ccece8de07a9608a3d1045029fe0e751440c2f831dd1c2884ca81ffe`。

完整attention调用约1.406–1.408×基线；此为质量验证实现，不宣称提速。两视频生成137.68/136.73秒，峰37.60GiB；解码11.92/11.47秒。两原生验收/视频生成/解码/指标任务均结束，GPU0/1已释放。视频距离四配对的逐帧归约与六源视频SHA复核通过（非重跑LPIPS网络）。

- [E081方案](../06_experiments/E081_center_mass_plan.md)、[结果](../../results/research/E081/probe.json)、[独立核验](../../results/research/E081/independent_verification.json)
- [E082方案](../06_experiments/E082_center128_video_plan.md)、[原生验收](../../results/research/E082/native_validation_v3.json)、[原生独立核验](../../results/research/E082/native_independent_verification.json)
- [视频指标汇总](../../results/research/E082/video_summary.json)、[root观察](../06_experiments/E082_root_visual_review.md)、[seed1独立观察](../06_experiments/E082_seed1_visual_review.md)
- [并排视频入口](/data1/models/svdquant-wjq/research/20261005/E082/review/index.html)。左BF16，中SVDQuant＋Sage3，右增加V-center128。原视频含音轨，并排片静音。
- [seed0并排视频](/data1/models/svdquant-wjq/research/20261005/E082/review/vbench0161_r0_triplet.mp4)、[seed1并排视频](/data1/models/svdquant-wjq/research/20261005/E082/review/vbench0161_r1_triplet.mp4)
