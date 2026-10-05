# 阶段报告082：BF16主干上的Sage3与V中心化对照

2026-10-05，MiniMax-H3，E083 complete。**移除SVDQuant主干量化后，两个拍手seed仍未观察到center128带来稳定的细节收益。当前V去均值与补偿属于已有方法验证，不能作为我们的创新点。**

## 动机／假设 → 实验

用户指出只在SVDQuant主干比较会混入上游量化和传播影响。因此保留完整BF16+SDPA参照，新增BF16+官方SageAttention3、BF16+同一center128两臂，每臂两个既有拍手seed。检验移除主干量化后是否出现明确收益，不预设此前阴性由SVD掩盖造成。

四条新视频均复用E038真实noise/embedding、原20步CFG1、1024×576/124帧及原video/audio VAE。BF16参考复用E073。200目标Linear均原torch.nn.Linear、BF16权重，未安装SVDQuant；80次DiT逐步均为0 scaled_mm、52 SDPA、50 native attention、0 disk_load。源、输入、时间表、SDPA环境及解码合同核验全部通过。

center沿用E082已验证的原生group128版本，只改变attention内V表示和均值补偿；没有新kernel、校准或层步筛选。三路自由生成的QKV会随轨迹改变，因此此处是端到端方法比较，不是固定PV误差贡献分解。

## “降低40%”究竟指什么

此前40.24%／42.07%来自E081：两拍手seed，固定s14/block24，72个query、全部56heads及完整key序列，**group32、保持同一份量化QK，比较PV输出相对同QK精确PV的平方误差和（SSE）**。

`改善率 = 1 − SSE(center) / SSE(original)`；精确参考是 `softmax(quantized-QK logits) × V`。

不包含QK量化造成的误差，也不包含SVD主干、整模输出或最终视频误差。SSE减少40%约等于L2距离减少22.5%。实际视频使用group128：同位置局部PV SSE降低约37%–40%；E082原生实现计入QK误差后，固定QKV的完整attention SSE降低14.42%／15.04%。这些仍都是原SVD轨迹上的局部读出，不能充当E083的BF16轨迹结果。

V去均值、量化残差再恢复均值属于[VC-Attention的V-Smooth](https://arxiv.org/html/2609.15810v1)已有组成。当前没有复现其V聚类与K/V重排，不能叫完整VC复现；普通接入Sage3也不构成新方法。之前C007双侧差分候选是另一件事，已因未过强对照门槛停止。

## 结果／结论

以下均为全124帧、原1024×576 RGB[0,1]，对同一完整BF16参考的配对距离；箭头为 **BF16+原Sage3 → BF16+center128**：

| 样例 | LPIPS-Alex | L1／MAE | L2／RMSE |
|---|---:|---:|---:|
| 拍手 seed0 | 0.16658 → 0.17749 | 0.03504 → 0.03594 | 0.06604 → 0.06899 |
| 拍手 seed1 | 0.34304 → 0.36946 | 0.12885 → 0.14014 | 0.21546 → 0.20717 |

两例LPIPS、MAE均增大；RMSE一升一降。这些是输出距离，不是画质百分比。特别是seed1服装、桌面、手表、取景及手势朝向发生改变，不能把距离增大直接解释为画质恶化。

先看图后读指标：root查看全部8页、每例固定32帧；另一代理独立查看seed1四页。seed0仍见手指叠边/模糊，未见跨帧稳定修复；seed1两臂均有清楚轮廓和运动模糊帧，center未稳定优于原Sage3。均保留拍手动作。BF16自身也有运动模糊，不能将所有模糊归因量化。这是有标签缩略抽帧观察，非实时完整播放或独立人评，未评价音质。

**决策：** 两例未支持“中心化在BF16主干下即可带来明确细节修复”，因此不能仅以SVD主干混杂解释此前缺乏收益；这不排除具体精度交互，也不否定其他场景或完整VC。保留中心化作为已知局部阳性对照，不升级主baseline，不扩group/层/步网格。后续若继续方法研究，需要先提出针对残留缺陷的新、可区分预测，不能把局部SSE下降本身作为创新或扩实验依据。

## 交付与核验

- [并排视频入口](/data1/models/svdquant-wjq/research/20261005/E083/review/index.html)：左完整BF16+SDPA，中BF16+Sage3，右BF16+center128。并排片静音，原片保留音轨。
- [seed0并排视频](/data1/models/svdquant-wjq/research/20261005/E083/review/vbench0161_r0_triplet.mp4) · [seed1并排视频](/data1/models/svdquant-wjq/research/20261005/E083/review/vbench0161_r1_triplet.mp4)
- [实验计划](../06_experiments/E083_bf16_attention_comparison_plan.md) · [输入与配置](../06_experiments/E083_bf16_attention_video_manifest.json)
- [指标与独立归约](../../results/research/E083/video_summary.json) · [逐帧指标](../../results/research/E083/metrics.json)
- [root观察](../06_experiments/E083_root_visual_review.md) · [seed1独立观察](../06_experiments/E083_seed1_visual_review.md)

六对视频指标共744对帧；独立stdlib归约与六源媒体SHA核验通过，最大归一差2.78e−17，未重复运行LPIPS网络。四条视频共80 DiT、4 video+4 audio VAE。原Sage3采样127.15/129.05秒，center141.77/142.81秒；主干峰值43.03/44.24GiB。两GPU从启动到全部解码完成分别328.03/331.17秒，均在原1800秒内；无运行失败。GPU0/1及评估进程均结束，没有后台研究GPU任务。
