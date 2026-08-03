# rCM-Wan 4-step 的 SVDQuant W4A4 实验

## 配置

- 模型：`TurboWan2.1-T2V-1.3B-480P.pth`，rCM 蒸馏，4-step，单 conditional
  分支，无 classifier-free guidance。
- 图像尺寸与长度：832×480，77 帧，16 fps。
- 配对样本：prompt 为
  `An astronaut feeding ducks on a sunny afternoon, reflection from the water.`，
  seed=44。BF16 和 W4A4 重新使用完全相同的初始噪声和逐步随机噪声。
- 量化：300 个 Transformer Linear 全部采用 W4A4，group size=64；
  对平滑后权重减去 rank=32 的低秩分支，再进行对称 signed INT4 weight
  fake quant。低秩 up/down 分支和输入为 BF16；FFN-down 激活加 0.171875
  后使用 unsigned INT4，其余激活使用动态 per-token/per-group signed INT4。
- 本实验是 fake quant，用于数值和生成质量验证，不代表真 INT4 kernel 速度。

## 1. BF16 基线与显存配置

复用了 TurboDiffusion 下载好的 T5、VAE 和 rCM checkpoint。原官方脚本会先加载
VAE 再加载 T5，在 24 GB 显存上产生峰值 OOM；已调整为先计算并保存 T5
embedding、卸载 T5，再加载 VAE。BF16 的 4 个 DiT step 用时约 8.2 秒（不含
模型/T5/VAE 加载与 VAE decode），成功生成 77 帧视频。

- BF16 基线：`paired_seed44/bf16_seed44.mp4`
- 初次独立基线：`bf16_seed44.mp4`

## 2. 扩大校准集并执行原版 SVDQuant 搜索

使用 VBench 子集中的 8 个 prompt；每个 prompt 覆盖 rCM 的全部 4 个 timestep，
每次模型调用在每个共享输入组上均匀抽取 2048 个时空 token。因此每组共有
8×4×2048=65,536 个 token。30 个 block 按共享输入关系拆为每 block 7 组，
共 210 组、300 个 Linear；原始 BF16 激活 memmap 共 62.34 GiB。

对每组执行平滑系数网格搜索、rank-32 低秩权重残差分离、W4A4 输出误差搜索和
迭代修正。校准集上的局部线性组输出 NMSE：

- 210 组平均：0.922%
- 最大：2.736%，位于 block 11 的 self-attention output projection
- 较大的局部误差集中在 block 9–17 的 self-attention output projection

数据与结果：

- 校准 manifest：`calibration/manifest.json`
- 激活数据：`calibration/group_*.bf16.mmap`
- 量化状态：`calibrated/shard_*_of_06.pt`
- 每组搜索记录：`calibrated/shard_*_of_06.json`
- 局部误差图：`paired_seed44/calibration_local_nmse.png`

## 3. BF16 与 W4A4 的严格配对 4-step 误差

| rCM step | Transformer 输出 NMSE | 更新后 latent NMSE |
|---:|---:|---:|
| 1 | 13.097% | 0.096% |
| 2 | 24.434% | 1.059% |
| 3 | 31.345% | 15.046% |
| 4 | 38.492% | 65.771% |

第一步 latent 仍由幅值很大的初始噪声主导，所以即使 Transformer 输出已相差
13.1%，更新后 latent 的归一化误差只有 0.096%。随着 rCM 仅用四个大步长快速
去噪，模型信号逐渐主导 latent，早期误差又改变后续每一步的模型输入，误差形成
闭环放大，最终 latent NMSE 达 65.77%。因此“无 CFG”只去除了 CFG 差分放大，
并没有消除少步蒸馏模型对单步预测精度和轨迹漂移的敏感性。

- W4A4 视频：`paired_seed44/w4a4_seed44.mp4`
- 完整数值：`paired_seed44/paired_metrics.json`
- 最终 latent：`paired_seed44/final_latents.pt`
- 综合图：`paired_seed44/rcm_w4a4_diagnosis.png`

## 4. 第一步逐 block 与 block 21–23 内部诊断

第一步累计 block 输出 NMSE 从 block 0 的 0.071% 上升到 block 12 的 2.087%，
在 block 17 首次跃升至 10.578%，block 21 为 15.462%，block 28 达 15.576%。
最后经过输出头后为 13.097%。这说明误差不是只在最终输出头突然出现，而是从
中段开始由残差网络逐步累积，并在若干中后段 block 出现明显跃迁。

block 21–23 的投影输出比较包含两部分：该层自身 W4A4 误差，以及此前 block
造成的输入漂移，不能解释为该 Linear 的“纯局部量化误差”。三者趋势高度一致：

- text-context 侧的 cross-attention K/V 仍只约 0.19%–0.52%，因为它们直接读取
  固定 T5 embedding，不继承视频 hidden-state 的累计漂移。
- self-attention V 为 58.6%–77.1%，self-attention O 为 48.1%–56.3%。
- cross-attention Q 为 26.9%–33.9%，cross-attention O 为 61.5%–64.3%。
- FFN-up 为 36.2%–38.6%，FFN-down 为 66.8%–75.8%。

这一对照直接表明主要问题是视频 hidden-state 路径上的误差传播和分布漂移；
固定文本 K/V 并未同步恶化。校准集上单层局部 NMSE 约 1% 并不保证闭环生成轨迹
稳定，尤其是 rCM 的四个大步长没有 50-step 模型那样多次小幅修正的余地。

## 结论

rCM-Wan 不使用显式 CFG，确实排除了 Wan 原 CFG 差分放大这一项，但当前
group64/rank32 SVDQuant W4A4 仍然明显失真，且严格数值结果比原 50-step Wan
更敏感。最可能的核心原因由“CFG 放大”转为：

1. rCM 少步采样的单步更新幅度大，单步预测误差更容易改变后续轨迹；
2. 中段 self-attention output projection 的局部量化误差偏高；
3. hidden-state 漂移进入 self-attention、cross-attention Q/O 和 FFN 后被反复
   传播，而固定文本 K/V 基本不受影响；
4. 校准目标目前是逐 Linear 的局部输出 NMSE，尚未直接约束 block 输出或完整
   四步轨迹误差。

下一步最有信息量的修复方向是优先保护 block 9–17 的 self-attention O，以及
block 17 以后的视频 hidden-state 路径（例如这些层保留 A8/BF16，或加入
block/trajectory-level reconstruction），而不是继续单纯增加同类校准 token。
