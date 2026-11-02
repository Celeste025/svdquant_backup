# E137 — H3 误差轴向位置的闭环判别

- 模式：探索/机制判别；预算不超过约两小时，最多 4 个完整 generation arms。
- 问题：E106 保留全部误差数值但全局打乱坐标后大幅改变生成轨迹；该实验无法区分空间与视频时间结构。
- 假设：在同一 H3 W4A4 SVDQuant 误差向量多重集和逐步 NMSE 下，破坏空间坐标关系或视频帧坐标关系的效果不同；若两者都无效，E106 的现象不支持简单误差拓扑主张。
- 最小有效设置：E101 的赛车 prompt 16/seed 20026 与拍手 prompt 161/seed 38001610，576x1024、124 帧、20 steps、CFG 1、同 rank-32 动态 scale state。每个 arm 在其自身闭环输入上运行 BF16 与 SVDQuant 两个完整 DiT。
- 空间臂：video error reshape 为 `[T=37,H=18,W=32,C=96]`，每步以一个跨所有 37 帧共享的 HxW 随机置换重排整条 96 维 token-error vector；因此保留每条误差向量随帧的演化，破坏它在图像空间的坐标关联。audio error 不改。
- 时间臂：每步以一个跨所有 HxW 位置共享的 37 帧置换重排整帧误差；保留每帧的空间纹理，破坏它和视频时间坐标的关联。audio error 不改。
- 数值门槛：每 video/audio step 的 BF16-cast 后 MSE 相对 paired W4A4 MSE 偏差不超过 0.5%；video BF16 rounding 引入误差不超过目标误差能量的 0.5%；所有 20 step 输出有限。完整 token error vector 在 FP32 操作前/后保持多重集一致。
- 决策：若至少一个轴向 arm 在两个 case 都显著降低相对 BF16 的最终 latent/RGB 距离，且另一轴没有同样效果，则进入更大 prompt/seed 的机制验证，并尝试与该轴对齐的可部署 rounding/codebook rule。若两轴都接近原 SVDQuant、或效果矛盾，则停止“简单时空误差拓扑”方法路线；若两轴都有效，只能说明位置而非边际分布重要，需做更多正交对照。
- 非法 proxy：不能重放保存的 BF16 trajectory errors；不能只跑单个 step、不同 seed/checkpoint/scheduler，不能用 RGB NMSE 单独声称质量提升。
- 输出：每 arm 的 20-step JSON、terminal latent、124 帧含音频 MP4、以及后续统一汇总。原 SVDQuant/BF16/E106 全局置换作为既有参照。
