# E045：原版 Wan 同状态末步误差与 decoder 响应

2026-10-03，探索性机制定位。E044 已完成完整原版 native 对照：SVD 相对普通 FP4 恢复多数读出，但相对 BF16 的 fineness 八例全降，拍手等例可见碎片形变。尚无论文级 claim。本实验不构造方法，不追求分数优化。

问题：在完全相同 teacher 末步状态，真实 SVDQuant 预测误差集中在哪里，固定 VAE 对它的时空响应是什么？选择已有拍手两 seed，全部保留；这是看过 E044 后的诊断选择，不是 held-out 验证。不能据此解释所有自由生成的损伤。

执行：复用 E043 实际 FP32 noise 与 BF16 正负 embedding；原版 BF16 DiT 完成50步 UniPC，81帧/480×832/CFG6/shift8，FP32 sampler state，原 pipeline output_type=latent，无 TE。在 step_index=49、scheduler.step 之前保存实际 sample、两支 exact DiT inputs/outputs、实际 BF16 CFG、完整 UniPC 状态。官方最终 teacher latent 为 B。按原 E044v2 原版非 rCM SVD 配方安装 native，使用同一 DiT 输入/条件/时刻重算 cond/uncond，保持 BF16 的原 CFG 操作顺序；完整复制预末步状态，分别执行一次 BF16 shadow 与 native scheduler.step，得到 Q。复制时保留 tensor 原设备，特别是 CPU sigmas，避免实例包装函数闭包复用。

注意：当前 UniPC 最后 sigma .146347→0、predict_x0=true、lower_order_final=true，末步降一阶。有限数值下输出退化为当前 x0 估计，不能把本实验叫作多步历史放大检测。仍执行原 scheduler，不手写等价公式。

四角：B、Q、B.clone 后仅 latent t0 拷 Q 的 first_only、B.clone 后 t1: 拷 Q 的 rest_only。使用真实值替换，不以加减 delta 重建，不放大、旋转或拟合扰动。每角用同一官方 FP32 VAE、一次原 mean/std denorm、无 tiling/slicing 解码。公共 VAE 内部已 clamp[-1,1]；通过 decoder forward hook 保留21块裁剪前输出，按官方拼接/必要 unpatchify，另验证其 clamp 与公共返回的对应。每角保存 preclamp raw tensor、latent、81帧媒体和9帧图。

读出：cond/uncond/CFG 的21时间片误差、真实末步 latent 差（diffusion 与 VAE denorm 两空间）、81帧裁剪前/后 RGB 差、四角 interaction Q−first−rest+B。按每帧均方能量比较，不能拿首片1帧与后片4帧总和比较；latent→RGB比率仍混入方向和空间分布，不是通用 decoder gain。第一 latent 输出首帧，后续各输出4帧是输出调度，不代表因果 receptive field 独立。CPU 分块 FP64 做差统计，保存完整原始 tensor 支持复核。局部反事实不跑 MJ/VBench 评分，不将它们当新生成质量基准。

决策：若单个实际末步已经产生可读碎片且四角显示相应跨时响应，继续定位这个具体作用；若响应主要随输入误差分布，优先追踪量化输出；若没有重现，只否定“单末步足够”，不排除全轨迹或 decoder。无任意百分比 gate。SSVAE 已覆盖通用 latent-noise decoder 训练，IV-VAE 已覆盖因果 VAE 重建帧不均，不把这些常识改名为贡献。

预算：两卡0/1，一卡一 seed；每例100 BF16 DiT+2 native DiT、600 native主GEMM、6120 DiT SDPA、52 scheduler、4 VAE decode；总204 DiT/1200 nativeGEMM/12240 SDPA/104 scheduler/8 decode/0 TE/0训练。worker900秒，总1200秒，启动前重查卡，独立进程组和期限，仅清理己方任务。大文件/缓存均 DATA1，源执行后冻结，失败与修复版本保留。
