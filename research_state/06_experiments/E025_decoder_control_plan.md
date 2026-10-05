# E025：同 final latent 的完整 Wan VAE 对照

2026-10-03，生成前固定。目的只是在 E024 官方公开 QAD 产品上分离 decoder stack 的影响；不把该诊断称为量化创新。

固定 E024 suite_run.json 的全部 16 条 final_latents，各为 [1,16,21,60,104]；沿用全部 prompt/seed/顺序，不根据图像或评分筛选。原 TAEHV FP16 视频已完整保存。新臂使用与官方模型包相同的 Wan VAE 资产，由本地已验证的 Diffusers AutoencoderKLWan FP32 路径解码，按 VAE config 执行一次逐通道 z*std+mean；core128/halo0 实际覆盖完整60×104 latent平面。视频为81帧480×83216fps。零次 DiT，不加载文本编码器或变更 final latent，不声称与 FastVideo BF16 VAE 逐位等价。

预算：GPU0解码最多600秒；大工件均在 DATA1，结果JSON在 results/research/E025。逐条记录实际输入、归一化合同、float输出有限性/裁剪比例、视频、时间与显存。所有失败及旧E024输出保留。完成后GPU1运行已有 AMT/RAFT/DINO、GPU5运行 MJ-VIDEO，同E024公式/抽帧/全部16样本，评价另有硬时限。查看全部8个replica0的固定五帧配对图；抽帧观察不冒充完整运动人工评价。

决策：若核心语义或运动恢复，E024现象首先归因decoder配置差异，撤回由其直接提出量化损伤的依据；若没有恢复，结束此VAE排错线，不追加dtype/分块网格。两种结果都不自动支持新量化方法。若只有锐度/纹理变化，分别报告，不用总reward掩盖语义/运动。完整采样历史核查另行收尾，不能据训练scenario静默改DMD。

启动前资源变化：GPU2–4被其他任务占用，MJ分配改空闲GPU5；temporal用GPU1。唯一实际监督器 scripts/research/launch_e025_decode.py，tmux e025_decode。
