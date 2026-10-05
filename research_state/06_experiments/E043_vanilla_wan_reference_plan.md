# E043：原版Wan2.1完整生成参照

2026-10-03，exploration / baseline readiness。上一轮E041/E042是有效progress，但没有形成论文claim。E022 rCM共同BF16失败且原TE完整发布入口未复现，E024是另一蒸馏量化产品；因此不能把它们当原版Wan能力的上限，也不适合继续据异常视频设计量化loss。

问题与假设：在不使用rCM/FastWan/DeepCompressor包装的官方WanPipeline路径中，原版非蒸馏Wan1.3B是否能在固定公开动作上产生可读的完整媒体，而不是持续大块条纹/结构丢失？这是下一轮选择teacher与量化归因入口的必要事实，不是新方法或跨模型量化因果实验。

最小有效设置：全部E038预先使用的四个公开动作×两个新seed，共8段；480×832、81帧、16fps、50步UniPC、CFG6、flow_shift8、原版Transformer/UMT5/tokenizer/WanVAE，官方negative，不扩写prompt。采样采用Wan官方README对1.3B的guide6/shift8推荐；本地checkpoint scheduler原配置flow_shift3须显式override并保存实际config，不默认为rCM四步。Diffusers0.40原入口保持FP32 latent/update，DiT/TE BF16、VAE FP32；只对DiT强制torch FLASH SDPA。无需换数学模型来追求逐位一致。

避免的无效代理：单层NMSE、小latent、少步smoke、只展示最好seed、重新从旧rCM checkpoint导出却称原版、用不同行为题的奖励差称量化因果。所有8条保留，图片与评分全量记录，读出不足单列unknown，不按任意比率gate否定模型能力。

必要产物：实际采样config/条件/初始噪声/终态latent、逐视频100 DiT/50 scheduler/1公开decode计数与时间、全部媒体、固定9帧sheet和MJ主四项/既有时序指标。初始噪声由各自CPU seed生成FP32并保存，后续如做同模型量化必须复用实际输入。TE正/负embedding每worker可缓存，仅相同prompt两seed复用。

决策：若原版参照没有此前普遍结构异常且能读出具体动作，后续量化研究优先使用此基础模型并做实际native配对；不据一组视频宣称全域质量优秀。若异常仍广泛存在，保留所有样例，核实际checkpoint/完整官方pipeline的具体差异，停止扩量化训练而不是扫prompt/decoder参数。若部分动作失败，逐题记录、不要后选成功题假装所有动作合格；先判断新的机制问题是否真的需要该失败能力。

资源/停止条件：GPU0–3每卡一个固定prompt两seed，实际启动前查空闲；命名tmux+launcher日志/进程组超时，单worker1500秒、总1800秒。8×100=800 DiT、400 scheduler、8完整decode，0训练/量化。OOM/非有限/模型加载错误保留真实失败并停止相应进程，不把未完成当阴性结果。源执行后不改，修复另版本；大文件与cache全在DATA1。当前不启动原生量化臂或新增loss。
