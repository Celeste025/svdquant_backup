# E079：新工作基线Sage3＋SVDQuant的可见质量检查

用户明确将官方SageAttention3＋现有SVDQuant设为工作baseline。保留完整BF16和仅SVDQuant用于比较，不改既有实验标签。新baseline具体为E009 200个原生NVFP4 W4A4线性层、BF16 rank32，加E078官方Sage3 per_block_mean=True主50block有效长段；refiner/padding依旧BF16，QK norm/RoPE保留。

问题：组合是否比原仅SVDQuant出现可辨的动作/形态/时序质量退化，同时报告相对完整BF16的情况。固定复用E038/E073四动作×两seed全部8例，不看结果选seed；拍手、叠衣、汽车转弯、大象喷水。实际prepare embedding/noise、原20步CFG1/flow12,3/576×1024/124帧/24fps/原BF16 VAE均匹配。只生成8条组合视频、160DiT、8video＋8audio decode。两GPU0/1各4例，生成解码共用每replica1800秒、<60GiB、全部新数据<4GiB。

复用E073执行源生成独立版本，仅增加原native权重安装及复用E078官方Router。CPU验证缓存哈希与schedule、执行时检查每步200scaled_mm/52SDPA/50Sage调用，8例均完整再评价。运行失败保留，不悄悄用FlashInfer替代Sage。

观察按每例三臂分别记录动作可辨性、手/衣物/刚体轮廓、重复/混融、持续性；不同构图或动作相位本身不算退化。模型只能使用顺序联系图，明确样本覆盖，不称原速完整观看或独立人评。提供同帧率三列并排mp4供用户播放；原始带音视频保留。模型主观察静音，不评价音质。

指标全124帧原尺寸RGB[0,1] LPIPS-Alex、MAE、RMSE，组合对BF16和仅SVD分别计算，旧SVD/BF16距离复用E074已核验8例。它们量化偏离而非质量百分比，不使用MJ替代可见证据。单例清晰新缺陷可称case-level退化；跨两个seed重复再称该动作一致现象；混合则如实报，不能声称全面无损或显著性。

材料放results/research/E079及DATA1/20261004/E079，简短报告写research_state/reports/077_20261004_h3_sage3_video_quality.md。此为用户授权的baseline评价，不作为新算法贡献，不扩排列组合。
