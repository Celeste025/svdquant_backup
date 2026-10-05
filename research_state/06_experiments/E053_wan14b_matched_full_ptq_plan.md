# E053：原版Wan14B完整匹配SVDQuant PTQ

2026-10-04，Exploration / strong baseline。E050完整14轨迹/64cache及独立汇总完成；E051未校准plain八例质量得失不均。E052完整首block资源pilot完成1647.21秒，真实覆盖全部10主Linear、两次各64激活记录；smooth 963.08秒、LR336.56秒。GPUallocated最大61.865GiB、CPU RSS最大367.462GiB。此证据支持安排单卡完整执行，但不证明其余39层和最终导出无更高峰值，也不线性外推总耗时。

**问题：** 当前原版14B是否能以本模型匹配数据完成完整NVFP4 SVDQuant强基线？本实验只生成完整checkpoint，质量要在后续真实native同八例对照中回答。不是新量化方法。

**完整配方与数据：** 固定官方14B模型、E050 CFG5/shift3/81帧480×832/50步UniPC自身轨迹的64条记录；原Random0政策、35cond/29uncond、缺0/48/49及成对状态少的局限保留。64样本/batch4/sample_size−1/g10（本配置实际19平滑候选）/rank32/LR最多100次及原早停/OutputsError，两级FP32 global与E4M3 local/group16/E2M1，禁历史gated，保留Diffusers0.40 RoPE兼容和FlashSDPA。

**执行：** 单worker/GPU4，全部40层400主Linear、完整BF16模型驻留GPU；调用原ptq全smooth→全LR→权重量化/五文件导出/动态输入hook安装。每阶段完整loader，不截断或挑层，不载E052partial，不迁移1.3B中间状态。沿原seed0顺序流，不引入per-layer RNG政策或分层并行。五产物model/scale/wgts/branch/smooth.pt与完整来源记录保存到DATA1；保留fresh_cache，链接不可单独丢弃。

**资源决策：** 367.5GiB已见CPU峰使两独立worker峰叠加接近742GiB整机，且未计其他进程，因此当前选择单worker。保留原loader的主机使用率>90%保护。worker最多36小时129600秒，父监督129900秒，GPU初始空闲等待≤900秒；这是宽松运行上限，不是预测耗时。named tmux、独立日志、两次空闲观察及启动前即时检查，只清理自己的子进程，GPU6/7不动。

**成功与失败：** 成功需原完整ptq返回、400/40实际配置与全部阶段正常结束、五文件保存并产生checkpoint_identity；后续安装前再检查完整400目标/280共享LR组及真实native roundtrip。任何OOM/主机内存保护/超时/异常保留原始日志、阶段记录和目录，不缩样本/搜索或静默重试。完成后再开展原八例native比较；不按plain最差样例修改集合，不自动把checkpoint称为质量修复。

**检查：** 先隐藏CUDA运行正式phase check，检查当前模型meta40/400与完整loader配置、实际路径和原配方，不载权重/校准样本。既有准备CPU首次PATH缺ninja失败及修正环境后的成功保留；E053检查使用现成统一environment。
