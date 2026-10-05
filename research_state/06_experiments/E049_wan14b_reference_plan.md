# E049 — 原版Wan14B完整BF16参照

2026-10-03。Exploration / necessary baseline evidence。E048匹配校准后1.3B仍有显著手/衣碎片和提示间得失；不能把该小模型结果推广到14B。此实验准备更大原版非蒸馏模型的完整参照，本身不是方法创新或模型规模单因素实验。

**问题与假设。** 在固定官方14B Diffusers采样参数下，原版teacher能否产生可用于后续同模型量化对照的完整视频？八个原有诊断样例是否有共同底座异常，具体动作和几何有哪些实际局限？不要求teacher所有动作完美，不设MJ数值准入门槛。

**最小有效设置。** 固定HF revision38ec498cb3208fb688890f8cc7e94ede2cbd7f68原版T2V14B transformer、同身份FP32 VAE；复用E043四提示×两seed八份真实FP32初噪及BF16正负embedding，不重新抽噪、不载TE/tokenizer。81帧480×832、50步UniPC、CFG5、shift3、FP32 latent更新、BF16 DiT/FlashSDPA，完整40层/CFG/decode。README明确分辨率/帧数/CFG5，scheduler明确shift3，50步显式沿pipeline默认。16fps是本项目播放/评价约定，官方示例15fps。与E043同时改变模型、CFG和schedule，非纯尺寸因果比较。

**无效代理。** 单层/短步smoke、任意噪声MSE、以1.3B代14B、只看最佳seed，均不能替代完整参照。八例已反复观察，不称新held-out；保留失败和较差案例。

**执行与预算。** download/supervisor均complete、rc0，官方分片身份及完整header/index核验后，运行隐藏CUDA的CPU输入/源/配置检查，再启动named tmux。GPU0/1/2/3各一个prompt双seed；忙卡在总预算内等待，绝不抢占。worker5400秒、launcher6600秒，失败保留并停所属进程组，不自动重试或改配方。计划8视频/800DiT/64000SDPA/400scheduler/8公共VAEdecode/0TE/0nativeGEMM。保存每DiT计数、50步标量链、初态来源/完整末态、81帧媒体和固定九帧图；大数据均DATA1。峰值与时间实测，预算不是速度预测或benchmark。

**评价与决策。** 新八媒体沿现有MJ四主项及AMT/RAFT/DINO，不重跑旧模型；固定九帧先于root读取分数检查，动作结论须全帧证据。如果有共同底座异常，定位原版加载/采样/decoder并保留协议/结果；如果生成可用，保留teacher全部局限，准备同14B有效NVFP4强基线。资产或BF16完成不算量化成果。动作含混时不挑seed重生成、不用严格语义gate无限推迟量化对照，而把比较限定于可辨视觉/内容；无论何结果，不自动启动PTQ网格或新loss。

[manifest](E049_wan14b_reference_manifest.json)与[入口说明](../00_state/wan14b_reference_entry.md)给出来源与精确配置。先前一次协议写入命令因尚未落到目标目录的index文件路径而停止，未写协议/运行任何模型；最终协议的权重/index身份由完成下载收据绑定，已有小配置单独固定。
