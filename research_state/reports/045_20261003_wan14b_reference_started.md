# 阶段报告045：原版Wan14B参照已开始完整生成

2026-10-03 22:09（上海）。E048确认1.3B匹配校准仍有明显失真后，下一步先建立原版更大模型的实际参照。**官方14B资产已完整核验，四个生成worker均完成首条视频10/50步；当前还没有完整14B视频或质量结论。**

资产固定为Wan-AI/Wan2.1-T2V-14B-Diffusers revision38ec498cb3208fb688890f8cc7e94ede2cbd7f68。下载约2小时45分，12分片及小文件共57.154GB全部完成，下载时逐文件核官方SHA；1,095个张量、14,288,491,584元素均为F32。root另核实际文件头、索引、size和小文件SHA，未重复重哈希大权重；原下载进程已退出。[完成核验](../../results/research/asset_wan14b/completion_review.json)。

E049固定四动作×两seed八例，直接复用E043真实初噪和文本embedding；加载14B为BF16、公共VAE为FP32，不重复文本编码。采用固定官方HF Diffusers的CFG5/shift3、81帧480×832，并显式设置pipeline默认的50步UniPC；16fps是本项目播放/评价约定，官方示例15fps。与1.3B的CFG6/shift8不同，因此不能解释为模型尺寸单因素实验。

完整CPU输入检查已通过，八份实际输入、配置和来源一致，CUDA未初始化、模型未加载；随后在named tmux `e049_after_assets` 自动启动GPU0–3四worker。22:09四PID1683412/1683596/1683773/1683887真实存活，均已10/50步。最近显存快照每卡34665MiB，非峰值或速度benchmark。总计划为800DiT/64000SDPA/400scheduler/8decode/0TE/0nativeGEMM，不能把计划数当已完成数。

每worker90分钟、总生成110分钟上限。现有监督器继续串行执行完整生成→MJ四主项及AMT/RAFT/DINO→独立CPU汇总；失败停止并保留，不自动重试。先看全部固定预览再读新分数，不挑seed，不把动作不完美作为无限推迟后续量化对照的门槛；仅在完整证据范围内描述teacher局限。资产与BF16参照本身不是量化方法或论文贡献，当前仍无可投稿核心贡献。

[协议](../06_experiments/E049_wan14b_reference_plan.md) · [顺序监督记录](../../results/research/E049/chain_launcher.json) · [生成监督记录](../../results/research/E049/launcher.json) · [评价入口](../06_experiments/E049_evaluation_adapter.md)。运行中状态以真实PID及日志为准，本页是时间点快照，不代表任务持续存活。最近完整质量结论仍为[报告044](044_20261003_matched_calibration_results.md)。

执行期间的有限入口核查：[14B量化适配](../00_state/wan14b_quantization_entry.md)确认400个主干Linear及底层格式可复用，但旧整模QAD安装会同时持有原BF16图和全FP32 master，最低78.96GiB，需改为未训练plain的逐层打包/释放。SVDQuant仍须本模型匹配校准缓存和完整权重；没有新量化checkpoint或质量收益，本轮未因此增加GPU任务。
