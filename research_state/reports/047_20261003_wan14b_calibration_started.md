# 阶段报告047：14B匹配校准采集已启动

2026-10-03 22:49（上海）。**E050正在采集14B自己的完整轨迹校准记录；当前没有14B SVDQuant checkpoint或新的量化质量结论。** 最近完整结果仍是[046：原版14B参考](046_20261003_wan14b_reference_results.md)。

保留E047的固定Random0政策：14提示、64选中记录、35 cond/29 uncond，35个时间位置且缺0/48/49；不为这轮另开采样网格。改用当前原版14B的CFG5/shift3、81帧480×832/50步UniPC，完整运行每条14B轨迹；只复用身份已经核验相同的文本embedding，0文本编码/0视频解码，不复用旧1.3B中间latent或输出。

正式CPUcheck通过，0.49秒、CUDA未初始化、未加载权重。named tmux `e050_wan14b_calibration` 下四worker在GPU2–5实际运行；22:48:49各自已记录首提示10/50步，部分worker又保存了后续选中cache。当前保存6条选中记录，尚无完整轨迹；1400DiT/112000SDPA/700scheduler/64cache是总计划，不是完成计数。每worker90分钟、总监督120分钟上限，完成后自动CPU独立汇总，不自动启动PTQ，不重复启动原任务。

**后续决策：** 完整缓存到位后，保持64样本、batch4、g10、rank32和LR最多100次/原早停，测首个真实block的完整校准资源成本。首block不能代表全模质量或全部峰值，也不以单层时间乘40承诺总耗时。原PTQ先全模smooth再全模LR，局部pilot必须在平滑后重新采集激活；部分cache不能直接冒充完整checkpoint。

并行准备E051未训练plain原生对照，使用E049同模型/真实初噪/embedding和采样，逐层pack400主Linear避免整模FP32 master显存峰值；将用GPU0/1，当前仍为代码准备、尚未GPU执行。该plain只提供未校准对照，不能替代SVDQuant强基线。两项都是研究底座，不是新方法或顶会成果。

[采集协议](../06_experiments/E050_wan14b_matched_calibration_plan.md) · [运行记录](../../results/research/E050/collect_launcher.json) · [plain协议](../06_experiments/E051_wan14b_plain_native_plan.md)。运行状态以实际PID/日志为准，本页为上述时刻快照。

**22:55后续执行更新：** E051已通过正式CPUcheck（0.3753秒/无CUDA初始化/400meta目标/八真实输入），原单次chain在GPU0/1启动prompt161/192，另外两提示排同卡队列。两worker已实际加载14B分片，尚无完整plain视频，不报质量结论。生成后自动新视频评价和CPU配对汇总，不重跑BF16。E050仍正常采集，22:55已保存15条选中cache，四首提示均有30/50步记录，完整轨迹尚0。两个原任务均应继续，不手动重启。

**22:58实现进展：** 两个E051 worker真实存活，首例均完成10/50步，运行时每DiT的400个原生GEMM、400次activation pack及80次BF16 SDPA核验均通过。由此确认逐层安装可在当前单卡完成真实40层采样前向；仍不代表完整视频质量、SVD强基线或优化后速度。

**23:17阶段更新：** 校准采集已完成8/14完整轨迹，保存45/64选中记录，四worker正常继续。Plain已完成拍手/叠衣各一条（2/8），root先查看并保存[视觉观察](../06_experiments/E051_visual_observations.md)：主体仍可辨，叠衣手与布料局部融合，不能据两例宣称等质量。两worker400模块转换实际0.96–2.20秒，allocated从27.242降至8.405GiB，转换峰值27.354GiB；仅此安装资源，不是完整推理峰值或速度benchmark。

E052首block资源测量已通过正式CPUcheck（5.35秒、0CUDA/0权重），[完整协议](../06_experiments/E052_wan14b_ptq_resource_plan.md)。监督器2653260实际存活并等待E050全量结束，尚无GPUworker；随后只运行保留全部样本/搜索的block0测量，不自动整模PTQ。两轮cache均核64实际数据、16次前处理/32次block0/后39层0次，且LR使用平滑后新cache。当前无新增14B质量结论或科研claim。

**23:29更新：** Plain拍手/叠衣双seed已4/8完整，预览均已记录，尚未读取评分。手指与布边局部融合/模糊，主体结构保留；没有根据这些中间结果调整样例。原队列已自动转到汽车/大象，各首seed30/50步。E050已55/64选中cache，但完整轨迹仍8/14，须等其完整结束与汇总；E052原监督继续等待。

**23:34新增证据：** Plain已6/8完整，前六条及各自BF16预览全部先读图记录；最后汽车/大象r1继续。前四条[实际运行成本](../06_experiments/E051_runtime_cost_notes.md)：完整pipeline 911–932→726–727秒，四时长和减少21.14%；推理allocated峰40.035→21.197GiB，但加载/转换所见峰为27.354GiB，reserved48.203GiB。计时含诊断、非预热重复/受控并发benchmark，质量结果尚未读；plain不是SVDQuant强基线。E050在23:31已有12/14完整轨迹、56/64cache，GPU4/5原worker已退出，GPU2/3继续最后两提示；E052仍等待全量结束。

**已完成归档：** E050/E051完整结果和E052真实GPU启动见[报告048](048_20261003_wan14b_plain_results.md)，本页保留运行过程快照。
