# 阶段043：匹配校准已启动，尚无新质量结论

2026-10-03。E047已经实际运行；截至本次记录，**12/14条完整校准轨迹完成**。仍在运行的worker为642354（GPU5），已通过进程状态核实。权重PTQ尚未启动，不能用采集进度推断质量改善。

本轮回答一个必要的基线问题：旧33帧、不同采样时间表的SVDQuant校准局限，在匹配当前81帧/UniPC50/CFG6/shift8后是否仍存在？保留rank32/group16/g10/64records/OutputsError及原生两级scale，直接使用原版Wan权重，不载rCM；不同时扫rank或改损失。

独立核对旧真实1600文件名、虚拟1600及Random0抽取名单一致。新实验只保存同一64位置，实际覆盖14提示、35个时间步，cond/uncond35/29，未选step0/48/49；原fast抽样局限明确保留，不事后加末步样例。所选提示不与当前四个诊断文本重叠。帧数和schedule同时改变，因此后续结果也不能隔离单因素因果。

采集在named tmux `e047_wan_matched_collect`持续执行，只保存真实模型I/O、初态/终态/embedding和标量步骤，不进行VAE解码。计划总1400DiT/700scheduler/28TE/0decode/64cache；**计划总数不是本次已完成数量**。启动前GPU2–4被其他任务占用，四固定分组改排GPU0/1/5，数据及worker不变。采集worker1800秒/launcher2100秒上限，失败不会自动重跑。

PTQ入口已经一次CPU预检，确认300个主干Linear和原固定配方。当前Diffusers0.40需要对仓库历史gated扩展的RoPE缓存接口作局部兼容；普通SVDQuant关闭该扩展，保留原版cos/sin tuple注意力重放，没有修改vendored源。旧完整校准约两小时，新81帧成本未知，预定六小时上限。所有64缓存完成并通过独立CPU核验后，才在重新确认空闲的GPU启动。

后续E048八例完整native生成入口已准备，只复用E043实际噪声/条件与E044生成逻辑，替换E047 checkpoint；尚未加载新权重或运行。完成后比较逐例视觉、MJ四项及AMT/RAFT/DINO，与旧SVD、BF16及末步BF16强控制对照。

目前无新方法、等质量优化或可投稿核心贡献。最新运行状态以[采集监督记录](../../results/research/E047/collect_launcher.json)和实际进程为准；本报告是时间点快照，不代表任务持续存活。来源：[E047计划](../06_experiments/E047_wan_matched_calibration_plan.md)、[抽样核验](../../results/research/E047/selection_audit.json)、[PTQ预检](../../results/research/E047/ptq_check.json)、[E048计划](../06_experiments/E048_matched_wan_native_plan.md)。


## 后续更新：采集核验完成，PTQ已实际启动

15:46左右，14/14采集全部complete，四进程rc0退出，wall1061.91秒。独立CPU核验3.93秒complete/CUDA未初始化：64实际缓存的分支/时间点/embedding和14初始噪声重放通过；总1400DiT/84000SDPA/700scheduler/28TE/0decode。[汇总](../../results/research/E047/collection_summary.json) SHA0644642126bd3c4a2c9d685a2a54a5177030adae48c35cc61b1829daa0d1a611。

GPU0的PTQ worker818406已载入原版DiT并进入真实smoothing激活收集，named tmux `e047_wan_matched_ptq`，六小时上限。[监督记录](../../results/research/E047/ptq_launcher.json)和[运行记录](../../results/research/E047/ptq_run.json)持续落盘。完整checkpoint与质量对照仍未完成，不将正常启动当作校准有效的证据。


## 16:12 执行衔接

PTQ已完成前三个block的smoothing，进入blocks.3；实际进程818406与日志持续推进，尚无新checkpoint/质量结果。后续八例的顺序监督器已在 `e048_after_ptq` 启动（PID1173926），实际处于等待、stages=[]。E047两完成收据及worker退出后，依次执行既有CPU预检、生成、评价、v2独立汇总；评价前等待GPU0/1空闲最多30分钟，各计算阶段沿用原预算，失败即停止并保留证据。见[监督记录](../../results/research/E048/chain_launcher.json)。不应另行重复手动启动这些阶段。

有限近邻核查也已收尾：[输出相位](../01_literature/patch_phase_geometry_nearest_work.md)需排除同一子像素输出头对普通token扰动的传递；[批间scale耦合](../01_literature/batch_invariance_collision.md)已有直接机制与实现先例。两项均未启动新GPU实验，E048仍是下一科学决策的依据。


16:45 运行检查：同一 PTQ worker 已持续运行约59分钟，平滑完成9/30个block，进入blocks.9激活收集；日志持续推进。GPU0最近一次资源采样约30.5GiB显存、主机available约544GiB，未见资源压力。E048监督器仍真实存活、stages=[]，未提前生成。此为执行进度，完整校准和新视频质量结果仍待完成。


17:46 运行检查：同一worker持续运行2小时，平滑完成18/30个block，正在blocks.18；日志正常更新，E048监督器真实存活且stages=[]。主机available约542GiB、DATA1剩余约1.2TiB。当前仍是完整校准的中间阶段，没有新checkpoint或视频质量结论，原六小时外限及顺序执行协议保持不变。


18:43 运行检查：同一worker818406已运行约2小时58分钟，smoothing完成28/30个block，进入blocks.28；日志18:43:17更新，进程Rsl。E048监督PID1173926仍存活、stages=[]。18:36资源检查GPU0约30.5GiB显存、主机available约539GiB、DATA1剩余1.2TiB，GPU1/5空闲；后续启动仍由监督器重新核实资源。独立入口检查确认PTQ文件收据增加sha256字段不会触发生成入口整字典比较错误，无需改源码。尚无完整checkpoint或新视频结果，维持原配方和六小时外限。


18:55 阶段切换：日志明确记录smoothing 30/30完成，耗时3:09:08；18:55:04保存smooth scales并链接checkpoint/smooth.pt，随后进入Quantizing weights / Adding low-rank branches to weights，重新收集64条激活信息。原worker818406仍在运行，E048监督器继续等待。此时checkpoint_identity.json尚不存在，smooth.pt只是中间产物，不代表完整PTQ或质量验证完成。


19:06 预算复核：LR已完成2/30个block，进入blocks.2；原worker与E048监督器均实际存活。首块含初始化、迭代数也不同，进度条曾给出约3小时剩余；第二块后降为约1小时56分，六小时外限尚余约2小时40分。独立读取旧真实PTQ日志：旧LR含缓存49:25、后续实际weight pass 53秒、activation日志约40秒；这些旧耗时不能保证新任务耗时。目前没有明确干预预算的依据，保持原运行与截止时间，不改迭代数、样本或配方。


19:42 接近四小时检查：worker818406已运行3:57，LR完成10/30并进入blocks.10，日志正常推进；E048监督PID1173926仍存活、stages=[]。LR进度条剩余估计1:33，而原截止21:45:46尚余约2小时；估计会随早停迭代变化，不作为完成承诺。19:36资源检查GPU0约18.8GiB显存、47°C，主机available约543GiB，DATA1约1.2TiB可用，GPU1/5空闲。仍无完整checkpoint或新增视频质量结果。并行启动的[官方14B资产准备](../00_state/larger_video_model_asset_entry.md)不占GPU、不改变本实验配方，当前仅有部分下载，非新跨模型实验结果。


20:45 五小时附近检查：原worker818406与监督818402实际存活，elapsed4:59；LR完成25/30并进入blocks.25，日志正常。进度条估计LR剩余19:19，原截止21:45:46约余1小时；尚需完整weight/activation导出与完成收据，未将LR完成比例当全PTQ完成比例。E048监督1173926仍存活、stages=[]。20:35资源检查主机available约542GiB、DATA1约1.2TiB，GPU1/5空闲。并行14B资产已有6/12权重分片通过官方SHA256，尚未完成；这些执行进度不构成质量结果。


## 21:18 执行完成记录：E047完整PTQ完成，E048六例已生成

E047的[ptq_run](../../results/research/E047/ptq_run.json)与[ptq_launcher](../../results/research/E047/ptq_launcher.json)均complete、launcher rc0，worker818406/监督818402已退出。实际worker19228.123446秒、launcher19239.203601秒，GPU0峰值allocated25.409592GiB、reserved29.904297GiB。五产物合计3,273,805,991B；[checkpoint_identity](../../results/research/E047/checkpoint_identity.json)完整，branch/smooth为本次fresh_cache内链接。独立只读核验确认五文件路径/解析路径/大小/mtime一致，原BASE、rank32/group16/g10/64records/batch4/sample_size=-1及原生两级scale符合冻结配方，9份源和3份配置的小文件SHA、manifest与采集绑定一致。权重内容SHA继承生产者；没有重新哈希GB权重、载入张量或使用GPU。此前所有“PTQ执行中”段落为对应时刻的历史记录，现由此完成记录更新。

E048原自动chain完成隐藏CUDA预检，于21:07:01进入生成。21:18:35根节点现场确认prompts161/192/269三个worker全部complete，六视频累计600DiT/180000native GEMM/36000SDPA/300scheduler/6decode；prompt316 PID1020311仍在GPU0运行。chain1173926及generation launcher902948实际live，后续生成、评价及CPU汇总继续由同一监督器衔接，不手动重复启动。此处仅记录执行进度，未读取新评分；完整科学结果仍是[042](042_20261003_terminal_repair_control.md)，不能把完整校准或部分媒体完成视为质量改善。

并行Wan14B资产监督3654981/worker3654983仍live，8/12分片已verified，完成文件加partial共41,821,043,967B，原deadline22:18:36不变。尚未完成全部头部/index及可用性核验，无新增GPU实验或质量结论。
