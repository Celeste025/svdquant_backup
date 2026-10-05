# Current State

**2026-10-04 用户恢复研究并指定H3主线（D078）：** 用户已明确恢复之前目标，要求每次尝试有合理动机/可证伪假设；主要实验使用MiniMax-H3、Wan1.3B仅快速验证，E053 Wan14B完整PTQ不启动。普通伪量化/kernel修复、现成融合属于基线工程，不当创新。E054强融合通信对照停在准备状态、未GPU执行，旧C006仍关闭；当前唯一科学实验为[E055 H3同状态输出几何诊断](../06_experiments/E055_h3_output_phase_plan.md)，先CPU检验普通head解释，0模型forward，阳性也不能直接建方法。工具最近仍返回goal调度paused且无resume API，本轮按用户恢复授权实际推进，不宣称调度开关已改变。

**2026-10-04 汇报整理（D077）：** 用户要求按动机/观察→实验→结果/结论梳理。已完成[汇报简版049](../reports/049_20261004_research_brief.md)，覆盖7项主线及早期停线；列出输出patch几何/内容耦合的未证实机制候选，以及consumer表示通信的工程备选。二者都不是已成立创新，C006不恢复active；未运行新实验，14B完整SVD仍未执行。

**2026-10-04 指标纠偏（D076）：** 已核对 MJ-VIDEO-2B 实际代码、原始输出及论文。总分含安全/公平性，当前81帧只抽0/10/…/70并整幅缩放448×448。MJ降为辅助读出，不单独作为方法有效性/等质量判断，不将原始分差解释成画质百分比；旧数据保留。见[评估边界与后续原则](../01_literature/evaluation_map.md)。本轮仅答复用户问题及更新记录，无GPU实验；`get_goal`返回调度状态paused，未恢复研究任务。以下active表述为此前历史状态；E053仍无启动证据。

2026-10-03。**目标 active；当前无可投稿核心贡献、无已验证的等质量优化。** 已安装用户指定14个research skills，完成原仓库更新/8卡审计，持续记录真实结果。E047完整匹配校准、E048八例生成/评价/独立汇总全部完成。匹配校准有小幅平均收益但没有消除主要失真，最近完整结果为报告048（14B plain对照）；最近1.3B量化科学结果为报告044。

**01:28最新状态与用户纠偏：** E052已完整complete/rc0且原2653260/3055486退出；总1647.210秒，smooth963.082、LR336.563，GPUallocated峰61.865GiB、CPU RSS峰367.462GiB。E053完整40层PTQ协议84b5524d86cd01b48ca1159c9a4e7697459a404734fbe6f819b644b7595eadf5及正式CPUcheck7628739304611a1a0d2aecf4b23121d464386363f1c03511d00f57c970cad14b通过，监督器5aec1ae4ffb4cc06b0827d4c289424cb64bdb547d83e2fee84b0709a4d76ec22准备；启动命令被用户中断，01:28实际tmux/pgrep均无E053句柄，launcher/run receipt不存在，**尚未实际开始整模PTQ**。保留原single-GPU4/36h/64b4g10r32LR100协议，无重复或失败运行。用户要求研究尝试先明确motivation，见[动机与决策价值](research_motivation_after_E052.md)；不能以基线数量替代核心问题选择。

## 当前问题、贡献与证据

当前已确认的问题：原版Wan2.1-1.3B真实NVFP4 SVDQuant的手/衣碎片和几何失真，在匹配81帧/UniPC50/CFG6/shift8校准后仍保留；14B plain完整八例则显示质量得失集中、四升四降，尚无同模型SVD强对照。当前无已成立新claim、论文形态或残余贡献；基线修复不是创新。rank32符合已读官方通用默认，不能先验判为不足。

- E043：原版官方BF16八例参照完成；无旧共同条纹，但叠衣/汽车有自身语义或几何缺陷。E044：同输入plain/SVD16条新增完成；SVD较plain MJ均值+.35378，7/8提高；对原BF16细节分8/8下降。见[039](../reports/039_20261003_vanilla_wan_reference.md)、[040](../reports/040_20261003_vanilla_wan_native_pair.md)。
- E045：两例同teacher末步实际SVD输出已足以产生碎片；仅替换首latent片的后帧影响弱，不支持首片传播解释。末步UniPC降一阶，不能称多步history放大；非整轨迹损伤来源证明。见[041](../reports/041_20261003_terminal_intervention.md)。
- E046：全native轨迹只换末步BF16，8例MJtotal/alignment/fineness都提高，总分.56357→.64060；对原BF16三项各6/8仍低，视觉缺陷保留。备用BF16额外allocated2.65063GiB，非等质量加速。生成/评分/独立汇总complete、GPU进程退出。见[042](../reports/042_20261003_terminal_repair_control.md)。summary SHA e1129e856ada4029df64e9b64fc0e8145e97869cb989dfee2167f30a3cf31a72。
- [基线审查](baseline_readiness_after_E046.md)：旧实际缓存33帧，step0/25/49为t=[999,749,57]；当前81帧为[999,889,146]。旧点与collector默认shift3一致，但完整旧shift配置未直接保存。现成r64资产分别为rCM祖先或INT4 fake，不能直接替代原版NVFP4。

## E047与E048已完成

[计划](../06_experiments/E047_wan_matched_calibration_plan.md)、[manifest](../06_experiments/E047_wan_matched_calibration_manifest.json)。原版81帧/CFG6/UniPC50/shift8，保留rank32/group16/g10/64records/OutputsError/原生两级scale；batch4、sample_size=-1，不训练新loss或扩rank网格。

旧真实1600文件名与虚拟名单一致，新64按同Random0抽取；实际14提示、35时间位置、cond/uncond35/29，未含0/48/49，仅两state双支成对。原fast政策及局限保留，所选提示不与E043四题重叠（完整YAML池有重叠）。[选择审查](../../results/research/E047/selection_audit.json)与[采集汇总](../../results/research/E047/collection_summary.json)均complete：14提示/64cache，1400DiT/84000SDPA/700scheduler/28TE/0decode，采集wall1061.907233秒；CPU独立核验3.929955秒/CUDAfalse。已核64实际I/O/分支/timestep/embedding及14初噪重放，不重算DiT输出或完整轨迹。采集使用GPU0/1/5资源队列，四数据分组未变，所有采集worker已退出。

**E047完整PTQ完成：** [ptq_run](../../results/research/E047/ptq_run.json)和[ptq_launcher](../../results/research/E047/ptq_launcher.json)均complete，launcher rc0；GPU0 worker818406及监督818402已退出。worker19228.123446秒（5.341小时），launcher19239.203601秒；峰值allocated25.409592GiB、reserved29.904297GiB。五产物共3,273,805,991B（3.048969GiB），[identity](../../results/research/E047/checkpoint_identity.json)完整；branch/smooth链接指向本次E047 fresh_cache。有界独立核验确认原BASE路径、rank32/group16/g10/64records、两级scale及源/manifest/采集绑定一致，五文件路径/真实stat匹配；权重内容SHA继承生产者，本轮未重哈希GB权重或加载张量。最终配方未缩小样本/迭代预算，未改vendored源。校准完成不是质量改善证据，执行记录见[043](../reports/043_20261003_matched_calibration_started.md)。

**E048全部complete：** 隐藏CUDA检查→八例生成→评价→v2独立汇总均rc0，chain已结束。生成1034.949101秒、评价71.070493秒；实际800DiT/240000native GEMM/48000SDPA/400scheduler/8decode/0TE。沿用E043真实noise/embedding和E044原生路径，只换E047checkpoint；旧三组参照未重复推理或评分。[summary](../../results/research/E048/evaluation_summary.json) SHA570b45fc00b20f39ba768a7f1f435f9e007ab20c78a3eaff1c334266a82c2ca0。独立agent再从新原始分数复算均值及逐例升降，无不一致；四生成worker、两评分worker及chain均已由ps核实退出。全部八张九帧预览在root读新分数前查看；最后一张晚于评分计算完成，非盲评或完整事件标注，[视觉记录](../06_experiments/E048_visual_observations.md)。

**科学结果：** matched/old/BF16 MJ均值.58369/.56357/.62983；matched比old 5/8总分提高，相对BF16 6/8总分、7/8细节仍低。拍手双seed细节/连贯性下降；汽车r1 MJ .52270→.09538且RAFT由true变false，DINO却提高，不能把一致性当动作恢复。四提示MJ变化依次+.00982/+.24730/−.23035/+.05370；明显得失被平均数掩盖。保留匹配基线、旧SVD和末步BF16控制，不逐提示择优。见[044报告](../reports/044_20261003_matched_calibration_results.md)。

**E049全部complete（22:37核实）：** 原自动chain生成/评价/独立CPUsummary全部rc0，原GPU/launcher进程退出。原版14B官方HF配方CFG5/shift3/81帧480×832/50UniPC，复用E043八真实noise/embedding，16fps本地约定。实际800DiT/64000SDPA/400scheduler/8decode/0TE/0native；生成1897.128619秒、评价70.902164秒、CPU1.373173秒/CUDAfalse。summary SHA48e880410ce36e3c6a4be75fbb0f5ba43c9f15136967678dd7262cf09d6f3dff。root八九帧预览全部在读新分数前记录，非盲评/完整事件标注。MJtotal/alignment/fineness/coh=.851667/.912109/.072021/.650879，RAFT8/8，AMT.976930、DINO.914935；相对E043总分6升2降、细节4升4降，模型/CFG/schedule共同改变，不作尺寸因果。折叠、汽车r0转弯、喷水来源仍有歧义，保留全部样例。见[046完整报告](../reports/046_20261003_wan14b_reference_results.md)。

**E050全部complete（23:46核实）：** 14提示/64cache，本模型CFG5/shift3完整轨迹，原Random0政策，只复用同身份TE，不读取1.3B中间latent。actual1400DiT/112000SDPA/700scheduler/64cache/0TE/0decode；采集wall3702.531961秒，CPU汇总3.876798秒/CUDAfalse，summary_launcher rc0。原2208229及四worker全部退出。[collection_summary](../../results/research/E050/collection_summary.json) SHAd77f49d35b2c3440861168e5769d43ba68e53db8b3c61cf44bd2ccaa5c362886。35cond/29uncond、35时间位置、缺0/48/49、仅2双分支state的政策限制保留。源/manifest沿冻结E050协议，未重试或改设置。

**E051全部complete（23:47核实）：** 八plain视频生成/评价/独立CPU汇总均rc0，chain/四生成/两评价worker均已退出。actual800DiT/320000native+pack/64000SDPA/400scheduler/8decode/0TE。生成wall2952.309899秒、eval71.964961、CPU1.349862；[summary](../../results/research/E051/evaluation_summary.json) SHA816751ceec00b03cb4daf94a0fe14762ad08e352d96bedd1ebb25b4422910173。root8/8固定预览和对应BF16在首次读分前记录；audit独立从MJ28题、AMT/RAFT/DINO原数据复算一致。MJ .851667→.755266、4升4降；细节.072021→.038589、coh.650879→.615479，各3升5降；RAFT8→7。拍手r0净降.739115占总净降95.8%，但双seed细节/coh均降；叠衣/汽车均值略升，不能挑例/宣布全崩或等质量。八pipeline时长和−21.6011%，推理allocated40.035→21.197GiB，安装已见峰27.354GiB，reserved48.203GiB，非成熟serving/质量匹配benchmark。见[报告048](../reports/048_20261003_wan14b_plain_results.md)。

**E052实际运行：** 原named tmux `e052_wan14b_ptq_resource`监督2653260、worker3055486，GPU4；23:47:34 ps worker Rsl。已等E050全量complete+summary rc0+空闲后才启动；deadline1791049575.35774（约10-04 01:46:15上海）。全部14B W驻GPU，64/b4/g10/r32/LR100原早停；只block0完整10主Linear，原始cache→smooth→重采cache→LR，后39层0forward。23:52:58核实worker仍Rsl；model_load已complete9.7668秒，smooth_cache完成150.0014秒、GPUallocated峰42.1353GiB/CPU RSS峰236.7361GiB，实际16prefix/32block0/后39层0forward，正完整smooth搜索。无完整pilot/整模PTQ/14B SVD checkpoint或质量结果。[launcher](../../results/research/E052/launcher.json)/[run](../../results/research/E052/run.json)。manifest23a67000b31102a6bfbd31ed77c00f1b6f4032b724b8fc8b67fc775fbf0c6202，CPUcheck3d40931d4ce5234da45685793c8520ad619ee1d235f89c88ca6897198c11358d，worker9ce43f14ef2cea3833abdc4f41b9622580a4f4871f8d62e685636b25981c2dc2/launcher5b6c2d43e5a715351efc9ab3a8ab831cc7bd058e4a53c3657b7b7b1d3d6a43e7冻结。

**后续准备与下一动作：** 未编号的 `ptq_wan14b_matched_calibration.py`保留原完整40层smooth→LR→quant/export顺序，不载E052partial。CPUcheck_v2 complete5.229秒、0权重/CUDAfalse/40层400模块及loader全列表通过；首次PATH缺ninja失败check.json保留，使用现有native environment后重试，无GPU/安装。尚无新运行manifest/deadline/launcher，E052实测后才决定完整预算。`wan14b_svd_native.py`薄适配真实meta40/400/280共享组+配置通过，未来五文件CPU mmap及逐层400native安装待真实checkpoint验证。两个准备入口非科研方法。继续原E052，保留八例与plain，依据完整资源结果决定同模型SVD强基线。

**10-04后续执行依据：** [阶段依赖](wan14b_ptq_phase_dependencies.md)已从实际iterator核实：同阶段下一层输入在yield校准前保存，允许研究按层分工，但必须完整smooth合并后才重采LR，保留真实prefix与每block搜索顺序，不能直接筛非连续层。跨worker随机SVD顺序需显式政策；等E052完整CPU/GPU峰再决定是否最多2worker，尚未实施。00:06:53原worker仍Rsl，完整smooth完成963.0823秒（7共享组各19候选），GPUallocated峰61.8546GiB、CPU RSS峰367.4615GiB。00:09:03 lr_cache已complete164.754秒，原3055486仍Rsl并实际进入self-QKV的QuantLowRankCalibrator（64/b4、最多100次原早停），尚无完成LR组。两相同CPU峰简单叠加734.9GiB接近整机742GiB且未计其他进程，当前不能直接开两完整worker；最终资源决策仍须LR结果。

**14B量化入口已有限核查：** [入口说明](wan14b_quantization_entry.md)从真实headers确认400主Linear、14,050,918,400 W元素，现pack/swizzle/GEMM形状可复用。直接扩展旧`install_qad`会保留BF16全模并构建全FP32 master，最低78.958GiB，不能在当前单卡照搬；未训练plain应逐层pack/释放，且从teacher已cast的BF16 W量化。SVDQuant另需14B自身匹配轨迹cache及五checkpoint文件；layerwise激活缓存不等于模型权重offload，完整校准资源/耗时未测。不迁移1.3B smooth/LR/scale，不将plain当SVD强基线。该入口说明形成时为只读准备；后续E051已实际逐层转换并运行原生前向，仍无新方法或SVDcheckpoint。

## 文献与研究边界

DSAQuant已覆盖CFG分支失配/晚步高频，PTQD覆盖晚步提升激活精度；[末步近邻说明](../01_literature/E045_terminal_cfg_nearest_work.md)。GCBT直接覆盖CFG共同/差分换基及线性后逆变换，并已有Nunchaku成本，停止该候选；[查重](../01_literature/CFG_branch_hadamard_duplicatecheck.md)。通用decoder抗扰/causal帧不均已有SSVAE/IVVAE近邻。空间patch相位已有[入口审查](../02_problems/wan_patch_phase_entry_audit.md)与[有限原文核查](../01_literature/patch_phase_geometry_nearest_work.md)，未测量；ICNR子像素输出头是直接结构近邻，未来任何相位读出须面对同head普通token扰动。批间scale耦合也已有[直接近邻](../01_literature/batch_invariance_collision.md)，不重启P002/C003。

保留历史停线：C006系统通信局部收益缺成熟融合/完整路径基线，关闭论文主线；P006音画事件不可识别，parked；H3中心调整E038无稳定跨任务收益；QAD E022张量拟合改善未带来稳定完整质量，旧rCM共同异常不能当量化因果；E018相位交替不是闪烁证明。具体证据在报告与decision_log，不重开旧网格或用一般数学/现成ABI自动否定系统贡献。

[更大模型资产入口](larger_video_model_asset_entry.md)：官方Wan14B transformer资产已完整准备，固定HF commit38ec498cb3208fb688890f8cc7e94ede2cbd7f68。download/supervisor均complete/rc0，耗时9926.867秒，原PID3654981/3654983均已退出；12分片及小元数据共57,154,198,566B。根节点22:06有界独立复核当前文件size、小文件SHA和原始safetensors headers/index：1,095张量、14,288,491,584个元素全部F32，payload57,153,966,336B；大权重全SHA继承下载时官方LFS核验，未重复重哈希/载tensor，[completion_review](../../results/research/asset_wan14b/completion_review.json)。download SHAad8b7167a4d4773bad27846d1f900178b95e521731aa79a474445d4a9d162dfb，supervisor SHAef3d6ab455b6d47af516bca304e07ce57b408f389beaf9b4055f121a733a149d。TE/VAE大文件沿E024完整SHA+当前size，小文件Git blob匹配官方，[组件复用](../../results/research/asset_wan14b/component_reuse.json)。资产任务未用GPU/载tensor；后续E049已实际加载，但E049完整视频pipeline及参考评价已经完成，尚无14B SVDQuant。


主要风险：诊断8例已被反复观察，不是独立泛化测试；校准帧数和schedule同时改变，不能隔离单因素；MJ原始分数非概率，AMT≠正确动作、RAFTdynamic≠物理真值；九帧sheet≠完整视频事件标注；单层/单状态成本≠整模型收益。不得把文件身份/byte一致性当科学质量门槛。

## 文件、环境与索引

工作树main；不push。保留用户dirty `scripts/rcm_vbench251_worker.py` 和历史dirty `scripts/minimax_h3_svdquant_common.py`。已执行源/结果不覆盖，失败保留；修改另版本。模型、媒体、env/cache全在 `/data1/models/svdquant-wjq`，root约3.2GiB空闲、DATA1约1.2TiB，使用前复查。服务器8×RTX PRO5000 72GiB；6/7是他人活跃任务，其余实际占用会变化，不kill他人进程。当前shell需require_escalated（sandbox故障），正常只对已授权工作升级。

Native Python：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python`；MJ/CPU画图环境：`/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python`。原BASE：`/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers`。E043的官方模型身份与实际噪声/embedding继续复用；旧checkpoint在`ckpts/wan2.1-1.3b-real-nvfp4-s16`。

[报告索引](../README.md) · [决策](decision_log.md) · [实验日志](../06_experiments/experiment_log.md) · [归档旧长状态页](../99_archive/superseded_notes/current_state_before_E047_compaction.md)。最近决定D075：14B plain得失不均，保持全部配对诊断与匹配校准，先看原E052实际资源结果再决定完整SVD预算。总目标保持active。
