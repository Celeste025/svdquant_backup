# 视频 NVFP4 W4A4 研究记录

**当前状态（2026-10-05）：新 idea 研究暂停；用户指定的 E084 历史补测已完成。**
最新结果为 [083：Wan QAD 逐时间步测试 NMSE](reports/083_20261005_wan_qad_timestep_replay.md)。
代码、结果文件与外部数据的边界见[仓库说明](../docs/repository_contents.md)。
当前授权与结论以[状态页](00_state/current_state.md)为准。

## 历史阶段索引

以下条目保留原阶段描述，其中“最新”“进行中”等措辞仅对应当时状态。

最新结果：[082：BF16主干上的Sage3与V中心化对照](reports/082_20261005_bf16_attention_comparison.md)。按用户建议完成四新视频；移除SVD主干量化后仍未见稳定细节收益。40%为局部PV平方误差，当前均值方法非创新；两组并排视频及全帧指标齐备，所有任务完成。

最新结果：[081：V中心化机制与两条完整视频](reports/081_20261005_v_center_mechanism_and_video.md)。两个来源均贡献局部改善，原生已验证；两例视频尚无明确细节修复，不升级主基线。视频和复核齐备，任务已退出。

当前进展：[080：双侧PV差分编码首轮结果](reports/080_20261004_pv_contrast_results.md)。局部对比误差下降，但强对照更优、总误差增加，当前版本停止；独立CPU复核完成，未进入视频/kernel。基线仍为官方Sage3＋现有SVDQuant；参照[077：组合视频](reports/077_20261004_h3_sage3_video_quality.md)、[079：VC/DIDB兼容性](reports/079_20261004_vc_didb_compatibility.md)。下面保留历史阶段条目，当前状态以[状态页](00_state/current_state.md)为准。

最新实验：[073：近等误差相位干预与真实视频续跑](reports/073_20261004_h3_phase_suffix_results.md)。19DiT与2新视频完成；相位改变确实传播至画面，但未看到明确缺陷修复，停止该窗口扩网格。

此前研究讨论：[072：三个观察与一个待验证想法](reports/072_20261004_observations_and_candidate.md)。聚焦误差结构是否具有独立功能影响，当时未形成已验证方法；后续有界实验见073。

最新指标：[071：H3 SVDQuant vs BF16 的 LPIPS、L1、L2](reports/071_20261004_h3_baseline_distances.md)。10配对1240帧完整评分：LPIPS 0.4201、MAE 0.1103、RMSE 0.1755；原始L2与相对L2及逐例数据齐全。用户指定现有配方为正式比较基线，不再推进昂贵官方默认校准对齐。

最新中期汇报：[070：按创新潜力排序的尝试、结果与下一步](reports/070_20261004_midterm_report.md)。没有已成立的论文级创新；最扎实的是局部与整模数值误差排序反转。E073八条完整BF16参考视频已生成解码完成，成对观察尚待，不重复启动。以下为历史阶段记录。

最新筛查：[069：数值边界与匿名视频](reports/069_20261004_h3_numeric_boundaries_and_anonymous_screen.md)。两条格式/低秩候选尚无超出已知数值性质的功能证据；E068匿名抽帧未辨认稳定A/B功能损失，独立人评仍待。0新模型/GPU，未恢复工程支线。

最新观察筛查：[068：手部可读性与身份混淆假设](reports/068_20261004_h3_readability_hypothesis_screen.md)。低位重影在初始分离构形已存在，未辨认接触后的身份转折；仍缺无遮挡控制，暂不为此开发干预。0新代码/模型/GPU；候选未成为已验证机制。

最新研究决策：[067：停止工程支线，回到机制假设](reports/067_20261004_research_focus_reset.md)。E072a未启动即停止，0新GPU。RoPE融合、AdaLN shift及Jensen偏置的通用方案均有直接近邻；弱运动问题尚缺可辨证据，未用含噪velocity帧差制造结论。下一优先实际缺陷与鉴别实验，非继续扩基线工程。

最新执行：[066：H3校准候选与部署一致性](reports/066_20261004_h3_native_candidate_identity.md)。单候选实际库内评分、native执行和导出重载通过，两例完整QKV逐字节一致；GPU96.02秒、独立CPU24.28秒，任务均已结束。仅基线工程，未完成完整配方或质量评价；下一进入实际校准，避免重复接口检查。

最新执行：[065：H3真实校准入口与精确SVD成本](reports/065_20261004_h3_baseline_entry.md)。两条变长序列通过真实缓存接口，attention输出逐字节一致；最大fc1完整FP64 SVD约59.40秒、峰10.45GiB。两项独立CPU核验通过，GPU任务全部结束。仅强基线工程，未完整校准、无画质或创新结论；下一验证单层单候选评分与导出一致性。

最新核查：[064：H3强基线比较合同](reports/064_20261004_h3_strong_baseline_contract.md)。固定上游commit的源码/CPU语义审计确认：旧平滑搜索缺identity及activation-only候选，评分未纳入LR，本地随机FP32 SVD与上游FP64完整SVD不同；E065b未覆盖这些差异。旧H3当前源码已有完整block评分，不能误称忽略非线性结构。0新模型前向，无画质或创新结论；下一先核验完整packed sequence的校准接入，非继续固定尺度网格。

最新进展：[063：H3完整配对视频就绪](reports/063_20261004_h3_blind_video_ready.md)。四条冻结配置视频及两条历史BF16参照已准备，82次DiT在349.42秒内完成；独立核验160份新step与四视频全部496帧通过。A/B标签隐藏，尚无人类偏好或质量结论；先判断SSE反转是否具有实际任务意义，不继续仅凭代理指标做机制扫描。普通kernel/伪量化修整不算创新。

[062：局部误差方向诊断](reports/062_20261004_h3_error_direction_and_video_check.md)：保存样本中两配置差值约85.6%的能量正交于原误差，排除简单径向缩幅近似，不能证明方向致害或特殊机制。

最新结果：[061：H3逐层收益迁移，但整模误差仍反向](reports/061_20261004_h3_local_transfer.md)。四个teacher状态各200层的video局部SSE全部改善，中位约3%；真实M512、full-M同样本与全部token三种读出一致，原完整DiT video SSE仍增加0.47%–12.93%。排除这些输入上的简单局部排序反转及形状/采样假象，尚未识别具体机制。GPU350.657秒，独立CPU247.514秒验证通过，GPU已释放；未新增视频或质量结论。普通工程不当创新，下一GPU须先有可区分的干预预测。

最新结果：[060：H3 carry-Q 局部改善、整模视频误差反向](reports/060_20261004_h3_carry_q_results.md)。200层/1600个校准格局部SSE全部改善，层中位降3.02%；四个整模video SSE反而增加12.93%/11.00%/0.47%/4.49%，audio三点改善。独立输出与局部选择复核通过，续跑正常退出、GPU已释放。不扩该配方迭代，不将反转或基线工程认领为创新；未证明视频质量变化或完整官方校准无效。[059](reports/059_20261004_h3_carry_q_baseline_progress.md)保留资源续跑历史，当前状态以[current_state](00_state/current_state.md)为准。

最新结果：[058：H3真实深度路径四角诊断](reports/058_20261004_h3_depth_four_corner.md)。6完整DiT/70局部block，plain及legacy SVD的16个非零深度对照中，输入依赖项净作用全部为抵消（−0.338%至−3.386%），去均值后仍负；停止稳定净放大候选。独立104PT核验通过。量化输入响应能量本身反而增大，被局部误差负交叉项抵消，不能称量化更稳定。无新方法或视频收益。

最新结果：[057：H3 QK径向收益诊断](reports/057_20261004_h3_qknorm_radial.md)。2完整BF16＋12局部native，共享激活packet对照中raw QK改善仅0.454%–1.383%，主要来自切向；norm后仍有同量级改善，停止径向容量错配候选。独立CPU复核通过，GPU已释放；没有新方法或质量收益。

最新结果：[056：H3 SwiGLU 乘性交互诊断](reports/056_20261004_h3_swiglu_interaction.md)。2完整BF16＋6局部native，六格交互净作用全部轻微抵消（−0.017%至−1.972%），同P及BF16算术控制一致；停止二阶交互修正候选。独立CPU复核通过，GPU已释放。不是创新成果，不启动generic非线性adapter。

最新结果：[055：H3初始化目标对照完成](reports/055_20261004_h3_initialization_results.md)。完整200层、9native预测，weight_init相对legacy视频SSE变化−11.7%/−7.5%/−0.2%/+5.9%；未满足预设一致收益，不追加shifted/完整PTQ。不是新方法，不证明完整官方校准无效，当前无新GPU任务。

最新审计：[054：H3低秩基线身份与范围](reports/054_20261004_h3_baseline_identity.md)。E060确认200层600因子与原state一致；当前候选代码不同于官方carry-Q校准，历史生产源码未绑定。保留原结果，降低“完整官方强基线”的外推口径；后续E061执行结果见报告055，非创新。

最新执行：[053：激活来源对照未通过数值分辨率门槛](reports/053_20261004_h3_activation_oracle_control.md)。六个zero完整调用精确复现；首oracle在固定小样本BF16加回比例10.3384%≥10%处停止，0完整oracle。没有激活来源主终点，不作机制阴性；不改阈值或扩精度网格，控制工程不当创新。

最新补充：[052：交互项不等于有害放大](reports/052_20261004_h3_update_arithmetic.md)。E058 CPU精确分离第二步更新舍入；SVD交互范数比降至30.5%/41.3%，仍超旧门槛，但交互净能量效应+0.17%/−6.17%，不能以压小范数为目标。固定W/动态A来源仍待有效native控制，不启动软件开关网格或普通精度优化主线。

最新结果：[051：H3相邻误差与传播判别](reports/051_20261004_h3_temporal_diagnostic.md)。E056/E057完成：SVD去通道常数/比例项后相邻cosine0.701/0.622；实际两步增强存在，但交互超预设门槛，停止直接推进线性互补方案，不把相关当创新。主实验H3；普通kernel优化不当创新。此前[050](reports/050_20261004_h3_phase_diagnostic.md)已停止patch边界损伤候选；E054未运行，Wan14B完整PTQ不启动。当前状态以[current_state](00_state/current_state.md)为准，下列为历史阶段记录。

最新汇报材料：[049：动机→实验→结果与两个备选方向](reports/049_20261004_research_brief.md)。本轮为报告整理，无新GPU实验；调度状态见[current_state](00_state/current_state.md)，下列active为此前历史记录。

当前目标状态为 **active**。已安装用户指定research skills，持续记录真实实验与停线决策；目前仍未建立可投稿核心贡献。

最近完整结果：[048：14B plain完整配对](reports/048_20261003_wan14b_plain_results.md)。MJ .851667→.755266，八例四升四降，净下降主要来自拍手r0；并非所有样例崩坏或等质量。实际pipeline时长和−21.60%，推理allocated40.0→21.2GiB。E050匹配14/64采集与汇总完成，E052完整预算首block资源测量已完成；E053完整PTQ协议和CPU检查已准备，启动命令中断后核实未运行，尚无14B SVDcheckpoint。

[047](reports/047_20261003_wan14b_calibration_started.md)保留采集/生成执行历史；[046](reports/046_20261003_wan14b_reference_results.md)为完整14B BF16参照。

最近1.3B量化结果：[044：匹配校准没有消除主要失真](reports/044_20261003_matched_calibration_results.md)。E047完整PTQ耗时约5小时20分，E048八例原生生成、评价和独立汇总全部complete。匹配校准相对旧SVD的MJ均值.56357→.58369（5/8例提高），仍低于BF16 .62983；拍手碎片保留、汽车r1明显退化，未建立稳定质量修复。保留强基线与全部阴性结果，停止扩校准网格；下一完成官方14B资产核验与原版完整参照入口准备。仍无可投稿核心贡献。

[043：匹配校准执行记录](reports/043_20261003_matched_calibration_started.md)保存采集、完整PTQ及成本。[042：完整量化轨迹的BF16末步控制](reports/042_20261003_terminal_repair_control.md)保留八例强控制，总分.56357→.64060，但手/衣碎片和几何缺陷仍在，备用BF16新增2.65GiB，非等质量优化。

此前报告：[041：同状态末步干预](reports/041_20261003_terminal_intervention.md)。单末步SVD足以致损，首位置传播解释不成立；通用CFG/晚步机制已有直接先例。

此前报告：[040：原版native完整配对](reports/040_20261003_vanilla_wan_native_pair.md)。SVD改善plain，但相对BF16八例细节分全降。

此前报告：[039：原版Wan完整参照](reports/039_20261003_vanilla_wan_reference.md)。八条BF16参照与全部评价已完成。

此前报告：[038：音画事件读出](reports/038_20261003_audio_event_readout.md)。E042未建立基线物理事件对应，P006 parked。

[037：拍手可辨性](reports/037_20261003_clapping_readability.md)：六条视频全部744帧显示低位前段仍反复开合，重影令后段计数不可靠，不支持统一动作变慢。非时间blend帧排除了“全部来自最后时间拼接”的解释。

[036：低秩信息对照与范围收尾](reports/036_20261003_projection_information_control.md)：E039/E040工程收益保留，C006独立论文主线已关闭。

[035：输出通信与原生SVD投影](reports/035_20261003_output_projection_boundary.md)：四臂完整成本与误差、显存及通信账本。

[034：完整视频中心对照](reports/034_20261003_center_video_results.md)：E038全部24视频、480DiT及独立评价完成，中心扩展未获得跨任务稳定质量收益，已停止扩大该路线。

此前阶段报告：[031：投影与稀疏预算](reports/031_20261003_projection_sparse_routing.md)。E035同输入配对及独立复核完成：mask确有改变，物理/有效预算净增均不到1%，停止当前预算膨胀假设，不测稀疏consumer。跨卡依赖核查及后续E036现已完成，见最新报告032。

此前阶段报告：[030：连续粗中心强对照](reports/030_20261003_coarse_center_control.md)。E034已执行156native：粗中心约29.7ms、聚类31.5ms，峰值同约2.755GiB；粗中心获得聚类相对global改善量的93.1%–97.3%。独立复核完成；停止扩聚类网格，下一只读核查既有稀疏工作图实验入口。

[029：共享校正行消费者](reports/029_20261003_shared_row_consumer.md)：E033及独立复算完成，真实消费者与完整成本已建立。

[028：自适应中心成本](reports/028_20261003_adaptive_center_cost.md)：E031/E032已完成构造成本与精度复核。

此前阶段报告：[027：受限query中心](reports/027_20261003_restricted_query_centers.md)。E028–E030完成并独立复核：rank16同样本保留global→block误差优势91.66%–96.66%，单文本固定basis迁移后降至62.02%–75.60%；可分解FP32合同影响较小。RoPE核查不支持简单低通解释，暂不写完整consumer；下一先评估自适应中心构造的实际成本，无视频质量/实际consumer收益。

此前报告：[026：校正成本与分块强基线](reports/026_20261003_correction_cost.md)。E027完整流程+6.91%时间、−26.09% allocated峰值，输出相同；停止朴素在线校正调度投入。

此前阶段报告：[025：块均值校正复用K4的精度代价](reports/025_20261003_qmean_k4_intervention.md)。E026两次native attention/零DiT与独立CPU复核完成：NMSE .002420→.003315，52/56head变差，但仍优于本例global参照；这不是全部融合设计的误差下界。下一步评估保留BF16 K校正与现有query-chunking的实际成本。当前所有GPU任务已退出。

[024：完整解码器对照](reports/024_20261003_decoder_control.md)：E025全部16个同latent对照完成，动态判定全部不变，主要语义偏差保留，结束VAE排错。

此前阶段报告：[023：官方FastWan-QAD完整产品基线](reports/023_20261003_official_fastwan_baseline.md)。E024已完成官方SM120原生线性/自注意力、16段81帧视频和全部评分：MJ均值.32240，RAFT动态2/16，内容对齐与seed差异仍明显。实际官方入口为UniPC三步[999,857,599]；发布历史核查已确认相同路由，没有支持改为DMD的权重绑定证据。不同权重/采样/decoder的完整产品参照不构成本项目创新；所有GPU任务已退出。

[022：扩大QAD的完整结果](reports/022_20261003_expanded_qad_final.md)：128次训练和64视频完成，按预定开发规则选更新64，native误差−32.22%，但MJ plain/QAD=.07106/.07193、两文本升六降。共同BF16条纹未由FP32 VAE或原架构对照解决；E023仍只保留计划。当前无可投稿核心贡献。

此前报告：[020：QAD完整视频评价](reports/020_20261003_qad_video_results.md)、[019：主权重QAD训练与部署](reports/019_20261003_mainweight_qad.md)。旧小数据64次训练拟合改善而native开发误差上升；plain/QAD单DiT约1.66秒，比BF16 1.79秒/旧SVD 1.99秒快，尚未建立等质量收益。旧16视频全部原始评分保留。

- [当前状态与下一决策](00_state/current_state.md)
- [实验索引](06_experiments/experiment_log.md)
- [决策记录](00_state/decision_log.md)
- [执行计划](08_paper_shape/execution_plan.md)
- [018：LSE诊断与流程重置](reports/018_20261003_lse_diagnostic_and_reset.md)
- [017：布局相位结构及边界](reports/017_20261002_query_group_phase.md)
- [016：更小张量误差没有保证本例更高视频评价](reports/016_20261002_fp4_attention_video.md)
- [015：完整FP4 attention的成本—保真取舍](reports/015_20261002_full_fp4_attention.md)
- [014：跨模态传播未改变本轮配方选择](reports/014_20261002_crossmodal_propagation.md)
- [013：普通 NVFP4 与 SVD 的完整模型取舍](reports/013_20261002_plain_native_tradeoff.md)
- [012：行为筛查未建立可用对照](reports/012_20261002_behavior_readiness_stop.md)
- [011：条件响应诊断止于 BF16 对照](reports/011_20261002_conditional_response_stop.md)
- [010：H3 两例配对视频与独立评价](reports/010_20261002_h3_heldout_results.md)
- [009：H3 原生基线实测——1.22× DiT加速与显存下降](reports/009_20261002_h3_native_progress.md)
- [008：原生 NVFP4 配对视频](reports/008_20261002_paired_video.md)
- [007：完整原生 Wan 基线与真实性能](reports/007_20261002_native_wan_baseline.md)
- [006：attention 假设止损](reports/006_20261002_attention_stop_native_baseline.md)
- [005：校准止损与原生算子](reports/005_20261002_calibration_stop_native_ready.md)
- [004：H3模态pulse](reports/004_20261002_modality_tradeoff.md)
- [003：CFG对照否决跨步路线](reports/003_20261002_temporal_stop.md)
- [002：基线修复与初步假设](reports/002_20261002_baseline_and_hypotheses.md)
- [001：接手、安装与审计](reports/001_20261002_intake.md)

目标是得到可复现、经最近工作核查的可投稿成果；当前仍是探索阶段。阴性与失败结果保留；协议变更显式另立版本，不覆盖历史；避免把不成立的跨实现byte假设当科学否决。原始数据和大文件留本地数据盘，报告和源码可评审；不自动推送远端。

- [035 输出通信与原生SVD投影](reports/035_20261003_output_projection_boundary.md)：E039完整两卡边界，22.85%延迟收益；单状态工程证据，下一同packet LR信息控制。

- [036 低秩信息对照与范围收尾](reports/036_20261003_projection_information_control.md)：同packet/main控制完成，保留工程成果，关闭C006独立论文主线。

- [074：H3动作观察与空间解码归因](reports/074_20261004_h3_action_and_decode.md)：raw tile已模糊，停止blend方法候选。

- [075：等成本候选组合方法初筛](reports/075_20261004_h3_composition_candidate.md)：16前向完成，二段候选无稳定收益，停止扩展。

- [076：官方Sage3与SVDQuant组合](reports/076_20261004_h3_sage3_interaction.md)：完整四角实测，视频误差增加但无净额外放大。

- [083：旧Wan QAD逐时间步测试NMSE补测](reports/083_20261005_wan_qad_timestep_replay.md)（2026-10-05，E084完成；新研究仍暂停）。
