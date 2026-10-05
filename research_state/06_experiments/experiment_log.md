# Experiment Log

E065（2026-10-04，worker exited / unfinished）：[carry-Q配对基线](E065_h3_carry_q_plan.md) 完成8BF16校准+1legacy exact、39/200两臂层；外部1800秒监督后退出但JSON未finalize，terminal_observation另存。4992候选局部调用有完整前缀，尾部0–128未知；无新臂整模结果，不作算法阴性。原源码/检查点/半成品保留。

E065b（2026-10-04，complete）：[同配方续跑](E065b_h3_carry_q_continuation_plan.md)，CPUcheck47.791秒/0CUDA/1923复用文件后，GPU0完成剩余161层、20608候选选择和9full，elapsed3294.944184秒、峰40377451008B、新数据58151212022B/两阶段115057889282B。累计有效候选25600、物理计费范围25600–25728、18full；exit_code0、worker537031/tmux已退出、GPU0释放。两独立CPU复核complete：输出0.787228秒/21PT/差4.45e−16；选择271.724096秒/2612PT/400manifest绑定/差4.18e−16。200/200层、1600/1600校准格carry SSE更低，层中位比.969830799；四点整模video carry/restart 1.129255874/1.109991657/1.004742296/1.044936389，centered亦更差；audio .697972857/1.004331650/.783446364/.719708758。191层选k7，184条recorded曲线单调不增，但不据此扩迭代。0.075606秒stdlib汇总，无新forward。evaluate SHA08ac1d487b8683ab9ae7dcaeb42b46a76795435165c8fef3314bfc77c4f56e48，summary SHA1e5c2133fd8adb5a64cefa89dca3d0d5d0d8637c89c5dc2b6d05ad593da37efa；D090/报告060。没有新机制或视频质量结论，不否定完整官方PTQ，目标active。

| ID | Purpose/claim | Status | Evidence | Decision |
|---|---|---|---|---|
| E000 | baseline hardware | complete | native SM120两shape+两级scale；systems_smoke_20261002.json | 可用现有torch做原生部署，无端到端加速主张 |
| E001 | baseline H3 repair | complete | 10回归，2prompt×2step，12 variant rows；首轮Sage视频−0.78%；显式torch修复+0.19%，无收益 | 修复保留；不能解释全部旧失败；发现模态取舍 |
| E002 | C001 | complete | 24branch rows、12series，cache回放0；CFG相关0.102/0.184 | 停止全W4A4互补路线 |
| E003 | C004 | complete | 1prompt×2step×5arm；zero0；实际torch BF16 | SVD video endpoint+13.67%/+8.52%，audio改善；有限校准对照 |
| E004 | C004简单解释 | complete; stop | 同容量低秩B refit，pooled/normalized/video4 | 先做廉价已知方法对照；不作为新方法 |
| E005 | 数值契约 | complete | 真实H3输入上的ties/RNE/native packing/accumulation | 决定旧QDQ证据能否迁移native |
| E006 | P004/C005 | complete; stop | 12case真实factorial；48次own-scale byte parity0、12zero replay0；交互31.71% | 0/12净放大、固定尺度改善6.46%不足10%；不继续 |
| E007 | native Wan baseline | complete | 完整正确性、fastpack byte0、整模SHA、三backend计时/profile | native1.975s vs BF161.770s；有storage节省，无BF16速度收益 |
| E008 | paired free rollout/video | complete | seed1 BF16四步历史exact；1200nativeGEMM/flags；同noise、同VAE | 实际视频已生成，单样本不作质量排序 |
| E009 | native H3 resident baseline | complete | 200W exact、迁移exact、50block slow/fast SHA exact；三arm独立进程、全部参考SHA匹配、trace独立归因 | BF16/QDQ/native 8.260/20.778/6.787s；native峰值allocated16.80GiB，启动37.62GiB；单calib DiT非质量等价生成加速 |
| E009b | 已有FlashInfer fused-up组件对照 | complete | 6synthetic＋4real，主支精确，完整linear舍入NMSE约8e-6 | 长M QKV组件1.53×，FC2和短M无收益；不外推整模或新贡献 |
| E010 | H3两例PTQ-heldout配对生成 | complete | 四段124帧视频，160逐步文件独立验证；8000原生GEMM/pack合法，102 BF16 SDPA/DiT；116条官方评价回答有效 | p30评分相同，p36两项初始形状较差；video自由轨迹NMSE .226/.325，不当质量分数；两例不能证明质量等价 |
| E011 | native投影与稀疏router dense guard | not_ready_not_executed | 官方utils CPU导入成功；无已验证SM120 consumer，真实tail guard语义未验证；readiness_audit.json | 暂不实现/适配新后端，无GPU，不当机制阴性 |
| E012 | p36最小条件编辑的局部响应筛查 | stopped_teacher_gate; independently verified | 4条件TE/两个原点velocity exact；26BF16、2652SDPA/0FP4/0disk，74文件独立验证；方向差分SNR1.032/.642<10 | 无合格control，自动跳过native；诊断条件不足，不是量化机制阴性；v1零forward记录错误保留，v2仅修复SHA |
| E013 | 封闭投影行为task-readiness | stopped_teacher_seed1_gate; independently verified | 首对50步BF16完成，100DiT/10200SDPA/0FP4；188文件独立审计。两例rho间距guard导致unknown，有效帧74.2%/66.1% | 第二seed/native未执行；接近动作可见，不能把保守读出失败当能力失败；不扩toy提示 |
| E014 | 完整plain_h3_recipe强对照 | complete; independently verified | 200原W numeric roundtrip exact、9共同状态前向＋15bench，269文件独立审计；BF16/SVD/plain中位8.291/6.804/5.823s，plain峰值15.06GiB | plain较快14.4%但六模态endpoint有取舍；保留两臂，不作质量/新方法主张。原空闲尾读数暂停保留，续跑未重置deadline |
| E015 | 实际单步误差的跨模态四角续接 | complete; stop; independently verified | 18DiT、1836SDPA/3200FP4/0disk，CPU/GPU更新exact、101文件审查；最大局部oracle改善p30/SVD音频11.42% | 净作用正负交叉、四组排序均不变，停止本轮扩展；不否认局部传播，不立固有敏感性/质量/新方法claim |
| E016 | 现成FP4 attention完整成本—保真基线 | complete; independently verified | 27DiT/323.64秒、两历史exact、116文件核验；BF16/block/global attention 6.791/5.124/4.968s，峰值allocated16.806/17.713/16.806GiB | 全部六端点误差增加；global仅p1更准、后四项block更准，保留成本—保真取舍；非质量/新方法主张 |
| E017 | 两种FP4 attention的实际配对生成 | complete; independently verified | 80DiT＋4decode＋新旧8视频232题，658.57秒、273文件核验；旧四raw答复完全复现 | p30四臂同分；p36 block多四题变差，global全部题同SVD；block更低tensor误差未对应更高评价，非泛化质量claim |
| E018 | 固定真实QKV的query分组相位干预 | complete; independently verified; parked | 1DiT source exact＋27attention，90.79秒、142文件核验；BF16/global所有相位exact、block128内区exact；追加CPU向量投影10.32s | block64有栅格能量结构但平均NMSE几乎不变；伪边界也交替、向量未整体翻号，保留有限观察，不扩为视频闪烁路线 |
| E019 | 固定FP4 packets分片合同；追加LSE失败诊断 | original failed_stop; LSE diagnosis complete; direction stopped | 原队列1attention后stop；独立False→True两调用，False exact、True39/161,559,552元素变化/NMSE2.84546e-12，CPU完整复算通过，累计3attention/0DiT | 原分片与真实数学对照未执行，不作机制阴性；现有工作覆盖且流程机会成本高，转主权重QAD强基线 |

| E020 | 真正主权重QAD与无LR原生导出 | complete; independently summarized | 300W/1.392B masters、64updates/168逻辑DiT，24teacher重算exact；native开发NMSE .137708→.148410，28部署DiT四臂计数/SHA一致 | 训练拟合改善、开发退化；plain/QAD1.66s快于BF161.79/SVD1.99s但未质量匹配；普通QAD不是新方法 |
| E021 | E020固定64终点完整视频与内容/运动评价 | complete; independently summarized | 4prompt×4arm共16视频/64DiT/14400FP4/3840SDPA，全77帧媒体核验；MJ16条+AMT/RAFT/DINO全视频读出 | MJ两升两降、均值由campus异常主导，QAD dynamic1/4；保留全部样例，不立普遍质量收益 |
| E022 | 扩充分布与优化后的QAD强基线 | complete; independently summarized | 256train/32dev，128updates/512backward；native0/32/64/128=.154799/.123383/.104920/.112768选64；64视频/256DiT/57600FP4全评分；VAE与原架构对照完成 | dev误差−32.22%，视频MJ相对plain+.000868、两文本升六降；RAFT运动减少未复现；共同条纹根因未定位，无稳定质量/创新claim |
| E023 | W-only固定初始outer global普通基线 | prepared_not_started; plan only | 同E022数据/128×4/选择规则的单因素计划，成熟ModelOpt配方来源已固定 | 未实现/未GPU执行，不是新方法或完整ModelOpt复现 |
| E024 | 官方FastWan-QAD外部产品基线 | official UniPC suite complete; independently summarized | 固定官方源/公开资产SHA，独立torch2.12/cu130；16×81f视频/48DiT/14400FP4GEMM/1440FP4attn/1440dense cross；MJ/时序全部完成 | MJ.32240、动态2/16、对齐异质；不同权重/decoder/帧数/采样，非rCM单因素比较。当前UniPC[999,857,599]与训练文档DMD差异待查，不称新方法 |

E025（2026-10-03，prepared）：固定E024全部16 final latents，完整Wan VAE FP32 vs已存TAEHV FP16，零DiT；计划见 [E025_decoder_control_plan.md](E025_decoder_control_plan.md)。当前尚无新GPU结果。

E026（2026-10-03，prepared）：复用E018真实H3单层packet，固定前六项、仅将块均值correction替换为无限精度μ×K4；零DiT，回答附加query行复用K4的误差下限。尚未GPU执行。

E025完成：0 DiT/16完整WanVAE FP32 decode，320.16秒/13.22GiB；MJ .322401→.332559，文本3升5降，RAFT16条判定均不变，主要语义错误保留。独立汇总complete，报告024；停止VAE网格。E026已开始唯一GPU1/600秒两attention干预。

E026 complete：2次native attention/0DiT、18.53秒、4.05GiB；固定原μ×K4相对block NMSE+36.98%，仍比global低6.93%。52/56head vsblock变差，53/56 vs global改善。独立CPU复算及真实packet/K4/μ核验通过。结果仅针对本例固定μ，不是所有融合方案的下界；报告025。所有本轮GPU任务已退出。

E027（prepared）：同E026真实H3层，完整correction与Qchunk4096成本比较，另独立BF16 M16-stripe；各3warmup+10重复，GPU0顺序运行。计划见E027_correction_cost_plan.md。

E028（prepared）：同E018三层真实Q/K及K4 packets，CPU-only三种中心变化几何（Euclidean/K-score/K4-error）谱；不训练/新量化器/新GPU，计划E028_center_spectrum_plan.md。

E027 complete：full/chunk完整流程31.780/33.975ms，allocated峰值3.518/2.601GiB；四输出NMSE0，182attention/27preprocess/0DiT。stripe独立14.815ms，不作融合成本；独立cost_summary complete，报告026。

E028 complete：CPU13.104秒，504 head-geometries，无GPU；rank16 K4-error尾能量block0/24/48=.00894/.08073/.03627。独立scalar汇总进行中；不推断native精度。

E029 prepared：同三层、rank16固定，Euclidean/K-score/K4-error同步中心与block/global端点，共15native/0DiT；先CPU check，不测性能、不新增视频。

E029 complete：15native/0DiT，65.57秒，6.04GiB（含oracle构造，不是部署性能）；三几何rank16较block NMSE+1.58%–3.29%，保留global→block优势91.66%–96.66%。独立CPU output归约进行中。

E030 prepared：p30/s14一次完整DiT捕获三层均值、固定Euclidean rank16 basis，迁移原p36/s14三层3native；0新视频、600秒GPU0。

E030 complete：1DiT+6probe，78.28秒、转换峰值37.60GiB、model释放allocated.00891GiB；跨p30→p36固定basis保留global→block优势62.02%–75.60%，比same-sampleEuclidean NMSE+10.63%–11.82%。166/168head优于global、两head更差。factorFP32 pooled变化−.023%至−.038%但head正负都有；六输出/basis独立复核complete。报告027已纳入；全部GPU退出。

E031 prepared：既有p36/s14三层，GPU fullSVD/range0/range1 rank16自适应中心成本＋9次native精度；0DiT，600秒GPU0。计划E031_adaptive_center_cost_plan.md；尚未GPU。

E031 complete：97.86秒、9native/0DiT，fullSVD构造210.50–212.58ms，range0 1.839–1.842ms，range1 3.613–3.614ms；rawprep候选10.34/12.15ms对block8.43ms。独立CPU归约进行中。

E032 prepared：同三层K16共享中心表、固定6轮weighted Lloyd，3native/0DiT；两prep计时scope与同轮global/block基线，GPU0/600秒。

E032 complete：43.69秒/3native/0DiT，三个输出和小中心/id/distortion及12scope计时独立复算通过。全部168head优于global，三层优势保留66.06/55.97/71.74%；prep8.49ms对block8.43，准备peak−25.7%，尚无实际consumer。CPU人工tie失败已保留且仅修fixture。

E033 prepared/implementing：私有共享DS行consumer、10正确性新旧调用和3层五臂完整路径（468benchmark attention），GPU0/900秒含首次JIT。尚未编译或GPU。

E033 launched：root独立审查与audit补充审查均未见阻塞，GPU0/tmux e033_consumer/worker3062111，900秒包含首次私有JIT。两实现源和CPUcheck冻结；validation三次means_and_key未列入preprocess/mean receipt，benchmark计数完整、native总数不受影响，保留此记录限制，不为修receipt重跑。

E033 complete：110.37秒/478native/0DiT；五samepacket控制maxULP0；15臂完整路径及25输出独立CPU复算complete。codebook31.480–31.538ms/2.755GiB对freeblock31.689–31.770/3.518，保留优势55.97%–71.74%；五臂均未被点估计支配。worker3062111退出。报告029。

E034 prepared/implementing：同K16连续均匀Qtile中心、四臂完整路径156native，GPU0/900秒，无新kernel；先检验简单粒度基线，尚未GPU。

E034 complete：156native/0DiT、44.38秒，worker3237350退出；coarse16三层NMSE .00285639/.00815243/.00757479，完整路径29.637–29.699ms/2.755GiB。独立复算进行中。报告030；停止扩聚类网格，保留强基线。

E034独立汇总complete：CPU46.31秒、0GPU；12 BF16/9 E033旧臂数值（旧臂差异0）、3实际C/id几何、36warm+120timed/156native全部复核。四臂各层均未被严格支配。报告030已闭环；共享中心路线保留为工程基线，不扩参数。

E035 prepared/implementing：固定E014 e010_p036_s14 teacher输入，1BF16 DiT/3局部SVD-native qkv capture；contiguous128/prefixdense/CDF.9/min4 CPU路由诊断。原始shape/有效段/投影前padding保留，无稀疏kernel。GPU0/900秒capture，CPUrouter180秒，尚未执行。


E035 complete（2026-10-03）：GPU前v2采用原H3 VSA3D128（prefix11/video200），此前contiguous版本未执行；正式router前按BSA澄清CDF>=p。捕获39.72秒/1BF16 DiT/102SDPA/3native projection；CPUrouter18.14秒，独立六小artifact复算1.42秒。物理块预算净增+.371/.904/.492%，真实有效token对+.370/.884/.468%，没有被padding掩盖的大幅预算变化。未运行任何稀疏consumer/视频。全部worker退出；报告031，停止当前预算膨胀故事。


E036 prepared/implementing：已有E034 block0、P2 token→head→token完整边界，bf16_a2a/mixed_k16/fp4_table16三臂同coarse16；78native/0DiT，GPU0/1同NUMA/900秒。新薄通信适配，复用冻结consumer；先CPU布局检查，尚未GPU。


E036 complete：launcher7.38秒，78native/0DiT；rank0/1 worker4053101/4053102均退出，torchrun4053068退出。同coarse16完整P2边界三臂28.633/23.214/21.553ms（每轮两rank wall取max）；双向A2A含回传619.500/396.977/324.488MiB。独立CPU9.39秒complete，全部输出等于E034 coarse16，BF16 NMSE .0028563913。报告032；下一global-Q/T1强对照，尚未执行新GPU。


E037 prepared/implementing：同E036已有block0，两卡fp4_table16 vs fp4_global/T1，完整含逆输出路径；52native/0DiT、GPU0/1、900秒。先核冻结源及新增T1/合并统计合同，尚未GPU。


## E037 complete / E038 prepared — 2026-10-03

E037 52native/0DiT、两卡launcher7.18秒，独立CPU8.17秒完成。T16/global中位21.592/20.205ms、NMSE .00285639/.00356130；报告033。E038固定4prompt×2seed×3arm完整视频，480DiT预算，协议/manifest已在生成前落盘；当前实现中，尚未启动GPU。

E038执行更新：prepare launcher22.25秒/worker16.71秒complete，4TE、8真实共享输入、峰值29.86GiB；生成三臂已启动，GPU0/1/5，tmux e038_center_video，sources/manifest/CPUcheck已固定。总480DiT，尚未得到视频质量结果。

E038 attempt01：launcher6.21秒failed_preserved，bf16输入绑定检查发现CUDA上下文已初始化而停止，0 attempted DiT、0 allocated/reserved。global/coarse进程被launcher正常清理；三卡释放。原runner/launcher/日志保留；失败denoise JSON原样移至results/research/E038/attempt01。修复仅将not-cuda条件限定CPU check，v2另文件，生成设置/样本/预备输入不变。

E038 attempt02：v2仅限定notCUDA检查适用CPU check，八份实际输入设备均CPU且通过真实输入绑定。新runner/launcher另文件、旧源不改；2026-10-03 10:12左右启动，worker470574/470590/470612，三臂首个真实DiT完成。正式生成仍running，无质量结论。


## E038 complete — 2026-10-03

正式v2三卡生成/解码1146.45秒、24条124帧媒体，480DiT/960双scheduler/96000 native GEMM/16000 FP4attention/32960 SDPA；所有worker已退出。评价216.08秒、generation_summary与evaluation_summary独立complete。MJ .96797/.87041/.85750，coarse−global −.012914、3/8配对更高；报告034。按预定决策停止扩大中心表示路线。首次0DiT失败保留attempt01，输入/协议不变；修复为CPU检查作用域，v2另源。新输出接口问题只记录候选，尚无E039执行。


## E039 prepared / input complete — 2026-10-03

协议E039_output_projection_parallel_plan.md，四臂真实完整输出通信与SVD投影成本。prepare_h3_projection_boundary.py 原helper2SDPA/0DiT，worker2.336秒、launcher8.178秒complete；完整O[22592,56,128]，53 modelpad RMS.80083/max4.0625保留。计时计划52boundary/rank、130 total native main GEMM，GPU0/1/900秒，尚未启动。新源/results/DATA1独立，旧结果保留。


## E039 complete / E040 prepared — 2026-10-03

E039 launcher9.23秒、104边界/130native/0DiT，两rank1196764/1196765及torchrun1196289已退出；独立CPU4.249秒complete。BF16/FP8/row/side wall6.348/11.090/8.762/4.897ms，side相对BF16输出valid NMSE5.9324e−8；报告035。E040协议已冻结，同packet/main的三种LR信息来源单因素控制，仅2native GPU0/300秒，尚未运行。


## E040 complete — 2026-10-03

GPU0 worker1369659完成6.851秒、launcher7.378秒/rc0，2pack/2native/2decode/2新down/6up，0通信/attention/DiT。独立CPU6.672秒complete（SHA881e88b516c2edcedcac352be77ab1ea3ff1a5af9fb70e2f3bf014309710d7d0），原/side重放零漂移。side/decoded valid输出NMSE5.9324e−8/1.83369e−6，报告036；关闭C006独立论文主线，保留工程实现与信息证据。不新增GPU网格。


## E041 complete — 2026-10-03

六条E038拍手媒体，保持原124帧/24fps/576×1024；CPU逐帧页渲染26.795秒，48PNG/744帧，0模型/0GPU。两agent各一seed全三臂，r0另补8张原尺寸静帧，非实时播放/盲评。BF16可辨开合22/20；低位仅给保守下界/未知，前段不支持统一频率降低。每事件闭合区间和前后证据帧、不可读区间保存在两个observations.json/md；原媒体/评分保留。报告037，D066关闭当前动作变慢解释；无E042 GPU计划。


## E042 prepared / implementing — 2026-10-03

固定E038六条拍手原始PCM，10ms RMS与20ms Hann高频能量、hop2ms；仅CPU全长/固定早段图，叠各自E041视觉事件区间，不检测/调峰、不拟移位、不声称已听音。先判断事件读出是否成立，再决定后续；0模型/0GPU，180秒CPU上限。


## E042 complete — 2026-10-03

一次固定信号读出2.540421秒，CPU/PTQ现有环境、CUDAfalse、0模型/0GPU；6原始PCM→6NPZ/4图。原decode环境前置import缺matplotlib，未执行readout，未安装依赖。两zoom初版布局截标题，另render_h3_clapping_audio_zoom.py仅从原NPZ固定边距重绘2图（0信号重算）；原源/产物保留。root审readout源及两full/两zoom_v2，事件闭合区间数22/5/5/20/0/10正确映射，但0只是未标注条数而非零接触。报告038，D067：基线语义事件对应尚未建立，不给同步损伤结论，停当前派生。


## E043 prepared / implementing — 2026-10-03

原版Wan2.1非蒸馏基础模型、官方WanPipeline，4公开动作×2seed全8条，480×832/81帧/50UniPC/CFG6/shift8、VAEFP32、DiTBF16。共800DiT/400scheduler/8decode预算，GPU0–3命名tmux/1800秒launcher；worker与评价适配实现中，尚未GPU。两个Transformer shards与官方revision0fad780a534b6463e45facd96134c9f345acfa5b LFS SHA完全一致，首次检查6.00秒；TE/VAE继承E024。协议/manifest冻结，不用弱rCM异常作新量化loss依据。

E043执行更新：2026-10-03约12:25启动tmux e043_vanilla_wan，GPU0–3 worker2291435/36/37/38；4条首seed均已实际到30/50步。最终worker e51e1021…65479、CPU check complete；早期缺PyAV预检仅import失败/0模型，源与记录保留，正式使用现成imageio/FFmpeg。计数与完整媒体尚待结束，无质量结论。


## E043 complete — 2026-10-03

四worker生成390.408645秒、8视频81×480×832/16fps，800DiT/400scheduler/8publicVAE/8TE；评价70.909998秒，temporal2432523/MJ2432524均rc0。独立CPUsummary complete SHA e2912b4da3cafdfa27a4a99ad4289d30fab65330a538374f0e0afc10eabd0e3c，CUDAfalse；八实际noise唯一、16初终FP32、400标量步序、8媒体SHA及评价原始数组覆盖。MJ均值.629829/.662476/.040802/.601074，AMT.966857/RAFT7of8/DINO.926998。root已看全部八张固定九帧图，非实时播放/盲评；保留叠衣含混/汽车r1形变。报告039，D069支持全部样本的同模型native配对准备，不是论文贡献。


## E044 prepared / implementing — 2026-10-03

上轮E043为progress。固定全部八实际初态与正负embedding、原版50步/CFG6/shift8/81帧；plain/SVD两臂16视频/1600DiT/480000native主GEMM预算，0训练/0新TE。协议manifest已落，当前五checkpoint文件SHA首次绑定2.48秒；六worker分三GPU队列0/1/5、总3600秒。runner/评价适配实现中，尚未GPU。保持原版失败样例，整个配方差不能归因单独LR。

E044 attempt01 failed_preserved：launcher9.796秒；plain两worker第一DiT在SDPA计数wrapper关键字签名处TypeError。观测2 attempted DiT/6 native/0 SDPA/0 scheduler/0video；SVD加载中被监督器SIGTERM，无worker结果，不补造计数。三个PID2711630/36/42已退出、三卡释放。旧源不改，失败json/log/空产物目录归档attempt01并保留映射；v2仅修query/key/value签名，输入/配方/协议不变。

E044 attempt02：run_vanilla_wan_native_v2.py SHA b219e44365a67ab672b277f449a109d712bee07466b084786e70de8a371382a5，两臂真实CPU检查及SDPA关键字/位置调用检查通过。13:01左右启动PID2772916/22/28、GPU0/1/5；plain首例各10/50，SVD低秩加载中。其余三worker由相同监督器排队，尚未完整媒体/评分。


## E044 complete — 2026-10-03

v2六worker正式生成1273.237114秒，16视频81×480×832/16fps；1600DiT/480000native主GEMM/96000BF16SDPA/800scheduler/16publicVAE/0TE/0training。全量评价72.616122秒，temporal3076312/3076313与MJ3076314及六生成进程均rc0退出。CPUsummary SHA a924257b1e38335e3e9b0b967bdabd6083c11ae573102d7d2dc5262ef51db1d5 complete/CUDAfalse，核24媒体与同输入/终态/标量链。MJ .629829/.209792/.563575，SVD比plain 7/8更好，较BF16 fineness全8下降；root全部16固定九帧观察完成，非实时/盲评。报告040，D070；旧attempt01原样保留，无新训练/新GPU实验。


## E045 prepared — 2026-10-03

真实同teacher末步干预，原版50步完整prefix两seed；BF16实际CFG和同UniPC历史的native末更新后，teacher/native_terminal/first_only/rest_only四角各解码。204DiT/1200native/12240SDPA/104scheduler/8decode/0TE，GPU0/1，900秒worker/1200秒launcher。预clamp与postclamp分开，latent21片/RGB81帧均按每元素/每帧统计；不把因果VAE已知帧不均衡当新颖性。实现/独立CPU读出并行，尚未GPU。

E045 executing：tmux e045_wan_terminal，GPU0/1 worker3451474/3451480；两CPUcheck complete/CUDAfalse，执行源SHA1dda5f84170b4b0e0d1b8e57d4aeb0d82f1e8983bfa7bb2c3429296a2385d947。两例实际teacher已20/50步，尚未解码/结果判断。

E045 attempt01 failed_preserved：两teacher完整50步/合计200DiT+100scheduler，末步capsule已保存；诊断signature的单元素int64零stride byteview错误，0native/0decode。launcher174.673秒，两个PID退出。保留原source/记录，v2从真实capsule恢复末步，避免重复teacher；科学四角不变，原final未存不能自比较宣称shadow验证。见E045_recovery_amendment.md。


## E045 complete — 2026-10-03

v2恢复launcher191.497367秒，worker3605018/3605024 rc0退出；继承200BF16DiT/100scheduler，新增4nativeDiT/1200GEMM/240SDPA/4scheduler/8decode，合计204DiT/12240SDPA/104scheduler/8decode，无重复teacher。CPUsummary_v2 10.680502秒complete/CUDAfalse，SHA703370b55e02cd96d5e9ac2684f1fc2f15b61da5b19b7bbd096b00c9688b3002。两恢复B对E043独立final相同，原E045final未存且未自比造验证；四角与八raw preclamp公共输出绑定已核。root八张九帧+两seed误差图已看：单末步可见碎片，首位置传播弱、rest自身输入误差更大。报告041/D071；DSAQuant与PTQD直接近邻，下一强BF16末步控制尚未执行。最初CPU reducer native环境缺matplotlib失败保留，切已有MJ环境同源check成功；无安装/无GPU。


## E046 prepared — 2026-10-03

E045为progress，现冻结完整native轨迹末步BF16修复强控制：原4动作×2seed全部保留，16同轮native_full/bf16_last媒体，816DiT/240000GEMM/48960SDPA/408scheduler/16VAE，无训练/新TE。独立原BASE BF16替换整套末步配方，输入和history来自量化末步之前；同时记录双驻留成本，不把调用比例当速度开销。GPU0/1/5四worker队列，1800/3600秒；实现/评价准备中，尚未GPU。

E046 executing：tmux e046_wan_terminal_repair；源SHA7a7b9fa594b98ec57c64a9231b302bc44542214d27fe97904e72e55c7e404775，四CPU checks complete/CUDAfalse。GPU0/1/5初始PIDs3907422/3907429/3907439，prompt316待GPU0前组结束。原始kwargs在root无prehooks保证下、native_forward前clone，真实pre-last history保存；独立原BASE BF16。并行候选查重以1query找到GCBT对CFG分支common/difference变换直接覆盖，停止该候选新方法投入；未增加GPU/新算法实验。


## E046 complete — 2026-10-03

四worker生成1126.007900秒，16新81帧媒体；816DiT/240000nativeGEMM/48960BF16SDPA/408scheduler/16VAE/0TE/0训练。评价72.583779秒，三PID4183688/4183689/4183690与生成3907422/3907429/3907439/4035944均rc0退出。独立CPUsummary complete/CUDAfalse SHA e1129e856ada4029df64e9b64fc0e8145e97869cb989dfee2167f30a3cf31a72，八native final对E044零漂移。MJtotal .563575→.640600、alignment/fineness同8/8提高，但对原BF16 total/fineness/coherence各6/8低；保留末步控制及全部视觉缺陷。备用BF16新增allocated2.650625GiB，未作速度claim。root全部16九帧在评分结束前查看并记录。报告042/D072；没有执行失败或重跑。


## E047 prepared — 2026-10-03

E046上轮为progress。匹配原版81f/480×832/UniPC50/CFG6/shift8校准，保留rank32/group16/g10/64records及原生两级scale。独立selection_audit确认旧真实1600=虚拟1600、Random0所选64与manifest一致；实际14提示35时间步35cond/29uncond，未含0/48/49，原策略明确保留。四采集worker预定GPU0–3，1400DiT700scheduler28TE0decode，1800/2100秒；后续完整PTQ预定GPU4，21600/21900秒上限，旧成功参考时长1h59m47s，无旧峰值记录。当前实现准备中，尚未GPU。

E047 collection executing：启动前GPU2–4出现其他任务context，资源调整为GPU0/1/5三队列，四数据分组/worker源/manifest不变。新launcher_v2只调整调度，v1未运行保留；named tmux e047_wan_matched_collect，初始worker525067/525070/525079，group3等待一组结束。CPUcheck complete/CUDA未初始化；worker SHA6cb53a8c9af6328d30383c9b4eb2202805db24afd8120cf02ba53d7ded3f304b。实际首prompt已进入50步轨迹并保存指定cache；尚未PTQ/质量结果。

E047 PTQ入口CPU预检complete/CUDAfalse，5.218秒；当前native环境直接原版DiT结构300target，rank32/g10/64records/batch4/sample_size=-1和two-level scale已核。仅适配已关闭的历史gated缓存及RoPE tuple，未GPU。ptq worker源SHAb11b7bb024507868373b3eec1314509cff8584bc51c4cb68a7b3ede3fe2c01e0，完整PTQ须collect_launcher和collection_summary都complete。

## E048 prepared — 2026-10-03

匹配校准checkpoint完成后复用E044v2完整native生成，E043实际四prompt×两seed固定输入不变，新增8视频/800DiT/240000GEMM/48000SDPA/400scheduler/8decode/0TE。只写薄worker/launcher与manifest/plan，py_compile通过，尚未权重检查/加载或GPU。任何check/run须E047 ptq_run complete且identity匹配。已有E043/E044/E046参照不重复推理，新增媒体后再做同评价；不是新方法或独立泛化集。


## E047 collection complete / PTQ executing — 2026-10-03

四采集worker525067/525070/525079/642354均rc0退出，launcher1061.907233秒；实际14提示、64cache、1400DiT/84000SDPA/700scheduler/28TE/0decode。独立CPUsummary首次complete，3.929955秒、CUDAfalse，SHA0644642126bd3c4a2c9d685a2a54a5177030adae48c35cc61b1829daa0d1a611，直接核64实际I/O/branch/timestep/embedding、14初态seed重放及末态。max prompt allocated16,434,568,704B、reserved17,563,648,000B；不是推理性能benchmark。无失败/重采集。

named tmux e047_wan_matched_ptq已在GPU0启动，worker818406，21600秒hard deadline；ptq_launcher.json记录实际deadline。原版DiT载入成功，已进入smoothing的collecting acts info（真实8/64时已观测），尚无层完成或checkpoint。后续以实际PID/logs判断进度，不能把锁或running文件当存活证据。E048生成/评价入口已准备，仍未新权重检查/生成/评分。

E047 PTQ实际首层适配通过：blocks.0的cos/sin shape均[1,32760,1,128]、FP32、原本在CUDA0，未搬移；156.07秒时layer ready，进入attn1.qkv smoothing并完成original outputs计算。此为执行进度，不是层完成/质量结果；worker818406仍live。


## E047 PTQ complete / E048 six videos complete — 2026-10-03 21:18:35

E047完整PTQ worker19228.123446秒、launcher19239.203601秒complete/rc0，818406/818402已退出；五产物3,273,805,991B，GPU0峰值allocated25.409592/reserved29.904297GiB。有限独立CPU读元数据、源/配置和五文件stat核验通过，原BASE/rank32/group16/g10/64records与采集绑定一致；权重SHA继承生产者，无GB权重重哈希/张量加载/GPU。

E048原chain隐藏CUDA预检complete，21:07:01开始生成；21:18:35 prompts161/192/269已complete，600DiT/180000native/36000SDPA/300scheduler/6decode。prompt316 PID1020311 GPU0 running，chain1173926/launcher902948 live；保留原自动顺序，不重复手动启动。尚无新完整评价或质量结论，最近完整科学结果仍042。

Wan14B资产监督3654981/worker3654983 live，8/12分片verified，完成文件加partial41,821,043,967B，deadline22:18:36；仍为下载进度，不称完整资产可用。


## E048 complete — 2026-10-03

原chain全部阶段rc0：CPU check、生成1034.949101秒、评价71.070493秒、独立v2CPU汇总。固定八例81帧全部保留，800DiT/240000nativeGEMM/48000SDPA/400scheduler/8decode/0TE，旧E043/E044/E046参照未重复生成或评分。summary SHA570b45fc00b20f39ba768a7f1f435f9e007ab20c78a3eaff1c334266a82c2ca0，CUDAfalse；八实际初态/embeddings、末态/400标量步/800调用/媒体和新原始分数绑定已核，不重算中间轨迹或大模型hash。

matched/old/BF16 MJ均值.583694/.563575/.629829；比old 5/8提高、比BF16 6/8总分和7/8细节低。车r1动态true→false且总分大降，DINO反升。root全部九帧图在读取新分数前查看并保存，不称盲评或完整动作标注。报告044/D073：保留匹配校准基线，停止扩该网格，准备更大原版模型有效参照；无失败重跑、新loss或可投稿claim。


## E049 prepared — 2026-10-03

上轮E047/E048完整结果为progress。固定原版14B BF16完整八例参照，继承E043实际noise/embedding，不重新TE/选seed。HF revision38ec498c配方CFG5/shift3/81帧480×832、显式50UniPC，16fps本地评价约定；原生Wan CLI shift5属于另配方，不混用。40层/每DiT80SDPA，总800DiT/64000SDPA/400scheduler/8decode/0TE/0native，GPU0/1/2/3四队列，5400/6600秒。协议manifest SHA8cc2c92fc497445b7f22290f2221073ae7699b1983ccb510786f1a1622716849，runner/eval/reducer实施中，未载14B或GPU。官方分片下载仍live；正式CPUcheck和生成须完整download+supervisor rc0及header/index核验。该实验为必要原版参照，不是新claim或纯模型规模因果。

E049执行准备完成：root审worker/launcher/adapter/eval/reducer并发现case.embeddings格式不一致，在运行前统一artifact三字段；源已冻结。评价CPU结构check complete（SHA0c875d9f18f6060352c4e75abc1cefde4741f42bcdd746d52ba81f66d9c3d4e4），chain结构check complete/0模型/0子进程（SHAc22b103ca01709d385863554892cd6e8e1e2aac2b9fce8ceab8dad4aa5cdfbf6）。22:01左右启动named tmux e049_after_assets，22:02:18 PID1645096实际live、stages=[]；仅等待原资产worker3654983/监督3654981完成，尚无GPU/14B模型加载。正式CPUcheck→八例生成→评价→独立汇总由原chain单次自动串接，不并行手动启动；原下载已有11/12分片校验通过。

E049已实际生成：14B下载完整rc0后，chain自动CPUcheck complete（0.252308秒，CUDAfalse/0model，8actualinputs，SHA22a0a5f3091bf958e317672ddc835b34e1162e3553bf5f508fd24f3b2c09dae3）；timestep0/25/49=[999,750,60]。22:06:42原chain1645096/launcher1683283，GPU0–3四worker1683412/1683596/1683773/1683887实际live，12/12shards载入完成，各约34665MiB显存/100%利用率。尚未完整视频，计划计数不是已完成计数。原下载两PID已退出；根节点资产header/index/small SHA/size独立复核complete，1095张量/14288491584元素全F32。


## E049 complete — 2026-10-03 22:37

原自动链生成/评价/CPU汇总全部rc0。八例/800DiT/64000SDPA/400scheduler/8decode/0TE/0native实际完成；生成1897.128619s、评价70.902164s，summaryCPU1.373173s/CUDAfalse。summary SHA48e880410ce36e3c6a4be75fbb0f5ba43c9f15136967678dd7262cf09d6f3dff。root八固定预览先看后读分；动作局限保留。MJtotal/alignment/fineness/coherence=.851667/.912109/.072021/.650879，AMT.976930、RAFT8/8、DINO.914935。不是尺寸单因素实验或量化改善；见报告046/D074。

## E050 prepared — 2026-10-03

冻结14B匹配64records/14完整轨迹，只采集，不自动PTQ。CFG5/shift3沿E049，E047原样本政策和同身份真实embedding复用；0TE/0decode，计划1400DiT/112000SDPA/700scheduler。GPU2/3/4/5，worker5400/launcher7200s。manifest SHA3ce29619bd5aeb2d2320511ff324b359aeb16f3bd9d4fcc748f95a13238bb646。新collector正式CPUcheck complete0.490415s，CUDAfalse/0weights/0calls；准备launcher/reducer，当前未开始采集。

E050实际启动：named tmux e050_wan14b_calibration，launcher2208229，GPU2/3/4/5 worker2208973/2208978/2208981/2208984。22:48:49均Rsl且首提示10/50已记，真实已保存6cache，完整轨迹0；不是仅launch成功。root运行前审查修正新launcher旧runner路径残留，未执行失败或重跑；final launcher SHAfffc3a070b96836b95629195197ed606ec611bda4e37087b2df3b801c316fe39。后自动CPU汇总，不自动PTQ。报告047。

E051协议准备：14B plain与E049真实输入/采样完全配对，400模块native、无训练/smooth/LR，八case。manifest SHA9f0469e89d34032e7ae80c47e0c91f362f92eca2603976be51966158ffa1e47a；GPU0/1各串行两个worker，5400/12000s。逐层packer已CPU meta确认40层400模块，不代表GPU验证。worker/评价/summary实施中，尚未GPU。

E051实际执行：正式CPUcheck0.375298s complete，CUDAfalse/0weights/400meta目标/8实际输入；named tmux e051_wan14b_plain，chain2360038，generation2361551，GPU0/1 worker2361947/2362130于22:55:37真实Rsl并载入12shards。另prompt269/316同卡排队，不允许并发叠卡。自动原链继续完整生成/评价/CPU汇总，尚无完整native视频或新分数。E050同时GPU2–5运行、当前15cache/0完整轨迹。

E051 22:58:25首次实际40层native进展：prompt161/192 worker均Rsl、首例10/50步；每DiT400native/400pack/80SDPA runtime检查均通过，未出错/重试。未写完整case receipt不等于失败；该worker在完成视频后才写receipt。全视频与评分尚未完成，继续原chain，不手动启动替代。


## E052 prepared — 2026-10-03

上一目标turn为progress：完成E049科学参考报告，E050/E051已真实运行；本轮23:00重新ps核全部worker存活，E050首4完整轨迹/23cache，E051两例20/50步。不是仅凭旧状态声明等待。E052冻结完整预算首block资源pilot，全部14B W驻GPU、64/b4/g10/r32/LR100earlystop，smooth后重新采集activations再LR，后层early-stop不减首层数据；不是质量替代。manifest SHA23a67000b31102a6bfbd31ed77c00f1b6f4032b724b8fc8b67fc775fbf0c6202。先实施/CPUcheck，实际须E050完整+GPU4空闲，未GPU/未全PTQ。

E052正式CPUcheck complete：5.347893秒，CUDAfalse/0weight，40block/400main/首10实际meta与64/b4/g10/r32/LR100earlystop配置通过。check SHA3d40931d4ce5234da45685793c8520ad619ee1d235f89c88ca6897198c11358d；worker9ce43f14ef2cea3833abdc4f41b9622580a4f4871f8d62e685636b25981c2dc2，launcher5b6c2d43e5a715351efc9ab3a8ab831cc7bd058e4a53c3657b7b7b1d3d6a43e7。named tmux e052_wan14b_ptq_resource，监督2653260于23:17真实live S+，waiting_for_collection，无GPUworker。等待原E050完整独立汇总再启动，不手动替代。root修了未执行launcher的source record字段比较，CPUcheck和GPU无失败重跑。

E051首2/8视频已完整，各100DiT/40000native与activationpack/8000SDPA/50scheduler/1decode/0TE。23:09 root看plain及匹配E049四张图，写E051_visual_observations，未读取新评分；实际pack400模块allocated27.242→8.405GiB，conversion peak27.354GiB，0.96–2.20秒。剩余视频继续，非整体质量结论。E05023:17已8/14轨迹、45/64cache，四worker真实运行。

E051 23:29已4/8，root在读分前追加r1两条及BF16对应预览；原前两worker退出，原队列2692208/2691457正在GPU0/1各第三提示首seed30/50步。实际已完成400DiT/160000native与pack/32000SDPA/200scheduler/4decode。E05023:28已55cache/8完整轨迹，E0522653260真实等待，无GPUworker。

本目标turn为progress：E051新增四条视觉观察（累计6/8），均在读新评分前；systems只读比较已完成前四条实际完整pipeline成本，时长和−21.14%、推理allocated40.035→21.197GiB，加载/转换已见峰27.354GiB。只是plain诊断基线成本，非新claim/等质量优化。E05023:31已12/14轨迹、56cache，GPU4/5原worker退出，GPU2/3最后提示继续；E052原2653260等待全部完成。未另开GPU任务、未改冻结源/设置、无新失败或重试。


## E050/E051 complete，E052 actual GPU start — 2026-10-03 23:50

上一目标turn为progress：新增E051四条视觉观察及前四条成本读出。本turn原PID核实后继续，无重复启动；E051八条全看后读分，全部生成/评价/CPUsummary rc0，agent独立原始指标复算通过。actual800DiT/320000native+pack/64000SDPA/400scheduler/8decode；summarySHA816751ceec00b03cb4daf94a0fe14762ad08e352d96bedd1ebb25b4422910173，详报告048/D075。E05014轨迹64cache complete，actual1400DiT/112000SDPA/700scheduler/0TE/0decode；wall3702.531961s、CPU3.876798，summarySHAd77f49d35b2c3440861168e5769d43ba68e53db8b3c61cf44bd2ccaa5c362886。上述原链与worker均ps核实退出。

E052原监督2653260等完整汇总+GPU空闲后实际启动3055486/GPU4，23:47 Rsl；model_load complete9.7668秒，smooth_cache运行，尚无完整pilot。未编号完整PTQ薄worker与SVD原生helper已准备CPU/meta（40/400，SVD280共享组）；首次独立CPU命令PATH缺ninja失败保留，沿现有环境重试通过，未GPU/安装/改变执行源。完整PTQcheck_v2 SHAa6b73f2fd785fbca1a48185e25974da6c2b3ee822228e67d4e3e65ffed06d8b7，workerSHAb19ae72ee275c9bd4c90cb03ecbdab5fe8fe55495033902de484d10ecbe2c1a0；SVDhelper f15dc67d5922d4fa7092bf6ad11fb42a6b466c88c7302dffd6173a391a78ceba，真实checkpoint/forward未验证。

E052 23:52:58实况：3055486仍Rsl，smooth_cache complete150.001361秒/GPUallocated42.135272GiB/CPU RSS236.736137GiB，actual16prefix/32block0/后39层0；smooth运行，暂无完整搜索或LR结果。14B SVD helper正式CPU结果另保存results/research/wan14b_svd_entry/check.json与日志，40/400/280、CUDAfalse/0checkpoint；第一次PATH失败只有原工具输出，不伪造补写失败文件。

上一目标turn为progress：E050/E051全量完成、报告048/D075和两薄适配CPU准备。本turn23:57/23:59核实E052原3055486/2653260仍真实运行，无重启。smooth已实际完成前五共享组，19候选/组，耗时322.9597/76.1160/94.1895/61.8644/75.3917秒，FFN-up继续；总smooth/LR未完成。新增有界代码证据确认同阶段下游输入在yield校准前生成，可按阶段探索分层执行，需全smooth→LR屏障、真实prefix、RNG政策及CPU峰约束；见wan14b_ptq_phase_dependencies.md，尚未实现或另开任务。

E052 10-04 00:06:53原3055486 Rsl：完整smooth complete963.082329秒，7共享组/全部10主Linear各19候选，64/b4；GPUallocated61.854585GiB、CPU sampled RSS367.461533GiB。实际FFN-up140.117694秒、FFN-down189.784439秒。lr_cache正在重采，低秩尚未开始。两并行峰简单叠加近整机容量，不直接启动两个完整worker；保留原任务/预算。该资源测量仍不是量化质量或新研究贡献。

00:09:03 lr_cache已complete164.753951秒，3055486真实Rsl，low_rank阶段实际self-QKV QuantLowRankCalibrator已进入，64/b4、max100，尚无完成LR组。原worker继续，无重复启动。

10-04 01:28用户要求明确每次尝试的motivation，root补具体研究/停止条件而非新增方法。E052已完整1647.210秒、runSHAe515e1f83db689112c150589519bc1e5119d22c917582b5264b819ce613bc9d5，parent complete/rc0且退出。E053协议和正式CPUcheck准备完成；启动tmux命令被用户中断，实际tmux has-session和pgrep返回不存在，launcher/run receipt也无，尚未执行整模任务。不要把该中断记成校准失败/已启动；后续继续前仍以实际状态为准。


## E054 / E055 — 2026-10-04 H3主线恢复

E054融合通信强对照已写计划，因用户强调工程基线不作创新，停在预备、未GPU执行，不计作完成实验。E055同状态输出几何诊断已预写计划：三状态×plain/SVD，以固定head解析null检验是否值得更强控制；先CPU结构检查，冻结manifest再分析，当前尚无结果。

E055完成更新：summary complete0.160727秒、CUDAfalse/0model forward；SHA5cdb65919a1ffa4d30ecf5a001f48cbe68f9c6d9825b3a0d66f6df856161f33f。独立统计复算complete，三方向六组及各37时间片通道均值phase差低于head null；停止patch边界特殊损伤候选。无新质量/性能实验。报告050/D079。


## E056 / E057 — 2026-10-04 H3时间轴判别

E056计划先写，12现成输出CPU分析：v1在E015实际输入包含tensor的字典相等比较处失败，未写summary；保留源。v2仅增加递归torch.equal，统计不改，complete0.433281s，0CUDA/0新forward。SVD video比例项移除后cos .701306/.621988，独立NumPy误差≤6.11e−15；D080/报告051。E057预写5forward上限计划，runner准备中，尚未GPU启动。

E057首次GPU0执行30.633秒：历史teacher完整replay raw/velocity/endpoint全部精确通过（1 complete），第二次attempt在prehook比较失败、模型尚未执行，worker已退出。源/失败evaluate.json/日志保留。根以空模型sentinel在GPU重现同样前三组打包，全部hash一致、0DiT，排除通常packing差异。审查定位prehook立即读取non_blocking GPU→CPU复制buffer的竞态（旧E015在forward后同步才读，不受此处新检查影响）；v2将改阻塞capture并保留严格检查，复用成功teacher，在原绝对deadline内只执行剩余4forward，不重启科学预算。

E057完成：v2 CPUcheck1.195s、CUDAfalse；原deadline1791054484.3786852，GPU0剩余864秒时启动v2。新增4forward complete40.146767s，累计含先前replay5次；峰40.895983GiB。原v1 prehook失败只计attempt，不计DiT forward。evaluate_v2 SHA2ab8c573a5147d1415fc17129b9cc296842c46cc3cce85b5f97397abeccf1575。CPU四角summary0.817292s、SHA2a704f620fce8439d118ffa475bfd2ce9905908bead091d8896ec5ea08b020c7；预设弱交互门槛失败，有限两步正增强保留。所有进程/tmux已退出，GPU0 0MiB/0%。D081/报告051。

E057独立检查完成0.489s/0GPU，NumPy八份PT直接归约最大差1.68e−11，原始与centered均确认正增强且弱交互gate不通过。independent_check SHA44093d36fae15d6dc9aeefd5f1efd63663c36d37688399211e2a9f4f5f89258c。本目标turn为progress：完成E056与E057、得到有限阳性及方法假设不通过的决策；不是wait或外部blocked，目标继续active。


## E058 — 2026-10-04 第二步更新舍入分解

先写计划后分析，CPU complete2.023906s，0CUDA/0新forward；原BF16 CPU更新对所有saved角逐元素一致。固定实际输入/velocity，仅去掉第二次采样算术，SVD video I/(F+L)为.305083/.412913；raw/centered均>.2。独立NumPy0.786s检查SVD8输出/16模态角，最大差9.51e−12，independent_check SHA3e680926ea271ee90524908685e3b389d9c0a87f2d5dd19c57aabc90006b0503。保留模型精度/第一步舍入局限，报告052/D082；未修改E057阈值或既有结果，未GPU生成/采样精度重跑。


## E059 — 2026-10-04 native激活残差oracle

预写12forward/1800秒计划；CPUcheck complete1.160秒、6actual input与历史一致、CUDAfalse/0forward，SHA0421077ef18cfe9134464e45818f8e6ad0684f9e8f72e8808d89b80d4a09b395。runner SHA7d91212dc09af739738d9d2cf1ff332007350c91f9ee4a9c969b81d7c721354b。GPU0复查0MiB/0%后启动named tmux research-e059-h3-activation-oracle、PID1873935，deadline1791057771.603179；先六zero精确旧结果，再六oracle，非W4A16或部署方法。当前前3zero通过，剩余继续原任务。

E059实际停止：82.838965秒，6zero全部raw/velocity exact；首oracle在block0 fc2固定sample加回舍入/补偿RMS .10338438693≥.1时failed_stop。7attempt/6完整DiT/0完整oracle，未运行主效果summary。四sample FP32/FP64补偿公式检查均通过，故不是公式误实现的证据；也不证明A机制阴性。源/evaluate/failed_samples保留，不改阈值重跑。evaluate SHA cfdc5751e8c5078b22112432b77eb65fd8aa1b5333b581773c4b3b03a74da561。原PID1873935已退出、GPU0 0MiB/0%；报告053，独立CPU复核进行中。

E059独立复核complete0.420秒/0GPU：6实际输入与24raw/velocity数组逐byte一致；4sample独立NumPy FP64公式张量差0，统计差≤2.22e−16，确认.10338438693 resolution stop。independent_check SHA84c1597ba9029aeeaec635df496ef98c36221dfd45c4f7fedf4abfb1344cbb64。未执行主效果summary。


## E060 / E061 — 2026-10-04 H3低秩配方范围

E060预写只读核验计划（注明前期手动观察）后CPU审计complete0.724671秒/0CUDA/0forward。state SHA=e5ccf2d809f0b03e6a0eaef400e4f610ccfc37478eb3715d75040cf07446652d；200层600因子301414400 bytes全部相同，11.140GB packed主体未重hash。audit SHA683ec67dfc1ab8876290977aa20a08923a6d101dd3bd258b07fbf04fcae38052，runner SHA9cb5bf83ce8fc70a6081be668f82d3b6aa5a2b66caf4552a4ba71e875490812b。原生产源码未绑定，不反推历史流程；D084/报告054。

E061 fixed-smooth initialization plan已写，root/独立agent审查通过；两个新臂同seed/旧BF16Ws与residual表达式，完整200层四teacher主终点。先1旧replay+8新calls，主gate过才4shifted；无同输入plain-shifted参照，不错用E015plainQQ。单GPU1800s含导出、26GiB新export上限，非新方法/不自动fullPTQ。runner准备中，尚未check或GPU。

E061 CPUcheck complete2.339120秒，6实际输入及同输入参考/200原权重header通过，0CUDA/0forward。check SHA40cb0a3d1c086f4fca32d510cc074bbed2bf5f457cb0cd37e82ff446ac02cf61；runner SHA334ab1747c8aad44f3f309d0bbcaa873f65c2bf382da6a495811447f12d35d65。GPU0复查0MiB/0%后named tmux research-e061-h3-initialization启动，PID2319906于51秒时真实Rl，deadline1791059775.2187965。两臂各76/200已导出/roundtrip通过、累计8.467GB，0DiT；继续原任务，不另开预算。

E061 complete218.451683秒、9attempt=9complete，teacher gate false，4shifted未运行。双export各200/200独立QDQ roundtrip通过，共22280860048 bytes；峰37.611GiB。evaluate SHA101a5c0b718c5020947ebd54c6214e64fbd2903473e05e1b51a8db415503afa0。根CPUsummary complete0.56109秒/0CUDA，freshlegacy输入/raw/velocity再比较exact，四teacher新臂actual input均同旧zero，gatefalse；summary SHA5aed4e2cc5f325d2c2588cc3118190985688eeac1cbadcf277b4219f9fa7c0a4。原PID2319906/tmux均退出，GPU0 0MiB/0%。D085/报告055，另agent NumPy复核进行中。

E061独立NumPy复核complete0.823秒/0CUDA：21实际PT重hash、9actual输入及raw/velocity绑定通过，freshlegacy四数组逐byte一致；FP64直接归约最大相对差3.75e−16，0/4gate及9call/0shifted确认。independent SHA beb1640f025c4cea6fbfdb31714eb38793124633e1bfd0ba01752c25a98fed91。本目标turn为progress：完成E060身份/范围审计和E061原生完整对照，改变后续投入决定；无外部阻塞，无可投稿贡献，目标保持active。

## E062 — 2026-10-04 SwiGLU 成对残余局部判别

执行前已写计划、独立数学/数值审查完成。固定两状态三层、512 video 行，完整输入 native fc1 后采样，精确乘性交互通过同一高精度 down/AdaLN 映射，独立保存 BF16 四角舍入。主判据是带符号 net/total，而非交互范数；共同 teacher 投影去中心/比例，循环移位 eu 仅辅助。最多2完整BF16+6局部native、900秒、GPU60GiB/张量4GiB。runner准备中，尚未check/evaluate。

E062启动：v1 SHA806c73a0468c8735cfc9f5873aebfbd086447ee0aec081632e0fe6b22ecf3507，两CPU预检在缺av、错误历史文件名处停止（0forward）；check_missing_av.json/check.json保留。已有recovered site-packages追加sys.path末尾供av，不改native包、不安装。v2只修E014历史文件名及自身v2报告路径，SHA50285b1f236cbd5b15e4b0f5103d14ef116ca67303febe8fa63f032430a75bfe；CPUcheck complete0.458095秒/0CUDA，SHA5f5328d73631f2dc50db5cb2b33975b129d448d4498fb1e5a1f311458b66c138。GPU0复查0MiB/0%后启动research-e062-h3-swiglu，deadline1791060935.480792，launch.json保存完整命令/环境。研究定义及GPU预算未变。

E062 complete34.436929秒，2/2teacher+6/6局部native，44847441408B峰allocated，2665843784B张量。六格raw net/total全负（−.006712224/−.000172768/−.019718803/−.013006875/−.012349100/−.016189959），P-residual全负，BF16四角与actual算术uncertain全false，primarygate三层全false。evaluate SHA4f7eaf207ebbc2f9af2307f0a9b4d91713f0d1030cc8e71e99d08eac0f9df3e1。独立NumPycomplete6.350秒/0CUDA，12PT重hash，两teacher actual/raw/velocity一致，能量相对差4.97e−16，FP64样本相对RMS最大2.87e−6；independent SHA f14b606717b0649bec6fd292b697edda816b1ef8ff1d68e473da5871cbdfadda。进程/tmux退出，GPU0 0MiB/0%；D086/报告056，停止乘性交互修正候选，无训练/视频/MJ。

## E063 — 2026-10-04 QK 径向收益必要条件

预写计划、独立数学/近邻审查，0/24/49层×p30s5/p36s14。实际ComfyPruned layout[row,3,head,dim]逐byte核对norm输入，fixed-smooth rank0 freshQ(Ws)对legacyrank32，共用实际activationpacket。主看raw净收益、径向所占带符号比例及actual norm后收益，非attention功能代理成功。runner SHA7f1f1622138e5ce0f02a0d76f7fc7dd02429f41b2e7c4c7d86567a5d15796128；CPUcheck complete0.423865秒/0CUDA，SHAad33dda32b0704fd60fbd6038316bdfc6c5fed1c931ee390f9d3c9b7f243ccdd。GPU0复查0MiB/0%后research-e063-h3-qknorm启动，deadline1791062529.8293138；唯一GPU预算2teacher/12native/6pack/900s/60GiB/2GiB，无新attention反事实/视频/训练。

E063 complete69.667179秒，2/2teacher、12/12局部native、6pack；峰45068017664B、张量1146225056B。evaluate SHA8ccea55691215118281dc16839d6f6c34e7f9274efa6e62418493a52b2f30723。六格raw相对改善.01383243/.01027821/.00566220/.00453502/.01138306/.00628784；径向净占比−.049663/.028526/−.052035/−.281231/.003730/−.113290；norm相对改善.014597/.009846/.007131/.003841/.011685/.017601。三层primarygate均false，停止具体径向收益错配候选。独立NumPy3.832038秒/0CUDA，12PT重hash、两teacher实际输入/raw/vel及实际norm/RoPE样本均byte一致，统计能量归一差5.98e−16；independent SHA8d6bde8b46e87c43acb6d0465b99bdcbd78dee437fcce2b2fca11d83afa21c83。完整packet仅签名/执行源审阅，未CPU复现GPUkernel。worker/tmux已退出，GPU0 0MiB/0%；D087/报告057。无新方法claim、无视频/MJ/训练，目标active。

## E064 — 2026-10-04 真实深度路径四角来源诊断

执行前冻结计划与独立数学review：两原状态×plain/legacy各自真实路径，预选五块0/12/24/36/49。四角分离teacher innovation/BF16 transport/输入依赖echo，主signed net、共同模态centering，另报identitycarry。block0 exactzero，full路径和diagonal byte重放。最多6full+80local，960native/772SDPA，60GiB/32GiB/1800秒。runner/checker实施中，未check/evaluate；未启动后缀相干草案或训练，恒等式与一般跨层累积不计创新。

E064正式启动：root完整review后runner冻结d51eb4381903b25e0d6a6accec2fd63668e0e04378406cdf6498be4c654a7cd1。CPUcheck complete0.345183秒/0CUDA/0forward，SHA7fb1ea4d6c51c459229733fc5c3cb7ab7f8bfbf535fba40598a375e4327145a9，原有av依赖追加查找路径不安装。GPU0复查0MiB/0%后named tmux research-e064-h3-depth启动，pane3565041存活/模型已装载，deadline1791065372.3453302。实际共享10BB对角重放，计划6full70local/960native+pack/752SDPA，低于冻结上限。launch.json保存完整命令/环境；独立NumPychecker准备中，无另开GPU预算。

E064 complete467.428615秒，6/6full、70/70local、960native+pack/752SDPA/0disk；tensor24545642978B、峰43919859712B。evaluate SHA514f2f21e16bc23904fbbceb9338481bb0f3fe3ce75640c81419ae0ff4546694。16非零video格raw net/out全负−.00338054至−.03385586；center全负−.00465236至−.03432089。四block0精确零，plain/SVD各0/4通过深度；停止具体稳定净放大动机。原PID3565043/tmux均终止，GPU0 0MiB/0%。没有工程失败/重跑。

独立NumPycomplete175.933192秒/0CUDA，104PT freshhash；6full actual/raw/vel、完整metadata、70local/四block0零及全模态Gram通过，最大统计相对差6.71606e−14，raw=center+bias残差2.79e−15。independent SHAde13a3d24d89155caf436649c550f8b2b7cf8944dfdd4f4a2ff821538084d175。root追加posthoc interpretation.json SHA2adb2190eaa5dbddf7875d066fbbae85f64cf2f6b2c8ca8b736ed8064c852b18：net=[||p+c||²−||p||²]+2<qB,c>，前项在16格raw全正1.833%–13.436%，后项全负−16.027%至−2.171%；独立Gram复核闭合差1.12e−15。故不把net负解释成量化map更contractive，不改原gate。D088/报告058，目标active。


## E066 — 2026-10-04 carry-Q 局部迁移诊断（运行中）

动机来自E065b全部校准局部格改善与四点整模video SSE反向。执行前审阅加入同packet真实M512控制，分开输入、整个线性层M形状算术及token覆盖；无拟合/新候选。协议在首次CPUcheck前固定为4BF16、3200local（full/subset各1600）、800pack、1800秒/60GiB/96GiB，源cc9041fc8f9094306cee0deded3cc31ff36d8a6da7a82e84043b1f87d22e9886，plan4bbef3feaf99ac0192dd26ebbb1ce845d0c9ac3dd3595dbbd7afba9d2e5afcf1。CPUcheck35.382482秒/0CUDA通过，SHA2269517b4cb00d00890017df09feab6863760ed36f1b837bd3ebd21d61095707。GPU0空闲后tmux research-e066-h3-local-transfer启动，deadline1791076815.6678288；launch/supervisor保存命令及退出码，不重复启动。独立checker预备SHA24d99aa7a31f28be9d96d97e136b9c839d2aefe344393a105883d708725d06f4，尚未执行。仅诊断、非创新或视频质量结论。

E066 GPU已complete350.657461秒、exit_code0；4full/3200local（1600+1600）/800pack/408SDPA/0disk，峰46949702656B、数据71877520775B。四BF16重放一致，GPU0已释放。独立CPUchecker已启动，PID1993758，未终结前不将数字作为独立验证结论。

E066独立CPU已complete247.513912秒/0CUDA，812PT/400导出验证，3200保存输出512行通道及全部per-row归约通过，最大相对差5.8134e−16。三模式四点video各200/200更好，full中位比.969639403/.969812696/.970606081/.970469527；joint/text全更好，audio两block3.fc2例外。描述汇总0.058611秒/无torch，summary SHAf42d86fdd0f36b76e8c854af36ded8bf06421295098613590f85d3ef932c7d4c。所有GPU/CPU任务结束；报告061/D091，不据此自动开新方法或GPU。


## E067 — 2026-10-04 保存误差的幅度/方向描述（CPU运行中）

动机为E066逐层SSE优势迁移但整模排序反向；只检验“近似径向缩小原误差”的模型是否成立。E066四状态×200层固定512样本video/joint、E065b四完整video/audio；raw/固定去均值控制、full-M与真实M512、直接FP64正交分解及同权重形状δ差异。0CUDA/0forward，不推因果/质量/新方法。执行前独立静态审阅通过，源1894392701d4ade6c2b53a4b57589c4ba419e6f403e7d0910012ee06c5dc3f11，计划906eeaf44b8f2537dbb60c7d2dadaa86d2908421c6072d8bbefb50a033ace122；一次600秒/8GiB/20MiB，外监督615秒含清理。状态见results/research/E067，不重复启动。

E067已complete551.028857秒，0CUDA/0forward、812PT、RSS2274955264B、exit0；全部800局部video样本差值由正交部分主导，中位.8566，形状控制能量最大.000324708。全局cos.65038/.62596/.60914/.57987；仅否定径向近似，不推因果/画质。原SSE复核与恒等式通过，未另程序全量重算新方向统计。analysis SHAf414e380aef5a30855082458e07c11e28314ddd79afe15b895e3b9e05236f776。报告062/D092；E068准备完整视频而非追加SSE机制扫描。


## E068 — 2026-10-04 冻结carry/restart完整视频任务相关性核验

执行前冻结4视频/2原prompt与seed、E010原embedding/noise/20steps/CFG1/decoder，2额外teacher replay，共82full；2GPU共享900秒/60GiB每卡/总10GiB，0TE/0新BF16forward/MJ。CPUcheck complete14.413961秒/0CUDA，SHA3c2db2cb40e3cded4d7393c739366d76ba631b48c17cec4ad02b4e9c1c54c08d；runner73be5bdd147e20f9fbebe772d89902e65f3ce9da2f996e5fe4fd3009c5e97aaf。并行agent在交接后加了不必要的CPU验证改版，root启动前检测并另存未执行3fa4版本，精确恢复CPU绑定73原字节；audit在results/research/E068/source_race，CPU结果未修改、0重复forward。

GPU0/1启动前均0MiB/0%，named tmux research-e068-h3-restart/carry，shared start1791078059.7994795/deadline1791078959.7994795。两臂p30s5历史重放已byteexact，自由生成进行中。A/B映射生成前固定、private保存，public commitment SHA02c5b1b009ddafa1ca9677ace261713a1448a6fa9ad8f02cf670da1d46ca11db；不得在用户盲评前泄露。独立checker和盲评页面builder准备中；尚无质量结论。

E068生成已全部结束：两臂四阶段complete、两exit0、GPU0/1释放；349.417秒内完成82full/16400native+pack/8364SDPA/160updates，4video+4audio VAE。denoise两臂300.421/297.812秒（进程边界含启动/退出），decode48.888/51.362秒，峰37.6044GiB/卡，新数据1096603006B。独立checker首跑complete4.256021秒/exit0，160新step/80历史schedule/496视频帧/PCM/两replay通过；SHAe3ae5d66c81bc758ffec5992d9b7d681220ed4238f7fe8e16567086c9d558672。checker源b206556ebb790ab964c3edea3d2c327282697d32aacd946196e1c4feaf4ed2dd冻结。

blind builder首跑exit0，四新视频＋两历史BF16生成中性素材26157199B，逐SHA验证、0重编码，public results/research/E068/blind_review。builder源2d8f3b49c8e5bb4c9865f2944436701efd46fb300b2b4b023ce62610bac4beb4及原输出冻结。人工偏好尚无，不泄露映射。页面控件异步播放取消问题另写展示修订，原视频/执行源保持不改。报告063/D093。

E068展示v2已生成，refiner710b74adc702db28e3bd6d868d4240f953ca8bac3015904a474769408ab7baab冻结；Node语法pass，6媒体/manifest byteexact，旧输出不改，未浏览器实播。26.19MB离线包SHA5d8ba9bdf11e037370a443bdc74dcdc5a5eeb3328973bcad9d91fe241b114f0d。已向用户请求两个案例A/B/难分和理由，review_status.json记录待反馈；0人评/无质量结论。


## E069 — 2026-10-04 H3强基线配方合同CPU审计

预写计划，固定upstream commit69f3473f5e1c1504bae35cc50c7858ef900a9b17；GitHub API限流后用git ls-remote成功获取HEAD，raw下载12文件243155B/source_manifest complete。第一次非实验配置导入因缺PATH下ninja退出，未编译/无模型运行；不安装依赖，正式审计直接执行纯AST方法。唯一正式audit首跑complete0.032997秒/0torch/0GPU/0forward，legacy9、同规则g10=19、defaultg20=39；绑定候选函数、默认配置、FP64完整vsFP32随机分解与量化YAML差异。audit SHAaa993946fbf01fc428067ba9a0d6cb59befec0cdbf7b5bbb4fd79de10480faf0，源/计划/结果冻结。未推质量优劣或新方法，报告064/D094。

E069独立复核complete0.027004秒，31文件freshhash，Fraction重建候选集合与顺序通过，0torch/GPU/forward，未重跑原方法；SHA6a98b15801658cb7ca2e5b56072063756b7089005a8892b1c779247d946b167d。只覆盖源码/枚举，不证明实际scale互异或模型效果；本阶段结束。


## E070 — 2026-10-04 H3基线真实入口与精确SVD（complete）

预冻结协议两臂A900秒/B1800秒，真实既有环境import1.516495秒、0CUDA。B CPUcheck0.003984秒，runner a400967abd67e342d9e03d2d9e00f503538460f34491875406ce1c888c6b62a6；唯一GPU1启动1791081483.4047542/deadline1791083283.4047542。SVD59.398311秒、总61.284249秒、峰11224525312B；evaluate SHA55b4b7647d430da54a06fe8087b89aebb49a8cabf733d6163230f0db3f20d4b6，exit0。独立checker82d25c63af35d1514090ccfd3ccfe3241334547dd8d9611065630c069c3b74c8首跑1.322301秒通过，结果SHA7c17c8d506cdbc515ef4df472a4af58d1e117301036aa3acfb09b935f90a76d7。

A v1 CPU0.109961秒失败0CUDA，原因inspect.getfile捕获torch装饰器包装器；原3a2a源/失败记录保留。v2仅inspect.unwrap与新报告/data路径，源8374c00f4421efd79b7b9da0dc7d352e4aeb976a8b88de359e8aaaafdb5477c4，CPU0.193170秒通过。唯一GPU0启动1791081858.606956/deadline1791082758.606956，run16.522474秒complete：2prefix+2wrapper/12SDPA，0full/quant/TE/VAE；M22400/22464，input/kwargs/output byteexact，峰40.2034GiB、1964222205B，SHA40fdde6e469055f2618019bf6967ff2f439755c08fc3b3467f485aecec04039c，exit0。独立checker f9477c239be962aa46ad77ce8f07eb7ad165d837be2c0f0b930721e21ce92a90首跑3.474768秒通过，4新PT和全44864行span重算，结果SHAb8e86887805c72ca4c313b487ed04084a587f6bec01a1fc9b1d918ff349ddd76。全部执行源/结果冻结，两GPU已释放，不重启。报告065/D095；仅入口/资源工程，不推完整配方/质量/创新。


## E071 — 2026-10-04 单候选实际评分/native导出一致性（complete）

2026-10-04。E071 CPUcheck2.607361秒通过后唯一GPU0启动1791083531.2891934/deadline1791084431.2891934；完整执行96.024907秒，SVD52.572899秒，峰44.795236GiB/产物6670219563B。真实calibrate/reset/ask/tell等8方法各一次，4attention+2reloadQKV/8SDPA/4native+pack/1SVD；0完整DiT/新视频。两BF16参照及候选/重载完整QKV、packet、LR证据byteexact，原模块完全恢复，supervisor/worker exit0。run SHA89a086bfbff78f74166f1d6ab9a4b232c6a5d892c78692c3510f276f98751d6d。

独立CPU首跑24.277667秒、143文件、0CUDA/0forward通过；完整输入/编码/scale/残余/因子与评分复算，库分数总差4.8854e−10、FP64差1.1967e−16。SHAe86b007f147f6fe957bcf96cb82cddf62b76f8acaefa69604d779ea905bd57d1。无重复GPU GEMM/完整SVD/库调用，边界保留。报告066。单候选身份不推39候选/100迭代/200层完整校准或画质，不计创新。下一进入实际配方方案，避免再做接口smoke。

有限社区检索补齐rootonchair calibrated-8x20 H3 Nunchaku-lite成品：固定模型/producer版本，保存6份源配置，0模型下载/执行；312SVDQ+50AWQ含refiner/分开QKV，与本地200目标不同。A4 metadata/runtime与发布权重完整同等性未核，列为外部产品基线候选，不能自动替代同模型强基线。E068人评仍待、不重问。研究目标active，本轮真实完成一致性判别为progress，无可投稿claim或外部阻塞。

D096补记：6份primary源的有界runtime核查已完成。固定Diffusers部署明确NVFP4 A4，不能以producer INT4 metadata反推部署INT4；必需Hub kernel本轮匿名访问401，未取得/安装/运行。312与200主要有QKV拆分计数因素，不证明不同底模；覆盖与rank预算仍需映射。E072实施草案完成，机械全配方成本尺度百GPU小时，未默认启动；保留一次同矩阵精确SVD执行成本比较备选，不当创新。当前无新GPU作业。


## D097 — 遵从用户纠偏，停止工程支线和无决策价值的代理

2026-10-04。用户再次要求控制工程投入、重点创新idea/观察/实验。E072a runner/checker均未创建，0CPUcheck/0GPU/0SVD；仅计划/监督草稿保留且明确停止，不得自动恢复。E072首块或全模型预算未启动。当前研究授权持续，目标active，非暂停或外部阻塞。

有界primary碰撞新增结论：RotateAttention §4.1直接覆盖可融合/可学2×2 RoPE旋转；HTG覆盖时变shift及zW bias/AdaLN吸收；Quantized Keys Steal Attention官方摘要已覆盖Jensen通用偏置修正；Q-VDiT/DeltaQuant/ResQ覆盖帧关系蒸馏和公共/差分量化区域。未把QVD diffusion timestep embedding误当frame-motion，未凭搜索未命中认领新颖。相关note与报告067保存；0新GPU/代码实验。

弱运动选择性衰减仅是问题，未成立。已查E014两状态具有完整BF16/plain/SVD velocity，但该域受扩散噪声混杂，一步clean估计也不能补motion标签；阳性与阴性均无法决定语义机制方向，因此不执行该代理。E041动作可辨性与相位、E067误差方向及待评E068均不证明该假设。下一由实际可辨缺陷与匹配对照建立可证伪预测，不再把完整基线重建当每个探索的前置工程。此轮有界近邻及数据适用性核查改变投入决定，属于progress；无新方法claim。


## D098 — 以实际媒体筛选身份混淆候选，暂不投入干预工程

2026-10-04。两个agent分别独立提案/反方核查，root实际查看E041 r0三attention臂page00–02（每臂0–47帧），另看E010两六帧联系图及E038汽车转弯两八帧三臂图。初始0–2分离构形已有低位多层细边，多次再分离仍有残影；未辨认闭合后新增且可跟踪的身份归属翻转。手始终局部投影重叠，缺干净无遮挡对照；早帧异常不否定双向attention错配。窄事件触发候选证据不足，决定暂不开发P重排/mask/kernel，不以此登记机制阴性或新方法。两个候选/反方note与报告068保存。0新实验代码、0模型/0GPU、未重评MJ，旧媒体与执行源不变。E072a停止，E068人评仍待且不重问。下一避免继续围绕不可识别的同一手部例扩文献/工程；目标active，无外部阻塞。


## D099 — 数值边界与匿名功能筛查

2026-10-04。完成pow2 global与低秩胞元响应的有界筛查：正常域解码齐次性不保证分区不变，t=2^-6、g从1到32为明确下溢反例；rank≤33仅为单token局部映射，全tensor为N×32+1且残差identity不受限。两个note已保存，不把格式性质升级新方法，不增加GPU或码值/SSE代理。随后只用E068公开匿名六媒体固定六帧渲染/查看，PyAV/PIL CPU1.9009秒、0模型/GPU；初次ffmpeg缺二进制0产物，未安装工具。三行文本及原料→产品主要结构两臂皆在，未辨认稳定A/B功能损失；0独立人评，不读private映射、不更新human review计数、不声称等质量。报告069与E068_anonymous_assistant_screen保存，新图仅DATA1。S013保持数值观察，未获新功能机制动机；E072a继续停止，目标active，无外部阻塞或运行任务。


## D100 / E073 — 固定八例行为发现开始

**E073 / D100 正在执行：固定八例主干W4A4行为发现。** 复用E038 `bf16`标签的既有native SVD+BF16 attention八视频，只补同prepare缓存/同20步/同原VAE的完整原始BF16八视频。非新方法；不复活已被GEAR/ResQ/VC覆盖的V低秩桥接。E073计划/manifest及独立入口已写，原E038源不改；两replica CPUcheck通过0.074606/0.077209秒、0CUDA。GPU0/1分担各4例80DiT，tmux h3_e073_r0/r1；原始共享deadline分别1791090066.3740733/1791090066.39087。launcher已确认denoise子进程686409/686410存活并实际前向，勿重复启动。每卡生成+decode1800秒/60GiB，合计8新视频、160DiT、8+8VAE、0TE/PTQ/MJ/RAFT。报告results/research/E073，data在DATA1/20261004/E073。

后续固定每2帧（0,2,...122,123）做匿名成对联系图，共8case×4页；观察者必须实际查看，静帧不称实时观看或人评。保留全部8例与unknown，不按输出换seed/扩样。需要实际完整BF16对照，现有E038不能充当；E068仍独立待人评，私有映射不读。研究目标active，本轮已有新生成在进行。



## D100 / E073 完成状态核查与 D101 中期汇报

2026-10-04。用户要求按创新含量排序的简洁中期报告，已保存报告070。两agent分别只读核查近期机制结果和早期方法尝试，root核对原报告及E073运行记录；本次无新GPU/代码实验。排名表示潜在研究价值，不认领已成立方法；局部-整体反转保留为最扎实数值观察，attention受限中心、跨步等当前路线停止，通信/native收益归工程。没有基于代理指标认领画质改善。

E073两个launcher均complete、denoise/decode各exit0；原共享deadline内r0/r1墙钟673.440517/686.635487秒，峰allocated40.772904GiB。共160次完整BF16 DiT，每次运行记录102SDPA/0scaled_mm/0disk；8video＋8audio VAE，8MP4实际存在57,718,138B。只读记录/存在性核查，不冒称独立全媒体验证。成对渲染与观察尚未执行，不重启生成；下一沿原固定8例观察协议继续，E068人评仍待且不读私有key。研究目标active，无新方法claim。


## D102 / E074 — 用户指定基线与配对视频距离

**D102：用户冻结现有可用 SVDQuant 配方为后续正式比较基线。** 后续表格统一称 SVDQuant baseline，不再因官方默认配置差异推进昂贵完整校准；实际实现/来源记录保留。该决定优先于此前所有“下一步补完整官方配方”的历史计划，E072a继续停止。

**E074 / [报告071](../reports/071_20261004_h3_baseline_distances.md) complete。** 用户要求现有基线对完整BF16的LPIPS/L1并追加L2。固定E010两例＋E038/E073四动作×两seed，共10配对；实际noise/embedding/schedule/settings一致、20MP4 SHA一致。原尺寸1024×576全部124帧，1240对帧，TorchMetrics1.9 LPIPS-Alex FP32，RGB[0,1]，无resize/crop/alignment。GPU0执行18.509秒，峰allocated0.163884GiB，exit0；每例首帧self-LPIPS<1e-7，stdlib逐帧归约及分组复算通过。10例等权均值LPIPS0.42008405、L1(MAE)0.110332594、L2(RMSE)0.175508504、未归一L2范数2599.745613、相对L2 0.553424166；8动作组分别0.443357309/0.116921104/0.175101531。详细逐例/逐帧/配对/CSV在results/research/E074。只测现有视频，不含音频，无新生成/训练/PTQ；距离不等于质量损失或人评。

E073成对匿名观察仍未执行；E074数值评价不替代该读出，不读取E068私有映射。研究目标仍active，本轮无新方法claim；当前评价进程已退出。



2026-10-04 D104/E075：近等误差量相位因果续跑，19DiT/2视频/1200秒。CPU检查通过；GPU0被他人占用，启动前退出0GPU，保留launcher.log；仅调度改GPU1，科学协议/runner保持不变。


D104/E075完成：[报告073](../reports/073_20261004_h3_phase_suffix_results.md)。19完整DiT/2新视频，一致性控制通过；近等局部误差改变最终视频，但固定抽帧无明确缺陷修复，局部BF16 oracle亦无明确修复。不扩此窗口网格；GPU1释放，原预算未延长，所有失败保留。


**D105 / [报告074](../reports/074_20261004_h3_action_and_decode.md)：E073匿名静帧观察及揭示已完成，E076 complete。** 8例×4页全部查看，先冻结观察后读E073映射；拍手r1/叠衣两seed有主干量化运动边界模糊线索，但不是人评或确定机制；大象后段亦喷水，撤回初段缺失判断。E068 key未读。E076两臂首时间窗30局部decode、8.415秒、4.91GiB、0DiT；raw tile内已见手部重复轮廓，停止主要归因空间blend，不扩tile网格。source/plan冻结，raw与JSON保留。

**E077探索已启动：** 固定block25分界的restart/carry二段2×2组合，16完整前向、600秒、0校准/训练/视频。候选是同rank/布局的静态版本组合；CLQ及联合离散码工作已有跨层叙事，尚无创新claim。四原teacher点只有两提示簇，非独立新测试；同一混合方案若四点均优于两个端点至少5%才进入独立视频验证，否则停止这个粗粒度候选，不扫分界。此决定有限重开可直接给出部署方案的组合测试，不重开泛SSE扫描。协议见06_experiments/E077_half_composition_plan.md。



**D106 / [报告075](../reports/075_20261004_h3_composition_candidate.md)：E077 complete，16完整native前向，138.298秒，峰37.61GiB，GPU0释放。** RR/CC八次raw精确复现历史；CR四点video SSE相对RR为−5.78%/+0.095%/−0.60%/−1.89%，RC为+19.16%/+17.58%/+1.92%/−0.29%。两混合均未通过固定四点至少5%改善门槛，停止二段静态组合候选，不扫分界/分层网格。RC音频SSE四点下降约18%–35%，仅模态收益不同的探索线索，不是音质/画质或模态竞争因果证据。有限向量交互大但不是有害性证明。CLQ及联合离散码工作已有直接近邻，无新颖方法claim。独立CPU复算完成，源/计划冻结；当前无GPU任务。E073/E076已完成状态以D105为准；E068 key仍未读。


**D107/E078：用户指定探索官方SageAttention3叠加正式SVDQuant基线。** 不把组合本身作为创新，也不以旧FlashInfer/QKV-only实验替代。固定两个完整teacher输入的BF16、仅SVD、仅Sage3、组合四角，共8DiT，检验相对实际误差向量和的净放大；原主基线200linear/rank32不变。官方源码pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5，隔离DATA1构建，不升级共享环境。接入先遇CUTLASS下载停滞、系统nvcc13与torch12.8冲突；改缓存CUTLASS4.5与NVIDIA官方独立12.8.1最小工具链，SHA验证。当前编译中、模型尚未执行；各失败日志保留。协议06_experiments/E078_sage3_interaction_plan.md，执行入口scripts/research/probe_h3_sage3_interaction.py。候选由用户明确提出，继续进行接入与可解释对照；普通接入不作创新。


**D107/E078 complete / [报告076](../reports/076_20261004_h3_sage3_interaction.md)。** 用户指定官方SageAttention3＋正式SVDQuant已完成四角对照。官方pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5原码构建，8完整DiT/69.627秒/43.26GiB，200次官方attention扩展＋1smoke，B0/S0四次raw逐byte复现。video组合/仅SVD为1.1620/1.3393；相对独立误差向量和的净放大比0.7519/0.7122，故未支持视频净放大；切换attention的输出变化能量比1.3784/1.3454，尚不知上游输入vs下游传播来源。audio净放大1.1277/1.0123，不稳定。无视频/音质结论，无新方法claim。CPU独立复算0.169秒/0CUDA、相对差2.22e−16；GPU0释放，无研究任务运行。官方接入成本另计：CUTLASS下载停止、nvcc13/cu128冲突、缺依赖头均记录；独立cu128工具链及缓存CUTLASS解决，无源码/共享环境升级。E078已执行源/计划冻结，勿重跑。下一若定位，只做能区分局部attention误差与后缀响应的固定输入干预，不默认扫描。


**D108：按用户指示，后续工作baseline更新为官方SageAttention3＋现有SVDQuant，原BF16与仅SVD作为参照保留。** 不改历史标签；具体合同与E078一致。E079固定四动作×两seed，复用E038准备输入及E073/E038已有参照，新增8组合视频/160DiT。CPU两replica检查已通过，GPU0/1各4例已启动，生成＋解码各共享1800秒。随后对三臂做明确标注覆盖范围的视觉观察、并排视频与LPIPS/MAE/RMSE；不把距离当质量百分比或静帧当人评。


**D108/E079 complete / [报告077](../reports/077_20261004_h3_sage3_video_quality.md)。** 工作基线已更新为官方SageAttention3＋现有SVDQuant，BF16/仅SVD保留参照。8组合视频/160DiT/8000 Sage3调用及解码完成；全部32页顺序抽帧观察：拍手两seed有额外手指模糊/重复轮廓（r0最明显），叠衣增量不确定，汽车/大象未见明确额外崩坏。非人评/实时完整播放，不归因有害交互。8例对BF16 LPIPS/MAE/RMSE为0.44881/0.11820/0.17986，仅SVD为0.44336/0.11692/0.17510；距离不代表质量百分比。16配对CPU归约复核通过，生成/解码/评价进程均已退出。视频在DATA1 research/20261004/E079/review，候选仅为局部细结构损伤，未形成新方法；停止追加本轮GPU。


**D111/E080 started：** 用户授权C007双侧PV差分表示实验。两clap seed原轨迹重放40DiT，固定s14/b0,b24捕获，要求最终latent exact；局部同bit/scale对照与预设差分误差gate详E080_pv_contrast_plan.md。初版CPU检查因继承manifest输出路径guard失败（0GPU）已保留；v2两CPU检查通过，GPU0/1各900秒capture启动。执行前计划已冻结，不改门槛，不做C008或新kernel。


**D111/E080 complete / [报告080](../reports/080_20261004_pv_contrast_results.md)：** C007当前双侧PV差分表示停止。两clap seed重放40完整DiT，最终video/audio latent逐byte复现E079；s14/b0,b24全部56heads、36对query、完整keys，QKV官方pack零差异。native QK下b24差分SSE降14.29%/13.30%，但总SSE增6.23%/5.57%；V中心化32差分降30.31%/31.04%，同预算shuffle已降12.96%/9.73%，pair只额外改善1.54%/3.96%<5%。b0总SSE增48.5%–56.7%，exact QK结论一致，未过预设门槛。contrast_gain未更接近1，不能认领细节恢复。独立CPU824项复核通过，归一化差≤4.44e−16；模拟/native gap能量比≤0.0113%。捕获128.22/126.60秒、峰37.60GiB，局部成功14.55秒/13.20GiB；CPU入口失败及FP32归约验证失败均保留，新版独立FP64验证通过。0新视频/VAE/训练/kernel，GPU0/1释放，不扫参数或启动C008；无新颖方法或画质收益claim。


**D113 / E081 complete，E082准备（2026-10-05）：** 用户授权跟进V中心化。两因素消融复现E080：b24/g32仅V表示SSE降22.32%/25.31%，仅概率质量补偿降23.02%/21.93%，合用降40.24%/42.07%；两者均有贡献，g128/QK条件/full参考一致。b0补偿可解释84%–85%组合收益，不能推广层规律。独立CPU856项验证通过，最大归一差4.44e−16；GPU13.414秒。下一E082以g128最小私有native接入进行原生验收，通过才两clap完整20步视频；已有方法验证非创新，官方基线不改。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。


**D114 / E082 complete（2026-10-05）：** V中心化g128已完成原生验收和两clap完整视频，40DiT/2video+2audio VAE。zeroμ逐byte复现，center模拟gap≤0.68%，原生b24全attention SSE降14%–15%；两视频的32固定抽帧/clip均未见明确稳定手部修复，r1另独立审阅。对BF16 LPIPS原→center r0 .27126→.26532，r1 .36829→.42654；MAE/RMSE均增，r1构图姿态改变，不把距离当质量。保留已知中心化局部阳性对照，不替换主基线，不扩kernel/层步网格，不否定完整VC。全部检查与评价结束、GPU0/1释放；生成137.68/136.73秒/37.60GiB，解码11.92/11.47秒。记录两次实施失败及修正，不记作方法阴性。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。


**D115 / E083 running（2026-10-05）：** 按用户建议，隔离attention：BF16主干＋官方Sage3 vs BF16主干＋同一已验收center128，复用完整BF16参照。两clap seed共4新视频/80DiT、GPU0/1各共享1800秒；四CPU输入/源检查已complete0CUDA，开始原20步完整采样。检查200原BF16 Linear、每步0scaled_mm/52SDPA/50FP4；不新增kernel或改分组。40%–42%仅g32固定量化QK下局部PV SSE，不是完整attention或视频误差；当前均值分解是VC V-Smooth已有组成，不认领创新。见[E083计划](../06_experiments/E083_bf16_attention_comparison_plan.md)。


**D115 / E083 complete（2026-10-05）：** 用户建议的BF16主干隔离对照完成：两clap seed×官方Sage3/center128，4新视频/80DiT。200原BF16 Linear及每步0scaled_mm/52SDPA/50FP4、输入/源/时间表/VAE合同全部通过。对完整BF16的LPIPS原→center为.16658→.17749、.34304→.36946；MAE两例增大，RMSE一升一降。固定32抽帧/clip未见稳定细节收益，seed1另独立审阅且有构图变化，不把距离当质量。不能仅用SVD主干混杂解释前次未见收益，也不否定具体精度交互/其他场景/完整VC。40%仅历史g32固定量化QK的局部PV SSE；V均值分解为已有方法，不新增claim，不扩参数网格，主baseline不变。六配对独立归约/六媒体SHA通过；全部进程完成，GPU0/1释放。详[报告082](../reports/082_20261005_bf16_attention_comparison.md)。


**D116 / E084 complete（2026-10-05）：** 补齐E022原16测试轨迹的逐步同状态/自由输出/latent NMSE。288 DiT、192步哈希与48终态精确复现；CPU4192项独立校验通过。测试同状态第1–4步pooled变化−22.24%/+40.45%/+33.37%/−2.03%，开发32%降幅未稳定迁移；自由输出末步−8.33%，正交叉项增加使终态latent pooled仅−0.96%，等权+17.17%。GPU0 506.78秒/4.15GiB，无训练/视频/VAE；补测完成即停止，不恢复新idea研究。[报告083](../reports/083_20261005_wan_qad_timestep_replay.md)。
