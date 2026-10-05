# Current State

2026-10-05。用户已暂停新 idea 研究，10 项提案仍待逐项审核。**本轮仅获授权补测旧 E022 Wan QAD 的逐步测试 NMSE 和时间步差异（E084），不恢复其他实验。** 后续研究仍以 MiniMax-H3 为主，普通基线工程不作为创新。当前无可投稿核心贡献、无独立人评验证的等质量优化。

## 最新决策与实验

**D116 / E084 complete（2026-10-05）：** 用户要求补测旧 E022 测试误差，288 DiT/67200 native GEMM/17280 SDPA完成；192步哈希、48最终latent精确重现，GPU0耗时506.78秒/峰4.15GiB，无训练/新视频/VAE。同BF16状态测试 pooled NMSE 0.12598→0.13268(+5.32%)，等权0.14691→0.14154(−3.65%)；第1–4步pooled变化−22.24%/+40.45%/+33.37%/−2.03%，开发32%下降未稳定迁移。自由轨迹输出pooled改善10.45%，末次预测改善8.33%，但采样更新正交叉项增加使最终latent pooled仅改善0.96%，等权反增17.17%；RGB约改善9%。不能写成每步预测都更准或纯误差累积问题。CPU4192项独立验证最大归一差2.55e−15。见[报告083](../reports/083_20261005_wan_qad_timestep_replay.md)。补测到此结束，新idea研究继续暂停。

**D115 / E083 complete（2026-10-05）：** 用户建议的BF16主干隔离对照完成：两clap seed×官方Sage3/center128，4新视频/80DiT。200原BF16 Linear及每步0scaled_mm/52SDPA/50FP4、输入/源/时间表/VAE合同全部通过。对完整BF16的LPIPS原→center为.16658→.17749、.34304→.36946；MAE两例增大，RMSE一升一降。固定32抽帧/clip未见稳定细节收益，seed1另独立审阅且有构图变化，不把距离当质量。不能仅用SVD主干混杂解释前次未见收益，也不否定具体精度交互/其他场景/完整VC。40%仅历史g32固定量化QK的局部PV SSE；V均值分解为已有方法，不新增claim，不扩参数网格，主baseline不变。六配对独立归约/六媒体SHA通过；全部进程完成，GPU0/1释放。详[报告082](../reports/082_20261005_bf16_attention_comparison.md)。

**D114 / E082 complete（2026-10-05）：** V中心化g128已完成原生验收和两clap完整视频，40DiT/2video+2audio VAE。zeroμ逐byte复现，center模拟gap≤0.68%，原生b24全attention SSE降14%–15%；两视频的32固定抽帧/clip均未见明确稳定手部修复，r1另独立审阅。对BF16 LPIPS原→center r0 .27126→.26532，r1 .36829→.42654；MAE/RMSE均增，r1构图姿态改变，不把距离当质量。保留已知中心化局部阳性对照，不替换主基线，不扩kernel/层步网格，不否定完整VC。全部检查与评价结束、GPU0/1释放；生成137.68/136.73秒/37.60GiB，解码11.92/11.47秒。记录两次实施失败及修正，不记作方法阴性。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。

**D113 / E081 complete，E082原生验收历史状态（现已完成，见D114）：** 用户授权跟进V中心化。两因素消融复现E080：b24/g32仅V表示SSE降22.32%/25.31%，仅概率质量补偿降23.02%/21.93%，合用降40.24%/42.07%；两者均有贡献，g128/QK条件/full参考一致。b0补偿可解释84%–85%组合收益，不能推广层规律。独立CPU856项验证通过，最大归一差4.44e−16；GPU13.414秒。E082 g128原生验收四capture通过（zeroμ逐byte，center gap≤0.68%），随后已完成两clap完整20步视频，见D114；已有方法验证非创新，官方基线不改。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。

**D112（2026-10-05）：停C007不等于停V中心化验证。** 用户指出强对照明显有效，前次沟通混淆创新性与实际效果筛选，现纠正。E080 center32在b24局部PV SSE降40%–42%，含QK误差的同输入完整attention SSE仍降15%–16%；值得有限端到端验证，按已有方法/增强基线对待。当前双侧表示仍停止；V中心化后续验证待执行，0新GPU/视频，暂无质量收益。center精确概率质量补偿也消除P舍入质量误差乘V均值项，不能将全部收益归因V范围。详报告080末补充。

**D111/E080 complete / [报告080](../reports/080_20261004_pv_contrast_results.md)：** C007当前双侧PV差分表示停止。两clap seed重放40完整DiT，最终video/audio latent逐byte复现E079；s14/b0,b24全部56heads、36对query、完整keys，QKV官方pack零差异。native QK下b24差分SSE降14.29%/13.30%，但总SSE增6.23%/5.57%；V中心化32差分降30.31%/31.04%，同预算shuffle已降12.96%/9.73%，pair只额外改善1.54%/3.96%<5%。b0总SSE增48.5%–56.7%，exact QK结论一致，未过预设门槛。contrast_gain未更接近1，不能认领细节恢复。独立CPU824项复核通过，归一化差≤4.44e−16；模拟/native gap能量比≤0.0113%。捕获128.22/126.60秒、峰37.60GiB，局部成功14.55秒/13.20GiB；CPU入口失败及FP32归约验证失败均保留，新版独立FP64验证通过。0新视频/VAE/训练/kernel，GPU0/1释放，不扫参数或启动C008；无新颖方法或画质收益claim。

**D110 / [报告079](../reports/079_20261004_vc_didb_compatibility.md)：兼容性与查新更正。** VC的V-Smooth可融入Sage3 FP4，需要kernel均值补偿；ExpCast-FP8不能直接套NVFP4。未确认官方公开实现，不能许诺直接安装。DIDB是需训练的二值ViT结构，不能插入现成H3；它也直接保护attention差分，C007动机碰撞比上轮描述更强，仅同预算PV双侧表示残余待验证。无代码接入/新GPU，现baseline不变。

**D109 / [报告078](../reports/078_20261004_fine_structure_hypotheses.md)：精细结构机制与方法候选。** 拍手是当前最明确可见缺陷，非总体最大问题/softmax变平/有害协同的证明。只保留P011下C007（量化前PV双侧差分表示）和C008（沿运动输运的量化误差结构），均proposed、untested。有限查新剔除V中心化/保排名/重要块高精度/泛时间loss/导数匹配作为独立创新；VC-Attention、DIDB-ViT、Q-VDiT、2ndMatch为关键近邻。先固定输入机制干预，后决定工程，尚未启动新GPU。主baseline仍官方Sage3＋现有SVD；本轮0GPU、0新效果。

**D108/E079 complete / [报告077](../reports/077_20261004_h3_sage3_video_quality.md)。** 工作基线已更新为官方SageAttention3＋现有SVDQuant，BF16/仅SVD保留参照。8组合视频/160DiT/8000 Sage3调用及解码完成；全部32页顺序抽帧观察：拍手两seed有额外手指模糊/重复轮廓（r0最明显），叠衣增量不确定，汽车/大象未见明确额外崩坏。非人评/实时完整播放，不归因有害交互。8例对BF16 LPIPS/MAE/RMSE为0.44881/0.11820/0.17986，仅SVD为0.44336/0.11692/0.17510；距离不代表质量百分比。16配对CPU归约复核通过，生成/解码/评价进程均已退出。视频在DATA1 research/20261004/E079/review，候选仅为局部细结构损伤，未形成新方法；停止追加本轮GPU。

**D107/E078 complete / [报告076](../reports/076_20261004_h3_sage3_interaction.md)。** 用户指定官方SageAttention3＋正式SVDQuant已完成四角对照。官方pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5原码构建，8完整DiT/69.627秒/43.26GiB，200次官方attention扩展＋1smoke，B0/S0四次raw逐byte复现。video组合/仅SVD为1.1620/1.3393；相对独立误差向量和的净放大比0.7519/0.7122，故未支持视频净放大；切换attention的输出变化能量比1.3784/1.3454，尚不知上游输入vs下游传播来源。audio净放大1.1277/1.0123，不稳定。无视频/音质结论，无新方法claim。CPU独立复算0.169秒/0CUDA、相对差2.22e−16；GPU0释放，无研究任务运行。官方接入成本另计：CUTLASS下载停止、nvcc13/cu128冲突、缺依赖头均记录；独立cu128工具链及缓存CUTLASS解决，无源码/共享环境升级。E078已执行源/计划冻结，勿重跑。下一若定位，只做能区分局部attention误差与后缀响应的固定输入干预，不默认扫描。

**D107/E078：用户指定探索官方SageAttention3叠加正式SVDQuant基线。** 不把组合本身作为创新，也不以旧FlashInfer/QKV-only实验替代。固定两个完整teacher输入的BF16、仅SVD、仅Sage3、组合四角，共8DiT，检验相对实际误差向量和的净放大；原主基线200linear/rank32不变。官方源码pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5，隔离DATA1构建，不升级共享环境。接入先遇CUTLASS下载停滞、系统nvcc13与torch12.8冲突；改缓存CUTLASS4.5与NVIDIA官方独立12.8.1最小工具链，SHA验证。当前编译中、模型尚未执行；各失败日志保留。协议06_experiments/E078_sage3_interaction_plan.md，执行入口scripts/research/probe_h3_sage3_interaction.py。候选由用户明确提出，继续进行接入与可解释对照；普通接入不作创新。

**D106 / [报告075](../reports/075_20261004_h3_composition_candidate.md)：E077 complete，16完整native前向，138.298秒，峰37.61GiB，GPU0释放。** RR/CC八次raw精确复现历史；CR四点video SSE相对RR为−5.78%/+0.095%/−0.60%/−1.89%，RC为+19.16%/+17.58%/+1.92%/−0.29%。两混合均未通过固定四点至少5%改善门槛，停止二段静态组合候选，不扫分界/分层网格。RC音频SSE四点下降约18%–35%，仅模态收益不同的探索线索，不是音质/画质或模态竞争因果证据。有限向量交互大但不是有害性证明。CLQ及联合离散码工作已有直接近邻，无新颖方法claim。独立CPU复算完成，源/计划冻结；当前无GPU任务。E073/E076已完成状态以D105为准；E068 key仍未读。

**D105 / [报告074](../reports/074_20261004_h3_action_and_decode.md)：E073匿名静帧观察及揭示已完成，E076 complete。** 8例×4页全部查看，先冻结观察后读E073映射；拍手r1/叠衣两seed有主干量化运动边界模糊线索，但不是人评或确定机制；大象后段亦喷水，撤回初段缺失判断。E068 key未读。E076两臂首时间窗30局部decode、8.415秒、4.91GiB、0DiT；raw tile内已见手部重复轮廓，停止主要归因空间blend，不扩tile网格。source/plan冻结，raw与JSON保留。

**E077探索已启动：** 固定block25分界的restart/carry二段2×2组合，16完整前向、600秒、0校准/训练/视频。候选是同rank/布局的静态版本组合；CLQ及联合离散码工作已有跨层叙事，尚无创新claim。四原teacher点只有两提示簇，非独立新测试；同一混合方案若四点均优于两个端点至少5%才进入独立视频验证，否则停止这个粗粒度候选，不扫分界。此决定有限重开可直接给出部署方案的组合测试，不重开泛SSE扫描。协议见06_experiments/E077_half_composition_plan.md。

**D104 / [报告073](../reports/073_20261004_h3_phase_suffix_results.md)：E075 complete，GPU1已释放。** 固定p36/s14/block24，近等误差相位干预与局部BF16 oracle，19完整DiT/2新视频完成；phase0逐步精确重放、phase128单步精确控制通过。局部SSE仅+0.0616%，相位差能量为原误差62.43%；phase64最终video latent relative L2 7.012%，相对原视频LPIPS .04395，oracle .04556。固定六帧看到细节改变，没有明确缺陷修复；不作视频等质量/方法收益结论。停止扩该位置的相位/层/步网格，下一优先既有动作配对可辨缺陷。主基线冻结，当前无active已验证方法。GPU0占用拒绝、两次未完成forward的接口/PATH失败、一次CPU隐藏设备检查失败均保留；成功v2运行116.765秒、解码21.340秒，原1200秒截止未延长。详见results/research/E075/v2，不重跑。

**D103 / [报告072](../reports/072_20261004_observations_and_candidate.md)：回应有趣观察与idea，0新GPU。** 保留三个不同层次的事实：局部Pareto改进却整模SSE反转；FP4attention分组相位近等误差能量下改变误差向量；已看拍手中可辨性下降不等于统一减速。后两项来自attention精度/布局干预，不冒充主干W4A4失效归因。建议讨论近等能量真实干预与最终可辨缺陷的因果连接；仅问题/待验证想法，尚无执行计划或active方法claim，不能将旧停线直接复活。CLQ/PARO/跨步补偿及noise shaping已有覆盖，泛loss/重排/随机化不认领新颖；若缺陷不可辨或只有构图变化则停止。E073匿名行为读出尚待，E074数值距离不代替；用户指定基线继续冻结，不恢复官方校准工程。

**D102：用户冻结现有可用 SVDQuant 配方为后续正式比较基线。** 后续表格统一称 SVDQuant baseline，不再因官方默认配置差异推进昂贵完整校准；实际实现/来源记录保留。该决定优先于此前所有“下一步补完整官方配方”的历史计划，E072a继续停止。

**E074 / [报告071](../reports/071_20261004_h3_baseline_distances.md) complete。** 用户要求现有基线对完整BF16的LPIPS/L1并追加L2。固定E010两例＋E038/E073四动作×两seed，共10配对；实际noise/embedding/schedule/settings一致、20MP4 SHA一致。原尺寸1024×576全部124帧，1240对帧，TorchMetrics1.9 LPIPS-Alex FP32，RGB[0,1]，无resize/crop/alignment。GPU0执行18.509秒，峰allocated0.163884GiB，exit0；每例首帧self-LPIPS<1e-7，stdlib逐帧归约及分组复算通过。10例等权均值LPIPS0.42008405、L1(MAE)0.110332594、L2(RMSE)0.175508504、未归一L2范数2599.745613、相对L2 0.553424166；8动作组分别0.443357309/0.116921104/0.175101531。详细逐例/逐帧/配对/CSV在results/research/E074。只测现有视频，不含音频，无新生成/训练/PTQ；距离不等于质量损失或人评。

E073成对匿名观察仍未执行；E074数值评价不替代该读出，不读取E068私有映射。研究目标仍active，本轮无新方法claim；当前评价进程已退出。

**D101 / [中期报告070](../reports/070_20261004_midterm_report.md)：按用户要求，以动机→实验→结果概括已有尝试，并按创新潜力排序。** 最扎实观察是局部误差全改善但整模视频预测SSE变差；尚未建立新颖机制或画质损伤。受限attention中心有局部信号但简单对照解释大部分收益，24视频无稳定优势，路线已停。跨步、结构与组合筛查有界停止；通信与native部署只计工程。未恢复工程支线、未启动新GPU；无可投稿方法或独立人评等质量结论。

**E073 / D100 生成与解码均已 complete，成对观察尚未执行。** 本轮只读核查两launcher及denoise/decode报告：两个phase各自exit0；r0/r1共用原始生成＋解码deadline，墙钟分别673.440517/686.635487秒，峰allocated均40.772904GiB。各4case×20次DiT，共160次；全部记录102 SDPA、0scaled_mm、0disk_loads；各4video＋4audio VAE，合计8＋8。已确认8个MP4实际存在，共57,718,138 bytes。此为运行记录与文件检查，未完成独立媒体解码/全数据复算或画质评价。

复用E038 `bf16`标签的既有native SVD＋BF16 attention八视频，仅补同prepare缓存/20步/原VAE的完整原始BF16八视频。0新TE/PTQ/MJ/RAFT，不重复启动h3_e073_r0/r1或旧PID。报告results/research/E073，媒体在DATA1/20261004/E073/decode_full_bf16_r0与r1。下一按原计划固定每2帧做匿名成对观察，保留全部8例与unknown，不按结果换seed/扩样；渲染工具尚未执行，不把静帧观察当实时播放或人评。E068人评继续独立待反馈，私有映射不读。研究目标仍active，当前E073生成任务已结束。

**D099 / [报告069](../reports/069_20261004_h3_numeric_boundaries_and_anonymous_screen.md)：完成两项数值边界筛查及E068匿名媒体观察，0新模型/GPU/工程框架。** pow2 global只在正常安全域有解码不变性；t=2^-6、g=1→32即给出下溢反例，不能声称无条件分区不变。固定胞元rank≤33只适用单token映射，全矩阵为N×32+1，残差block不受此界。两者暂不立方法或追加统计代理。

E068仅访问公开blind_review_v2，未读private key；PyAV/PIL CPU1.9009秒渲染六条×六帧并全部查看，保存在DATA1/E068_screen。A/B均保留三行文字及原料→产品结构，未辨认稳定功能损失；这不是人评、全片评估或等质量证明。human review状态不变，S013仅数值观察，不启动新SSE扫描。下一需明确、可判断的最终任务缺陷/问题，不为同一不可识别例继续扩工程或文献；目标active，无运行任务。

**D098 / [报告068](../reports/068_20261004_h3_readability_hypothesis_screen.md)：低工程投入的实际媒体筛查已完成，0新代码/模型/GPU。** 两个 agent 将“动作尚存、边界难读”收敛为相似对象接近/遮挡后的身份错配候选，root实际查看r0三attention臂page00–02（每臂0–47）。0–2初始分离构形已见低位重影，多次再分离仍持续；未辨认闭合后新增且可追踪的身份归属翻转。两手仍部分投影重叠，不能冒充无遮挡对照或以未见交换否定全部双向attention对应。窄事件触发版本证据不足，暂不开发P重排/mask/kernel；候选note不是active已验证claim。

另看E010两六帧配对及E038汽车转弯两八帧三臂图，差异不能直接推画质/速度；无独立人评或新的因果归因。Decoder时间压缩解释已有前期note，不再次包装。下一优先能区分机制的观察/小实验，不继续为同一不可识别手部例扩工具或文献。E072a仍停止，E068人评仍待且不重问；研究目标active，无新运行任务。

**用户最新纠偏：控制工程投入，重点思考有创新性的idea、观察及实验检验。** E072a SVD提速支线已停止，0CPUcheck/0GPU/0新SVD；协议与监督草稿保留但不得据此启动。完整强基线扩展亦非当前优先项，不能以补基线为由持续派生工程小检查。研究目标仍active，非暂停。

D097 / [报告067](../reports/067_20261004_research_focus_reset.md)：有界机制/近邻筛查已完成，无新增GPU。可融合RoPE旋转、AdaLN确定性shift移出、Jensen偏置通用修正均因直接已有方法覆盖不立项；详见两个prior-work note及fresh机制note。帧轴弱变化问题暂无本地阳性，Q-VDiT/DeltaQuant/ResQ已有直接区域覆盖；不把QVD timestep embedding误当frame-motion。

Root只读确认E014 p30/s05、p36/s14保存完整velocity及原sigma/latent；反方判定含噪velocity或一步clean预测都不能识别真实motion，阳性/阴性均不可靠改变该机制决策，因此0新张量统计，不执行代理。新增有界来源与数据适用性检查改变研究投入，当前turn progress；无active新claim，无外部阻塞。

下一从实际、可辨识的生成缺陷与匹配控制建立1–2个可证伪假设；必要工程只为具体研究实验服务，不把完整基线重建当所有探索前置条件。E068人评仍待且不重问、不泄露映射；不能围绕未证画质的SSE反转继续扫描。研究目标active；本轮所有agent任务完成，无新增GPU/CPU实验在跑。


D096 / [报告066](../reports/066_20261004_h3_native_candidate_identity.md)：**E071 GPU与独立CPU核验均complete，worker/supervisor exit0、GPU0释放，不重跑。** 真实单Manual(.5,.5)候选，两E070 wholecase，4attention+2reloadedQKV/8SDPA/4native+pack/1exactSVD，0prefix/fullDiT/TE/VAE/video。96.024907秒（SVD52.572899秒），峰44.7952GiB，6670219563B。两BF16参照复现、候选与重载完整QKV/packet/LR证据byteexact，原对象/权重/存储/hooks恢复。run SHA89a086bfbff78f74166f1d6ab9a4b232c6a5d892c78692c3510f276f98751d6d；runner1ea7d66138db93570bb062243fda375259afb6afa62fe9f098bb2e56465f8b74，已执行源/计划/产物冻结。

独立checker f4669079448ed7f9023624ba3e85a66758653f7f2f3dc85a6417f0534ad388bf首跑24.277667秒/0CUDA/143文件通过；result SHAe86b007f147f6fe957bcf96cb82cddf62b76f8acaefa69604d779ea905bd57d1。完整xs、packet编码、平滑/残余、因子cast与抽样奇异方程核验；实际BF16差→FP32平方库分数与独立同定义CPU总分差4.8854e−10，另FP64读出差1.1967e−16。未重演GEMM/完整SVD/库调用。仅接线身份成立，不推搜索/完整配方/画质/创新。

已被D097用户纠偏搁置的工程路线见[E072实施草案](../06_experiments/E072_native_baseline_design_draft.md)：已完成设计，尚未实施，12GPU小时首块方案与全200层均未选定/启动。机械使用E071单QKV分解时间作尺度，7800 smooth约114 GPU小时、连20000 LR上限约406小时（不是精确预测，未含case评分/I/O），不能默认投入。有界备选是同一已保存Ws的一次reduced FP64/gesvd，比较既有top32与部署因子/packet，判断能否节约成本；不是接口smoke/创新，也尚未执行。用户持续研究授权有效，无新许可门槛。

[社区runtime核查](../04_prior_work/h3_community_runtime_contract_20261004.md)已收束：固定Diffusers8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd，6份primary源明确部署调用NVFP4 A4，旧sidecar INT4不能推部署INT4。必需rootonchair/nunchaku-lite-kernels version2本轮匿名API/main README返回401，未取得kernel/运行，不能宣布永久不可用或已复现。312/200主要含fused QKV拆分，不证明不同底模；额外refiner/AdaLN与分拆rank32预算需映射。已查本地native/recovered无相关包、legacy Diffusers0.33.1、无标准HF凭据/查找位置缓存，未读secret。0权重下载/安装/执行；外部产品基线候选保留，当前不能据此取消同配置强基线。原intake记录保留，后续以runtime note为准。

E068仍待人评，不重问；强基线工作独立继续，工程不当创新，无新方法claim，研究目标active。当前本轮已完成E071与有限外部运行核查，所有本轮GPU/CPU任务结束，无外部阻塞，不重做已完成验证。


D095 / [报告065](../reports/065_20261004_h3_baseline_entry.md)：**E070两GPU pilot及两独立CPU复核均complete，两个supervisor exit0，GPU0/1已释放；勿重启。** A v1CPU因装饰函数源定位失败0CUDA，v2仅inspect.unwrap修复后CPU0.193170秒通过；真实变长attention入口16.522474秒，2prefix+2wrapper/12SDPA/0full/0quant，byteexact，峰40.2034GiB，data1964222205B，run SHA40fdde6e469055f2618019bf6967ff2f439755c08fc3b3467f485aecec04039c。A独立CPU3.474768秒/0CUDA，4新PT全输入kwargs输出逐byte、2case、44864行span独立重算通过，SHAb8e86887805c72ca4c313b487ed04084a587f6bec01a1fc9b1d918ff349ddd76。B严格一次原始fc1完整FP64 SVD59.398311秒，总61.284249秒、峰10.4537GiB、data15300421B；结果SHA55b4b7647d430da54a06fe8087b89aebb49a8cabf733d6163230f0db3f20d4b6。B独立NumPy1.322301秒/0CUDA，选定权重freshhash、抽样top32方程/Gram/BF16舍入通过，SHA7c17c8d506cdbc515ef4df472a4af58d1e117301036aa3acfb09b935f90a76d7。此为强基线入口工程，尚未完整calibrate/search、部署一致性或质量证据；不将单原W资源推到全PTQ。已执行源/计划/产物冻结。

E070所列单候选下一步已由顶部E071完成；以D096为准，不重复其检查。

D094 / [报告064](../reports/064_20261004_h3_strong_baseline_contract.md)：**E069源码/CPU配方审计complete0.032997秒，0torch/0GPU/0forward。** 固定upstream commit69f3473f5e1c1504bae35cc50c7858ef900a9b17，12份源243155B逐SHA保存。直接执行AST提取候选方法：上游g20默认39、同规则g10为19，当前H3g10仅9（缺恒等及9个activation-only候选）。当前H3smooth评分无LR，上游allow_low_rank；upstream完整FP64 SVD、本地FP32 randomized q=r+8/niter2；当前H3已有全block评分/块间student传播，E065b才是8teacher逐linear。上游模块选择/缓存和NVFP4尺度合同须单独对齐，不把上述差异解释成效果优劣或历史生产身份。audit SHAaa993946fbf01fc428067ba9a0d6cb59befec0cdbf7b5bbb4fd79de10480faf0，源/计划/结果冻结。没有新方法claim，E068人评仍待且不重问。

E069独立复核已complete0.027004秒，31文件fresh SHA（12上游源），纯Fraction另构造集合与顺序通过，未导入/重跑原方法、0torch/GPU/forward。independent_review SHA6a98b15801658cb7ca2e5b56072063756b7089005a8892b1c779247d946b167d。候选指数tuple不保证实际scale张量互异；未验证完整配置loader/H3 adapter/质量。源码审计阶段已结束，勿重复运行/扩文献审计；下一是下面的实际接入与资源合同。

下一独立行动是强基线接入正确性＋资源pilot，不依赖E068先到人评：H3变长packed序列不能直接交给TensorCache按M切批；unsqueeze也不解决不同M形状assert。考虑独立case-index eval wrapper取完整x/rope/cu_seqlens，x_acts只统计span，需实际零干预byte回放后才接受。没有GPU实验已启动；不得直接将E065b叫完整官方强基线，也不修改现有frozen数据/用户dirty文件。旧64raw校准输入可复用，H3为主。普通基线工程非创新；上一turn E068完成是progress，本turn新增可复现配方差异改变强基线选择，也属progress，目标active无外部阻塞。

D093 / [报告063](../reports/063_20261004_h3_blind_video_ready.md)：**E068两臂四阶段 complete、supervisor均exit0，GPU0/1均已释放；不要重启。** 原共享900s内总用时349.417秒，82full/16400native+pack/8364SDPA/160updates，4video+4audio VAE，峰37.6044GiB/卡，数据1096603006B。独立CPUcomplete4.256021秒/0CUDA/0forward，160新step/80历史schedule/496解码帧、两teacher replay、起点/衔接/终点/PCM均通过；未重演调度算术或模型。independent SHAe3ae5d66c81bc758ffec5992d9b7d681220ed4238f7fe8e16567086c9d558672。四新视频＋两历史BF16已盲化，public路径results/research/E068/blind_review；**未有人评，不泄露private映射，不造偏好或质量结论**。等待反馈期间不继续仅凭SSE反转做机制扫描；无新方法/可投稿贡献，工程不算创新。已执行runner/checker/builder/plan冻结。目标active，本轮完成实际视频与核验，属于progress。

展示补记：blind_review_v2修异步播放取消/成对暂停，六媒体与manifest byteexact，旧产物不改；refiner710b74adc702db28e3bd6d868d4240f953ca8bac3015904a474769408ab7baab已执行冻结，Node语法检查通过、未实播验证。远程file URL不能在本地browser打开，提供26.19MB离线包blind_review_bundle.zip（SHA5d8ba9bdf11e037370a443bdc74dcdc5a5eeb3328973bcad9d91fe241b114f0d），没有部署/发布外部站点。已用异步输入工具请求case01/02各A/B/难分及理由，反馈尚待；不重复询问或将等待视作回答。

D092 / [报告062](../reports/062_20261004_h3_error_direction_and_video_check.md)：E067 CPU complete551.028857秒、0CUDA/0forward、812PT重验、RSS2.12GiB、exit0，无运行中分析任务。四状态各200层保存video样本的两臂误差cos中位.73884/.73897/.73929/.73948；误差差值中正交比例中位约.8566，800/800均>.5（范围.800879–.908254），去均值后亦800/800。形状造成的δ差异能量/候选δ最多.000324708，范数约1.80%。完整video误差cos.65038/.62596/.60914/.57987。仅排除保存样本的简单径向缩幅近似，不证明方向致害、NVFP4特殊性或视频质量。raw/centered SSE绑定原独立检查，分解恒等式通过；新增方向统计未由第二程序全量复算。analysis SHAf414e380aef5a30855082458e07c11e28314ddd79afe15b895e3b9e05236f776，已执行源/计划冻结。

下一行动改为先核验任务相关性：E065b的teacher DiT SSE不是画质，继续围绕它修复可能追错代理。E068已完成，当前状态以顶部D093为准。CPUcheck14.413961秒/0CUDA，原shared start1791078059.7994795/deadline1791078959.7994795；复用E010两原prompt/seed/embedding/noise与BF16参照、原20步自由轨迹/decoder。每臂1次p30s5本臂重放及2条视频，共82full。不得通过MJ/模型看帧伪造人评，无意见时仅交付材料。E067任何方向结果不自动开GPU机制扫描/新loss。S013/P010暂定问题无方法claim；SVDQuant/ARHQ/QDrop近邻及跨精度表示审查均未立项。目标active，本轮已有真实诊断与任务选择进展。

D091 / [报告061](../reports/061_20261004_h3_local_transfer.md)：**E066 complete，独立复核与描述汇总均完成，无运行中GPU/CPU任务，勿重复运行。** 四个原teacher状态各200层的video局部SSE全部carry优于restart；真实M512、full-M同512样本及full-M全部有效行三种排序一致。全video层中位carry/restart为.969639403/.969812696/.970606081/.970469527，800格范围.944804576–.993002120；joint/text全部改善，audio798/800改善，两例外均block3.fc2（+0.3925%/+0.3280%）。M512到full-M同样本的video比率相对变化最大3.71071e−5。与E065b整模video四点仍全部更差结合，排除这些输入上的简单局部迁移失败与形状/采样排序假象；不能识别方向、student输入、模态或组合的因果贡献。

运行350.657461秒，4full/3200local（1600full＋1600subset）/800pack/408SDPA/0disk，峰43.7253GiB、数据71.8775GB；四BF16 actual/raw/velocity复现，exit_code0，GPU0释放。独立CPU247.513912秒/0CUDA/0forward，812PT＋400导出，409600样本行/3200输出张量，最大相对差5.8134e−16；只独立复算保存512行通道及全部per-row归约，不声称未存full-M通道/native重演。stdlib汇总0.058611秒，无torch。evaluate SHAb2846d22dd2d06b846b54b9b3efcdeacb342f5d5aeb22f37fe305c1179df0ea1；independent16203c576c0082bd1ad5a27df002b895c7f7c953c5b19ba37e914fb26d4579c4；summaryf42d86fdd0f36b76e8c854af36ded8bf06421295098613590f85d3ef932c7d4c。runner/plan/checker/summary脚本均已执行冻结。

下一边界：固定非负逐层video/joint SSE加权仍会偏好carry，不能翻转这两个现有候选的排序；不否定重新训练第三候选。暂无新方法claim，不自动开功能loss/迭代/rank/校准或student全层扫描。先提出能区分误差方向与实际路径输入变化、并改变部署受限方法选择的具体干预预测，再投入下一GPU。普通工程不当创新；本轮取得有效新诊断与独立验证，目标active，无外部阻塞。

D090 / [报告060](../reports/060_20261004_h3_carry_q_results.md)：**E065b已complete、exit_code0，worker537031/tmux均退出，GPU0已释放。** 只读复用旧39层后，补齐全部200层与9个完整native DiT；续跑3294.944184秒、峰37.6044GiB、新增20608候选局部调用。两阶段共18full，有效候选调用25600，计费物理范围25600–25728；累计数据115.058GB。不要重启续跑或重复执行拒绝覆盖的checker。

结果：200/200层与1600/1600校准格的carry局部SSE均较restart低，层中位比.969830799（−3.017%），层范围.943659–.984777。四点完整video SSE比却为1.129255874/1.109991657/1.004742296/1.044936389，centered亦全部更差；audio比.697972857/1.004331650/.783446364/.719708758。191层选k7，8层k6，1层k3；184条记录曲线单调不增。仅说明本校准局部改善未兑现video整模收益，不能归因具体传播/模态/A4机制，也不证明完整官方PTQ无效。固定smooth、逐linear teacher目标、8候选截断边界不变。没有视频质量/新方法结论。

独立CPU检查全部完成：outputs0.787228秒/21PT，输入、重放、完整输出/误差通过，最大差4.45e−16；selection271.724096秒/2612PT、400个最终manifest与选中导出绑定、3200候选记录及200首轮配对，选中局部SSE差4.18e−16。未重演SVD/native或未选中候选输出。两个原checker与新增stdlib汇总均已执行冻结，不再改写；后者0.075606秒/0forward，保存results/research/E065b/summary.json。evaluate SHA08ac1d487b8683ab9ae7dcaeb42b46a76795435165c8fef3314bfc77c4f56e48；summary SHA1e5c2133fd8adb5a64cefa89dca3d0d5d0d8637c89c5dc2b6d05ad593da37efa。源码resume_v2 SHA1903d7a523b9be08ee256c91d78c1da7882aec41cd2f4061b851c2b776a2838a未变。

下一决定：不扩当前固定配方迭代/rank/seed，不因局部曲线继续下降就做更多搜索。待澄清“同一未见teacher输入上的局部优势是否仍存在”，先查现存缓存可否构成有效对照，以区分覆盖差异与组合效应；当前已完成顶部E066判别，结果见D091，无新claim。正式强基线缺口仍保留，不能用本次结果一票否决必要对照。新[问题筛查](../02_problems/h3_post_carry_candidate_pending_20261004.md)核实ARHQ直接覆盖激活残差协方差加权低秩保护，TwinQuant覆盖较宽可学分解叙事。另有限审查“无害分布输运”已被旧continuity/gauge近邻覆盖，现有少量状态不足以验证，不开展新实验。用户已有研究授权持续有效；本turn完成整模诊断/独立验证并改变投入决定，属于progress，目标active，无外部阻塞。

原E065历史：[报告059](../reports/059_20261004_h3_carry_q_baseline_progress.md)。8BF16校准+1legacy/39层后1800秒监督下退出，原JSON仍running且保持不改，terminal_observation另存；4992已记录局部候选调用、尾部0–128未知，原峰43.41GiB。v1 CPU合法padding分段检查失败0CUDA，v2修复后check2.545秒通过。E065b续跑CPUcheck47.791秒/1923文件，独立3600秒预算，正常结束；不将两段说成原30分钟内完成。

本轮[激活残余表示核查](../04_prior_work/h3_activation_residual_representation_20261004.md)未得到独立立项剩余：SVDQuant已隐含残余低秩保护，HeadQ/QUADS有直接相关结构；不为换adapter输入开新方法实验。E005已有RNE midpoint局部对照，不重复。E061仅初始化替换，不能将其20%门槛扩大为否决必要carry-Q基线。

D088 / [报告058](../reports/058_20261004_h3_depth_four_corner.md)：E064 complete467.429秒，两原状态×plain/legacy×blocks0/12/24/36/49，6full/70local、960native+pack/752SDPA；峰40.90GiB、24.546GB张量。六完整路径复现历史，四block0 exactzero；16非零格raw net_echo/eout全部负−0.338%至−3.386%，center后亦负，预设gate两臂通过深度均0。独立NumPy175.933秒/0CUDA，104PT重hash、actual/raw/vel及局部签名全部通过，最大统计相对差6.72e−14。停止本轮稳定净有害反馈候选，不追加后缀干预/训练/网格。posthoc展开显示量化有限输入响应能量反而增加，被与qB的负交叉项抵消，不能称quantized map更contractive；W/A来源仍未解，非视频质量结论。worker3565043/tmux已退出，GPU0 0MiB/0%。CLQ/跨层联合补偿等覆盖一般叙事，不立claim；后缀相干草案未执行。本轮新增有效判别并改变投入，属于progress，目标active。

D087 / [报告057](../reports/057_20261004_h3_qknorm_radial.md)：E063 complete69.667秒，2完整BF16/12局部native QKV/6共享激活packet，峰41.97GiB。fixed same-smooth fresh Q(Ws) rank0 对 legacy rank32，两teacher×三预选层的raw QK改善只有0.454%–1.383%；径向净占比四负，两正仅0.373%/2.853%，实际norm后改善0.384%–1.760%。三个层均不满足预设必要条件，停止径向预算错配候选；不追加attention反事实/训练或扩层找阳性，不外推所有QK校准。实际Comfy布局与norm/RoPE hook核验通过；独立NumPy3.832秒/0CUDA，12PT重hash、输入/raw/vel逐byte一致，统计归一差5.98e−16。共享完整packet仅执行源/签名合同审阅，不称CPU复现kernel。worker/tmux已退出、GPU0已释放，当前无新GPU任务。本轮真实判别改变投入决定，属于progress；目标active。F1/F2均停止，无active新方法claim，下一须新的可区分问题，普通工程不当创新。

D086 / [报告056](../reports/056_20261004_h3_swiglu_interaction.md)：E062 complete34.437秒，2完整BF16＋6完整输入局部native fc1，峰41.77GiB。固定两状态×block0/24/49的SwiGLU乘性交互净能量全负，raw为−0.017%至−1.972%，去通道均值/比例后仍全负；实际BF16四角及down/gate算术均不改变判断。停止二阶乘性交互修正，不扩层/提示/adapter训练；不否定SiLU单支曲率或所有FFN问题。独立NumPy6.350秒/0CUDA，12PT重hash、两teacher输入/raw/vel精确复现，统计差≤4.97e−16。原worker/tmux退出、GPU0释放，无新任务。SPEAR已有门控低秩补偿、NA-LoRA已有SwiGLU残余分析，泛化非线性adapter不能认领新颖；QK备选尚未执行/立项。两次CPU启动失败及v1源保留，v2仅修路径、已有av追加依赖查找，无GPU重跑。

D085 / [报告055](../reports/055_20261004_h3_initialization_results.md)：E061完整200层双初始化对照complete218.452秒（含22.281GB导出/装载）、9完整native DiT、37.611GiB峰；旧基线精确复现。weight_init/legacy视频SSE为.882778/.925071/.998096/1.058650，weight_init/error_init为.917848/.902244/.993839/1.004678；四点均不满足≤.8的双基线门槛，未补4shifted/视频/fullPTQ。停止固定smooth初始化路线，不扩rank/seed/提示；不能将prompt×timestep混杂当早晚步规律，不否定完整官方校准。CPU summary complete0.561秒，独立NumPy复核complete0.823秒/0CUDA，21份PT重hash、9输入身份与重放通过，统计最大相对差3.75e−16。原worker/tmux退出，GPU0释放。


D084 / [报告054](../reports/054_20261004_h3_baseline_identity.md)：E060 CPU0.725秒，state重hash与E009一致，200层600个smooth/A/B字节相同；packed主体未重hash。当前生成脚本重复初始化Q0、换random SVD seed，官方calibrator支持carry-Q；残差初始化本身官方支持，不是错误。历史生产源码SHA缺失，不能反推执行过程。后续称legacy smooth+低秩残差基线，既有同实现对照保留，但不据其排除完整官方SVDQuant。候选数实际2–6，max50不是50次优化。E061固定smooth全200层初始化目标对照CPUcheck已过（2.339秒/6输入/200weight headers/0CUDA），GPU执行现已完成，结果见D085；非carry-Q/完整官方复现或新方法。尺度/支路筛选均未建立可立项残余。


D083 / [报告053](../reports/053_20261004_h3_activation_oracle_control.md)：E059六个zero native完整调用逐元素复现历史；首个activation residual oracle在block0 FFN-down数值分辨率门槛停止，BF16加回舍入/补偿RMS=10.3384%略超预设10%。FP32补偿公式通过FP64检查；7attempt、6完整forward、0完整oracle、82.839秒。没有完整来源主终点，**不能作激活机制阴性或权重主导结论**。按原资源规则不改阈值/重跑精度网格，当前activation时序方法不晋级；该控制工程不是创新。原worker退出、GPU0已释放，独立NumPy复核已完成（0.420秒/0GPU，六输入及24输出数组精确一致，FP64补偿张量差0、统计差≤2.22e−16）。

D082 / [报告052](../reports/052_20261004_h3_update_arithmetic.md)：E058 CPU2.024秒、0新forward，精确分离第二次更新算术。SVD交互范数比从.422/.441降为.305/.413，仍高于预设.2；但交互净能量效应仅+0.17%/−6.17%，不能把交互范数直接当损伤目标。独立NumPy差9.51e−12。固定W/动态A来源问题仍未判定，E059未获得有效完整oracle输出；新精度合同笔记只约束未来协议，不自动重跑或称FP32修正为创新。

D081 / [报告051](../reports/051_20261004_h3_temporal_diagnostic.md)：E056→E057完成H3时间轴判别。SVD两个窗口实际传播后F/C正交叉项占自身能量和40.0%/48.5%，但输入漂移交互范数占简单相加范数42.2%/44.1%，超过预设20%门槛。支持有限两步增强，不支持弱交互线性抵消；不直接推进互补权重或扩网格，无新方法claim。E057累计5次BF16 forward（1历史replay+4反事实），v2阻塞D2H修复检查竞态后完成40.147秒/40.896GiB，原deadline保留，原失败源/结果保留。独立NumPy逐PT复算最大差1.68e−11，确认预设gate不通过。所有worker已退出，GPU0已释放。

D079 / [报告050](../reports/050_20261004_h3_phase_diagnostic.md)：E055已完成H3 CFG1三个teacher状态×plain/SVD的CPU同状态输出几何诊断，0.161秒、0CUDA/0模型forward。横/纵/对角的patch内外协方差差，六组全部小于普通iid hidden经同一真实BF16 head的解析模板；独立逐数组复算确认每组37/37时间片通道均值也更低。停止“量化导致额外patch边界断裂”候选，不增加hidden捕获/decoder干预/phase loss。实际误差空间相关不等于量化特殊机制；有色/各向异性普通误差未被排除。完整summary SHA 5cdb65919a1ffa4d30ecf5a001f48cbe68f9c6d9825b3a0d66f6df856161f33f。

E054仅有[计划](../06_experiments/E054_h3_fused_boundary_plan.md)、预备runner/launcher及py_compile，无CPU结构check或GPU结果；按用户提醒停在准备状态。旧C006通信候选继续关闭，22.85%单边界工程收益保留，不自动重新立项。E053 Wan14B完整PTQ未启动，不再作为近期主线。E059已按数值门槛停止；E061–E063亦已完成，当前无新GPU任务。

E056复用E014/E015两个teacher窗口p30 s5→6、p36 s14→15，SVD video去均值/输出比例后的cosine为0.701/0.622，保留92.8–98.7%原始误差能量；独立NumPy复算差≤6.11e−15。不能把早期Wan CFG6阴性结论直接推广到H3，也不能作CFG因果归因。E057进一步限定结论如上；E059未取得完整来源判别，无active新方法claim。De-biasing Diffusion全文仍不可取得，不宣称跨步互补的创新成立。下一次投入须有新的来源/可干预性预测，不能仅因存在正相关扩展校准、训练或参数网格。

H3模型身份、CFG1及2×2输出行序见[入口](../02_problems/h3_output_geometry_entry_20261004.md)。本地pruned约20.1B存储元素，不称完整33B；原配方20step、124帧576×1024、video/audio shift12/3。

## 保留的阶段证据

[汇报049](../reports/049_20261004_research_brief.md)按动机→实验→结论整理7项工作。H3真实native DiT有1.22×工程收益，官方FP4 attention在固定native线性条件下再快1.33–1.37×；局部误差/中心校正没有稳定视频改善证据。H3模态重加权、跨模态两步传播、稀疏预算膨胀、音画/统一动作变慢路线均已有限检验并停止或parked，不无理由重启。

Wan仅保留已有诊断：E045两seed单次末步SVD即可致损；E046末步BF16未充分修复；E047/E048匹配校准约5h20完成，残余碎片仍在。14B E049 BF16/E051 plain配对已完成，有成本收益但未匹配质量；E050仅校准数据采集完成，E052仅block0资源pilot complete1647.21秒/峰61.86GiB GPU、367.46GiB CPU，不是完整SVD checkpoint。

[评估规则D076](../01_literature/evaluation_map.md)：MJ总分含五方面含安全/公平性，仅作辅助；Wan既有81帧只抽0/10/…/70，每帧448方形输入。自动指标、非盲抽帧观察和反复查看的4提示×2seed都不是独立质量证明。禁止把分差解释成画质百分比，或单独按总分接受/淘汰方向。

## 工作区与调度

用户恢复授权已生效；2026-10-04本轮get_goal已确认active，后台目标恢复。本轮前E055完成且改变候选决策，属于实际进展。本轮E056新增有效诊断，不把查新页数或基线修补作为创新成果。

仓库main，不push；保护用户dirty scripts/rcm_vbench251_worker.py及历史scripts/minimax_h3_svdquant_common.py。已执行实验源/结果不覆盖，改动另版本。大模型/media/cache放/data1/models/svdquant-wjq；根盘仅约3.2GiB、DATA1约1.1TiB可用。8×RTX PRO5000 72GiB，每次GPU运行前重查；不得干扰他人任务。E059等旧任务已退出；E061亦已完成退出，当前无新GPU任务。shell当前需require_escalated。

Native Python：/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python；复用已保存native/fused environment与DATA1缓存，不重新安装。E055 CPU无需CUDA环境。用户指定research skills已安装且使用。

[决策](decision_log.md) · [实验日志](../06_experiments/experiment_log.md) · [报告索引](../README.md) · [本轮前完整历史状态](../99_archive/superseded_notes/current_state_before_H3_E055.md)。归档内相对路径以原00_state位置解释。
