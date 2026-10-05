# Decision Log

## D001 — 2026-10-02
决定：安装14个community skills并启用持久research_state；先做文献/证据/算子三项有限审计。
依据：用户要求从失败研究重新开展有新意且可验证的工作。旧仓库并不直接证明任何新观察。
结果：源码和旧结果原样保留；大文件使用数据盘；先验证假设再扩大GPU预算。

## D002 — 2026-10-02
决定：修复H3低秩输入精度，但不把修复当新方法或假定它解释全部失败。
证据：10项CPU回归；E001四个真实block0 case视频token聚合NMSE仅改善0.78%。
后果：旧state作固定参数诊断；正式比较须重新校准。采样原始shape，按模态拆指标。

## D003 — 2026-10-02
决定：最多保留两个探索问题：跨步误差结构（E002）与H3模态误差因果归因（E003）；batch global scale挂起。
证据：定向文献碰撞；E001全block指标与video-token指标反向；global-scale合成控制不足以说明实际服务失效。
后果：先做可证伪诊断，再决定方法。C002互补权重方案因SR近邻和双权重内存开销暂挂起。
上一目标轮分类：progress（安装skills、完成审计及kernel证据）；本轮新增基线修复和真实配对结果，不把计划等同完成。

## D004 — 2026-10-02
决定：停止C002跨步互补全W4A4误差的方法路线，C001强版本rejected_in_tested_setting。
证据：E002两prompt conditional高相关0.603/0.654，但真实CFG6只有0.102/0.184；触发预注册≤0.2停止门槛。BF16缓存回放误差0。
含义：不把conditional结果冒充实际solver输入；W4A16仍相关不能证明它主导完整W4A4。后续只推进E003因果筛选，不扩大E002。

## D005 — 2026-10-02
决定：H3实验必须显式固定并记录attention后端；修正E001第一轮的BF16 attention标注。
证据：恢复环境自动选择Sage，H3调用链确实通过该选择。E003在forward前guard退出；显式torch重启并添加SDPA调用计数/BF16断言。
复验：E001 torch四case完整成功；hook修复video NMSE+0.19%（无改善），SVD相对plain全block−9.99%、video+76.16%。保留两组数据，主张为模态取舍待因果检验，非hook收益。

## D006 — 2026-10-02
决定：E003已完成，只保留为受限的局部/最终目标取舍证据；先做E004廉价强对照，不扩benchmark。
证据：两步SVD局部总误差下降而最终video NMSE+13.67%/+8.52%，audio改善；实际torch BF16 attention、zero replay0。
边界：自然误差能量不同、一个旧校准prompt、block0，不能推出内在模态敏感性或视频质量。MBQ/MixDQ/RGSQ/PulseQuant严重重叠。
后果：E004固定同一rank容量与大部分参数，比较普通校准目标；即使有效也先归为工程修复。真正heldout/跨block仅做有限可行性核查。

## D007 — 2026-10-02
决定：E004按预设门槛停止模态加权当前路线，不扩rank/权重搜索；C004只保留原p1受限诊断。
证据：三臂局部video改善39–47%，完整输出仅video4对pooled−6.74%（<10%），normalized+3.32%；p11/p46的SVD/plain视频音频排序与E003相反。四zero replay0、固定参数哈希不变。
含义：没有稳定模型级模态取舍；有限控制阴性不证明一般重加权无效，也不提高新颖性。

## D008 — 2026-10-02
决定：将下一有限诊断放在原生projection与原生FP4 attention接口；先通过格式与kernel入口门槛。
证据：E005真实2层×2步×2recipe的native/QDQ对BF16误差变化最大0.175%，未改排序；E006原生attention4case smoke及固定scale7类byte parity通过。
边界：不把kernel可调用当吞吐/质量收益。文献已有SVD+Sage、scale search、量化近似交互；只测特定可干预机制。不同模态实验不因阴性自动升级为新理论。

## D009 — 2026-10-02
决定：E006按预注册停止C005当前有害QKV-scale-switching路线，不跑剩余网络continuation、不改门槛。
证据：固定12case视频行，交互占比中位31.71%，组合/加和误差中位0.9608且0/12放大；oracle固定scales中位改善6.46%<10%。48次独立pack byte parity0、12次BF16回放0，root独立脚本复算一致。
含义：大交互项不能推出额外损伤；负交叉项抵消。局部oracle小收益不是部署或视频质量结果。

## D010 — 2026-10-02
决定：先完成E007完整Wan原生NVFP4基线和profile，再选下一系统方向。
依据：旧模型多数QDQ+BF16 GEMM，缺真实运行瓶颈；静态核查确认300weights有原scales可恢复，无需先重跑PTQ。
约束：保留旧smoothing/LR语义，先慢reference正确性后快速packer；同环境同attention对照；不把共享down、常规fusion包装新意，不扩大无证据校准搜索。

## D011 — 2026-10-02
决定：E007完整native正确性通过后进入真实profile；不将数值传播差异直接升级为新机制。
证据：300weight roundtrip和全30block bypass全等，4个主支单层NMSE5.4e-6–1.27e-5；完整native/QDQ终点NMSE4.28%，两者对BF16为16.40%/16.19%。快速packer真实三input+7synthetic全byte一致，另检查无效global/NaN/Inf及SF0非零code；零输入使用数值等价canonical表示。
边界：一次校准raw call、没有自由rollout/视频质量；尚无速度结果。单层算术微差不代表完整模型数值等价，完整分歧也不证明native质量更差。

## D012 — 2026-10-02
决定：保留原生Wan部署作为有效工程基线，不报告相对BF16加速。
证据：独立三进程同GPU的median BF16/QDQ/native为1.770/3.666/1.975s，全部输出SHA匹配。native常驻模型0.86GiB、allocated峰值3.01GiB，但reserved7.01GiB高于BF16的5.77GiB。Chrome严格kernel归因后，main节省约215ms被LR/pack/smooth抵消。
后果：先比较成熟fusion和更大的H3，未融合Python路径慢不支持NVFP4固有限制论。raw profiler有GPU annotation重复，保留raw并另行修正，不能重复累计。

## D013 — 2026-10-02
决定：完成E008一对真实视频以闭合采样验证，暂不扩质量benchmark。
证据：seed1 BF16四步历史输入/velocity exact；native共1200GEMM/flags全过；同随机输入独立递推，同VAE解码，77帧480×832。四帧可见服装/头盔/背景变化，不能据此比较感知质量。
后果：局部或最终latent误差不替代视频评价。KV token permutation候选由VC-Attention直接覆盖，未做实验，park。

## D014 — 2026-10-02
决定：E009先建立H3完整resident原生基线；保留已知fusion为部署对照，不新增研究主张。
证据：200W确定导出19.27B元素数值exact，BF16 offload/resident 50block+双endpoint SHA exact。FlashInfer SM120已有fused-up可运行，长M的QKV组件1.53×，FC2和短M无收益；单层融合改变舍入，需整模另验。
边界：普通fusion、fake/native差异本身、V低秩跨attention均有强先前工作；KV聚类与V桥接候选park，不投入GPU。当前完整native因block1.fc2非零组SF0的保守domain guard停止，不把它当量化发散或native不可表示证据。
后果：先捕获同SHA失败输入，验证旧H3 QDQ在SF0是否数值等价于E005 canonical zero。原失败和frozen源保留；只有真实证据通过后允许新增callable扩展已验证域，原200W无SF0不重导。性能测量等待完整slow/fast SHA通过。

## D015 — 2026-10-02
决定：E009正确性及性能闭环完成，保留H3原生部署为实测工程改进；不升级为方法贡献或质量等价加速。
证据：95个SF0音频组旧公式本来归零，真实3.21亿元素数值exact、独立adapter完成200native GEMM且slow/fast 50block及双输出SHA exact。三独立进程BF16/QDQ/native中位8.260/20.778/6.787s；native1.217×BF16、3.061×QDQ，allocated峰值16.80GiB，reserved30.09GiB，启动峰值37.62GiB。全部5次输出各自参考SHA匹配，三arm trace kernel分类独立重算精确。
边界：一个原校准输入、eager实现、仅完整DiT，不含生成循环/VAE。52个小元数据D2H同步的CPU等待并非6.6s纯传输，也不是scaled_mm每层global读回；不声称CUDA graph。已知fused-up只有部分形状收益，未加入此次整模基线。
后果：所有当前GPU任务释放；CPU核查heldout配对生成的资产/预算与稀疏路由的近邻覆盖，只有能改变研究决策的实验才进入下一轮。

## D016 — 2026-10-02
决定：E010使用原提示表p30/p36的文本与seed做两例独立生成；E011暂不进入GPU。
依据：在已检查记录中，两提示未用于旧PTQ校准或旧生成，原叙事时长均≤5秒，按这些条件预选且未看任何新输出。本地TE/DiT/VAE齐备，延续旧20step/CFG1协议并明确不同于pipeline默认50steps。复用原pipeline __call__/step，分阶段加载，仅冻结共享文本embedding与两模态initial noise。
E011：宽泛的稀疏路由稳定性/认证回退已有强近邻；剩余mean-sim guard问题需要合法完整运行路径。官方utils可CPU导入，但真实K尾块masked0的normalize产生0/0后转boolFalse，可能属于保守处理，语义尚未证实；六个本地环境无已验证SM120 consumer。不为弱候选新造后端，不把静态风险称为实测bug或机制反证。
边界：E010两提示以文字动画/产品微距为主，不覆盖人物口型或通用运动；若BF16本身不遵循提示，不能归因量化。之后据其输出形成的方法假设必须另设未看过的评测集。

## D017 — 2026-10-02
决定：不为朴素continuity/Stein校准目标启动实验，不新增claim。
证据：ICLR2024 gauge工作明确给出divδ+score·δ=0及L2惩罚分布无关分量；FDM已直接提出该continuity residual及条件化训练目标。DMD2去掉seed绑定回归，LongLive已用于NVFP4视频。原生动态量化的不连续面还使普通/STE导数漏掉通量；有限步Euler不继承连续gauge保证。
后果：只保留数学/先前工作碰撞记录；不以新的术语重包装已有目标，也不把相同分布与配对误差的区别当新发现。E010继续用于真实部署质量诊断。

## D018 — 2026-10-02
决定：E010配对生成与有限评价闭环，保留完整视频与原始逐题回答；不扩大为质量等价或新方法结论。
证据：p30/p36各BF16/native，124帧1024×576/24FPS及32kHz双声道；160步文件、4最终latent和全部媒体独立核验，共8000原生GEMM/pack通过。video自由轨迹NMSE .2261/.3253，不能当质量分数。VisionReward 116条有效回答，p30逐题相同，p36两项开头形状变差，分数.196032→.180285。
边界：旧20步协议非默认50步，两个提示以文字/产品为主；本轮已看过，后续方法必须新设heldout。音频未评价。旧用户工具的大写Yes映射不适用于本轮小写yes，保留原文件，独立冻结评价器；未取得历史原始回答，不否定全部历史分数。总墙钟含加载/hash/落盘不作加速比。当前GPU任务全部释放。

## D019 — 2026-10-02
决定：宽泛量化×高阶/自适应求解器候选park，不追加GPU。
依据：SAQ已直接处理多个方向评估与高阶收益损伤，QuAKE清理多步输出，mixed-precision RK已给出embedded误差导致错误接受的实例。原生NVFP4微块跳变是未验证的具体setting，不能仅凭setting立题。本地H3无embedded controller，Wan rCM是重噪consistency式；H3实际BF16输入/时间表插值后cast本身不连续，末步sigma很粗。
后果：记录严谨区分同一量化RHS的积分误差与相对BF16的模型偏差；没有适合且对BF16有价值的sampler前不造新采样器。legacy num_grids为smoothing-alpha搜索，已排除“时间参数切换”的错误前提。下一候选只做条件响应的文献/CPU筛选，尚无观测或claim。

## D020 — 2026-10-02
决定：条件差分新损失kill、方向park，本轮不追加GPU或训练。
依据：DASH已研究压缩后的guidance-gap衰减及双分支监督；GAMP直接使用差分重构并指出共同漂移。独立双prompt MSE满足L=2||mean_error||²+.5||difference_error||²，已约束绝对差分；新增差分损失只是关系度量的重权重。原生CFG1 H3与两真实提示不同于其训练合同，但仅换setting不够。
剩余：可保留待预注册的极小异常诊断，询问语义编辑的teacher方向gain是否比幅度相近的普通编辑更易下降；必须区分g≈1的正交噪声、弱分母放大和真衰减，同时确认BF16响应可靠与packed layout合法。无matched control或teacher不响应则停止，不无限搜索提示。当前没有现象证据或正式claim。

## D021 — 2026-10-02
决定：在不撤销“新差分损失kill”的前提下，预注册E012一次有界数值排查，先teacher后native；不继续纯文献猜测或训练已有目标。
依据：E010已提供同一原提示的完整BF16状态；p36的一处光照左右方向可精确交换，两个固定普通编辑均保持813tokens。独立CPU核对source/真实packed/±1ULP合同通过。此前goal轮次属于progress：完成E010完整评价归档及两个方向的直接先前工作筛选。
预算：26BF16调用用于4条件×原点/双向共同1ULP×2时刻+重复；teacher差分SNR≥10且有两步normratio[.5,2]的预选对照才运行6native。固定control优先级、选择只看teacher且先冻结。25分钟总墙钟、GPU0、无VAE/采样/训练。
边界：两时刻不是独立样本，BF16数值响应不证明提示语义已被理解；只有强选择性gain/能量下降且普通编辑保持，才要求另行独立复现。所有门槛在新模型输出前写入plan/manifest，不合格即停，不追加提示。

## D022 — 2026-10-02
决定：E012按teacher gate停止，native不执行，不以大条件差分误差立题。
证据：26BF16调用完成，两个原点video/audio与E010 exact、重复误差0；方向差分对共同±1ULP的SNR为1.032/.642，低于10。颜色编辑幅度匹配但SNR1.732/1.220不合格；材质早期normratio5.71超界、晚期不稳定。独立CPU核验74文件、全部输入/输出/扰动及选择逻辑，native报告和目录均不存在。
解释：同一输入可复现不等于其微小条件差分在邻近状态稳定；该实验前提未成立，不能当量化损伤不存在或BF16语义控制失败的证据。新差分loss的先前工作KILL保持；停止本次局部代理，不再扩提示搜索。
工程：首次prepare成功，BF16零调用时遇标量SHA记录错误，原失败保留；v2只展平字节做哈希、保留原shape并复用TE，8真实标量CPU检查通过，原25分钟deadline不重置。全部GPU释放。下一步需先定义可验证的生成行为和BF16任务基线。

## D023 — 2026-10-02
决定：根据三个独立审查，先做一次E013封闭投影行为筛查，不再用缓存便利和微小局部差分决定研究对象。上一目标轮为progress：E012执行、独立审计、停止与报告011全部完成。
协议：新写两球接近/远离提示×固定两个seed，50步、576协议、双模态同初始noise。第一seed两BF16均成功才第二seed，四BF16全过才四native。自动颜色/间距/大小读出先通过CPU合成控制，人工只能否决读出有效性，不能升级自动失败。预算首次GPU起90分钟，GPU0/5，最多400DiT。
依据：E010只两复杂广告20步，E012未验证真实方向执行；当前尚无跨case稳定量化功能损伤。已有运动评价及反馈PTQ近邻，此次不构成新贡献。
边界：只读二维投影相对间距，不是3D物理或完整提示遵循。teacher失败停止此任务，native未知不算退化；原生至少两例额外失败且跨两方向才触发未来独立复验。不得更换提示/补seed/改门槛追阳性。

## D024 — 2026-10-02
决定：E013按首对teacher门槛停止，第二seed/native均不执行，不扩大toy提示窗口。
证据：首对50步BF16完成，188文件独立核验、100DiT/10200SDPA/0FP4、共享输入与媒体正确。自动unknown/unknown，92/124和82/124有效帧；实际帧级失效只有rho<1.5，共32/42帧。root及独立agent查看接触图/曲线，接近运动清楚，远离先靠近经过蓝球。预注册rho guard比不接触严格，不能把这种拒绝写成teacher无运动能力；摘要size不稳定标签也不能当真实变形。
含义：功能读出前提未成立，无量化比较或新机制结论；不事后放宽阈值。代码输入及CPU合成控制在GPU前冻结，CPU发现的二维embedding检查错误在GPU前修复并保留，原生实现未改。首次GPU起约452秒完成生成，GPU0/5释放。
另：P002批独立性的一般目标与per-request scales/grouped GEMM常规处理不构成新贡献；公开per-token量化接口存在，不因本地API限制造新后端。无真实服务需求、质量损伤和不可接受隔离成本证据，不新增GPU。下一阶段应从具体未满足使用约束选研究对象。

## D025 — 2026-10-02
上一目标轮分类：progress。E013首对完成、188文件独立验证、两例unknown按门槛停止；报告012及全部索引归档，非仅计划。整体论文目标仍未达成。
决定：不再要求先证明量化造成可见能力损伤，才允许核查性能/容量取舍；补E014完整plain_h3_recipe强基线。
依据：E007低秩/平滑开销可超过主支节省，而完整plain从未建立，E003仅局部干预不足；SVDQuant原文NVFP4消融已显示收益随配方改变。三个独立新审查均未找到成熟baseline已经失败的部署机制；先对照，勿新建系统。
协议：同原W、同200层覆盖、同H3 signed-E2 ties-lower/E4RNE，BF16/SVD/plain三臂；plain从原W重pack不从residual删LR。固定E009p1s0+E010teacher p30s5/p36s14，匹配p1性能。45分钟总墙钟、最多30DiT，无采样/训练、无10%收益gate；数值与质量结论分开。不把去SVD或训练移除额外算子立为创新。

## D026 — E014完成：保留两套基线，复杂配方不单调改善

2026-10-02。200原W NVFP4导出exact，3state×3arm+15bench共24DiT，269文件独立CPU审查通过。完整DiT BF16/SVD/plain中位8.291/6.804/5.823s；plain延迟较SVD少14.4%、稳态allocated16.81→15.06GiB。plain/SVD video误差比2.037/.850/.999，audio .869/1.621/1.080；不符合全部端点不更差的替换条件，保留两套基线。此为整个smooth+LR配方比较，非LR独立因果效应、质量等价或新方法。原调度器因memory0/util13尾采样停下，独立冻结resume1经稳定空闲检查续跑，原deadline不变且无重跑，总墙钟624.7s。阶段013完成。

后续不追加E014输入/调参追阳性。对实际跨模态轨迹传播先做近邻与最小反证，不能把单状态排序交叉升格为机制；普通FP4 attention部署仍是另一个强基线缺口，不因E006交互门槛失败就宣称无价值。目标仍未完成，无可投稿claim。

## D027 — E015预注册：有限四角续接，不重开普通模态加权

仅E014 p30/s5和p36/s14，以三臂已存共同teacher输入velocity经原scheduler构造实际one-step B/Q下一状态；量化两臂各四角继续native，再更新到t+2。两个BF16 next重放加16量化调用，共18DiT/20分钟/60GiB。CPU原step已独立验证8模态步byte exact。研究读出是oracle另一模态恢复是否改善净endpoint误差及改变配方排序，不能仅凭cross-delta或不同sigma称刚性/新意。TurboT2VA cross-JVP、SynchronySparse、DeltaQuant H3近邻已核查；当前仍无新方法claim。源码/CPU构造实施中，GPU未启动。

## D028 — E015完成：跨作用不稳定，停止本轮机制扩展

18次原生续接、181.2秒，CPU/GPU首步byte exact、两个BF16次步历史exact，101文件独立审查通过。oracle恢复另一模态对净error能量作用有正有负，p30/SVD音频改善11.418%是保留的局部观察，其他case作用不一致；4组plain/SVD排序均不变。不围绕E014排序交叉开发新模态预算loss，不把未翻排名写成传播不存在。全部4corner/Gram交叉项保留，无追加样本/训练/视频，当前路线stop，阶段014完成。下一实际缺口为现成低位attention整模成本—保真对照，未执行；顶会目标仍active且无claim。

## D029 — E016预注册：完整成熟低位attention对照

上一目标轮为progress：E014完整plain基线与E015四角续接均完成并独立核验，报告013/014已经落盘。原顶会目标仍未完成，无阻塞、无active claim。E016固定原SVD主层、三个原state，50main有效长段替换官方FP4 attention，原padding/refiner保留；True/False各用自己QKV原官方配方。新环境先BF16与SVD+BF16attention各三状态byte exact E014，再比较低位两臂。27DiT/30分钟/60GiB，正式性能只比较本轮3SVD臂；不借旧BF16计时、不开发kernel、不宣称简单组合为贡献。源码/CPU验证实施中，尚未GPU。


## D030 — E016完成：已有FP4 attention有整模收益，保留两种保真取舍

27DiT/323.64秒完成，GPU5释放；两级历史三状态byte exact，71来源冻结，独立CPU核验116文件。固定SVD主层下，本轮BF16/block/global attention中位6.791/5.124/4.968秒，1.325×/1.367×；稳态allocated16.806/17.713/16.806GiB。两低位模式全部六端点NMSE增加，global仅p1两模态优于block、后四项block更好。保留三配置，不以固定state误差作生成质量或泛化证据，不把成熟组合当新方法。全部在线布局/centering/quant/correction成本包含，main元数据缓存共同；无旧轮BF16计时拼接。报告015完成。

CPU汇总首次因预期CPU/GPU CUDA_VISIBLE_DEVICES差异直接比较而停，失败JSON/log/源码保留；仅审计器允许该预期字段差异且仍强制实际GPU5，其余检查未改，原运行源码和数据不变，无GPU重跑。环境缺av仅文件枚举以旧核实清单替代，原全部prerequisites原样执行并finally恢复函数。

现成correction消费融合、compact布局与开放Q分块PR已核查，不能当新意；缺已合入的同合同on-the-fly开关也不能证明存在研究机会。下一步从完整生成的成本—质量边界判断可用基线，不开发新的correction kernel或给本轮套论文主张。目标仍active、无阻塞、无可投稿核心贡献。


## D031 — E017预注册：完成真实生成边界，停止过度基线扩展

上一目标轮为progress：E016 27次完整DiT、同轮性能/显存、116文件独立核验、官方实现近邻和报告015均完成。仍无论文核心贡献，目标active，无阻塞。E017复用E010两已见prompt与原noise/20步协议，固定SVD linears，两个官方FP4 attention模式各两自由生成，共80DiT+4解码；旧四媒体与新四媒体同次VisionReward全29题，共232题，旧回答变化单列。30分钟共享预算/GPU5/60GiB，无新teacher gate、不重复导出/大权重SHA/完整teacher生成，不以局部NMSE先判质量。

独立策略审视认为此前串行门槛过多；完成E017后停止扩展当前基础网格。成本匹配地用低位节省时间增加普通Euler步数，只能先作为部署问题，LongLive-2.0已有NVFP4步数—吞吐对照，不当新方法。现阶段没有足以直接启动新loss或kernel开发的残余claim；需要从实际生成表现与既有技术边界选出可证伪问题，而非将成熟组合包装为贡献。


## D032 — E017完成：低局部误差没有保证本例更高视频评分

80DiT、4新视频与8视频同次232题完成，总658.57秒，273文件独立CPU验证、160原scheduler重放/实际input与链全过，GPU5释放。旧四媒体raw回答/完整文本/input token完全复现，232题全有效。p30四臂评分全0.010473；p36 BF16/SVD/block/global=.196032/.180285/.100116/.180285。block相对SVD仅额外四题（运动平滑/真实、稳定、细节）yes→no；global全部题同SVD。block在p36局部teacher s14与最终latent都比global更近BF16，不能用低张量误差直接选视频配置。root已查看四臂固定六帧图，block约2.04秒花瓣边界更模糊；未独立完整观看运动或听音，不把六帧评分升级全面质量结论。报告016完成。

保留global作为这两个样例上较快且未改辅助回答的已有部署参照；不立质量等价或新方法claim。停止扩大当前baseline网格/额外步数；原目标active且无阻塞。下一问题先基于精确机制与现有方法差异筛选：P4 E4M3尺度的分区可组合性已写精确边界，PNQ row-sum不自动推出partition invariance，但BAPS PoT max与Sage原论文tile scale构成直接why-not-simpler。Qmean分组与视频实际layout相位另在只读核查，未据单例反推机制/启动GPU。

## D033 — E018完成：保留布局相位结构，不升级为视频闪烁解释

上一目标轮完成E017真实生成/统一评分及报告016；本轮完成E018，属于progress。近邻核查明确PAROAttention布局量化、DeltaQuant三维均值与Sage固定128分组，普通重分组不算贡献。真实p36视频起点1227=813text+414audio，576tokens/latent帧，起点phase75/11交替；不是误差或输出帧周期的先验结论。

GPU前冻结协议和77文件，固定E017 block p36s14、blocks0/24/48；一次完整DiT重放actual/raw/velocity全exact，捕获真实QKV后单独27次attention。只移video Q，K/V packets与storage固定，0/64/128三相位、BF16/block/global三模式。90.79秒，两个进程均退出，GPU5释放。142文件独立CPU核验：BF16/global整段byteexact，block128主mask帧2..34 byteexact，block0重现原router。

64相位使误差方向明显改变（差异能量/原误差.292/.624/.350），平均NMSE却只变+0.165%/+0.062%/−0.170%。误差能量有可干预的奇偶分量，非33帧DC泄漏；但+256token的同相位伪边界也交替，部分层更强且反号，不能叫真实帧边界特有。全部正负交叉项、每row图和边缘保留。

看到主要结果后追加明确标注的纯CPU向量投影（10.32秒，无GPU）。原/64交替投影占比1.561/1.571%、2.345/2.352%、1.717/1.723%；投影cos+.545/+.544/+.429，未出现整体奇偶翻号。没有使用1/33显著性门槛，也不据小占比断言无质量影响。原生P随Q改变，是完整operator效应，不归为纯Q4误差。

决定：保留有限可干预布局观察，当前路线park，不据此做新loss/分组kernel或扩大视频网格；不是严格证明该机制无害。报告017和索引完成，仍无active claim或可投稿核心贡献。下一轮用已捕获真实QKV/packets低成本核查P4分区合同的实际量级，直接包含已有PoT max/Sage tile缩放对照，不先搭多卡框架。CPU一次预检缺CUDA_HOME在0GPU调用前停止，失败保留，补原协议env后通过；GPU没有失败/重试，冻结源未改。目标active，无阻塞。


## D034 — E019失败定位与流程重置，转向主权重QAD

2026-10-03。本轮有可复用数值定位与明确实施入口，但没有新的科研贡献；目标active，无阻塞。原E019在第一BF16 full+LSE输出未逐字节复现E018时按协议停止。原inputs/packets/correction通过；原失败未保存实际输出，不能估计原次漂移。另立补充协议，固定block0两调用False→True，先落盘再比较；False全tensor复现E018，True仅39/161,559,552元素改变，NMSE2.84546e-12，独立CPU全输出复算通过。ReturnLSE选择不同CUDA模板，尚未定位具体机器算术变化，不称bug或质量机制。

诊断CPU一次文件记录schema失败0调用、worker一次3MiB显存绝对零检查失败0调用，均保留；v2仅改重复空闲检查为排除外部计算进程，launcher稳定空闲及所有数值参数不变。累计仅3次attention、0DiT；全部GPU进程退出。原分片对照、真实数学七合同与模态mass统计都未执行，不以此作阴性结论。P4方向停止扩展，原因是强近邻覆盖与研究机会成本；保留源/自测/失败，不扩新kernel或框架。报告018完成。

纠正“算子边界→小审计→park”循环。下一阶段直接建立rCM-Wan主权重QAD成熟基线，300矩阵可学习，导出无需在线LR；固定合法四步采样和BF16 attention。现有LR-only训练不能代替这条路线。普通QAD本身不称贡献，但不再要求先有novel claim才做强基线；真实独立视频、延迟/显存、训练与导出落差决定后续研究。简化重复gate/多重审计，保留必要合同、梯度、真实native和失败记录。实现就绪度与流程复盘已落盘；本轮尚未实现训练器或训练。


## D035 — E020开始实际主权重QAD，简化正确性流程

2026-10-03。上一轮为progress，报告018与LSE实测定位完成；当前顶会目标仍active，无新claim。E020新增薄QAD/无LR导出模块与64步训练器，固定16训练/8开发验证缓存，现场BF16重算24目标，原rCM完整31,200 tokens、300主矩阵共1,391,984,640参数；其余BF16，无旧INT4 norm/smooth/LR。ordinary identity STE与legacy-Wan NVFP4配方明确，不称新方法。

首次smoke在单kernel decoder的E4 masked-load整数0转换处编译失败，0完整DiT/0更新；保存原源码和失败。修为浮点0，同时按代码review让FLASH_ATTENTION上下文覆盖checkpoint backward，原45分钟deadline保留。第二次真实GPU检查通过：BF16包与旧packer相同、解码与独立LUT一致、FP32 master未预舍入BF16、STE实值及梯度正确、原生FP4 GEMM实际执行。小矩阵native/QDQ NMSE9.09e-6仅记录累加差异。

训练已实际启动，300主矩阵第一步梯度全非零，optimizer实际改变master；当前每步约7.8秒、峰值25.42GiB。初始native开发验证NMSE约.137708，不是视频质量分数。按原64步完成并保存学习曲线与初末packed模型后再看真实导出收益，不中途凭loss挑checkpoint或修改lr。独立生成脚本在准备，prompt/noise不进入这24缓存，但不把同seed数值是否在别处使用等同数据泄漏。后续完整四步/77帧/16fps保持，仍以真实质量—成本为目标。


## D036 — E020/E021闭环完成，扩充QAD数据与优化而非转回算子小审计

2026-10-03。本轮是progress，顶会目标active、无阻塞、仍无可投稿贡献。真实300主矩阵64Adam更新完成，全部300梯度/更新及packed代码变化核实；训练QDQ pooled NMSE−34.14%，两开发prompt native pooled NMSE+7.77%，step0两例恶化、其余六state改善。固定32检查点QDQ更好仅探索性，本轮仍按原64终点生成。独立fresh-model部署四臂各2warm+5repeat，BF16/SVD/plain/QAD中位1.789/1.990/1.659/1.658s；无等质量速度结论，旧SVD recipe差异不归LR一因。

E021固定四prompt共用一个实际noise序列，16完整视频/64DiT/14400FP4/3840SDPA全部通过输入、final和媒体验证。MJ全16条、dense AMT/RAFT与全帧DINO已完成，CPU汇总通过。QAD比plain的MJ总分逐例−.0524/+.2093/+1.0167/−.2398，campus异常主导平均上升；该例BF16也严重绿条纹，保留不剔。QAD AMT更高但RAFT阈值通过1/4（BF16 4/4），不把平滑度作质量收益；部分binary贴阈值且campus失真flow巨大。全部contactsheet固定5帧已看，不宣称完整运动观看。训练文本当前映射未提蓝球，旧缓存SHA和teacher exact但无完整文本/seed来源，不能由图断言特定样本复制。

MJ首两次启动在推理前缺decord/av失败，原记录和30分钟deadline保留；已有MJ入口虽symlink同binary但prefix不同，切正确环境后14秒完成，eager attention、实际8帧indices已记录。tokenizer仅复制已有资产补缓存code，词表编码核验，不改原模型。报告019/020完成，E020/E021所有GPU退出。

下一步E022从原BF16 fresh开始，以现场32新trainprompt×2独立noise×4步=256状态、4新dev×2×4=32状态训练；每prompt/replica不同seed。固定lr3e-6、128updates、每次四timestep各一microbatch累积，512micro共两遍数据；开发native0/32/64/128预设最小选择，保留固定128终点，再独立视频验证。一次更充分baseline同时改数据/lr/accum，不做因果单归属或novel claim；不因E020负面放弃主权重训练。实现准备中，GPU由root核查后启动。

E022实际启动更新：collector PID1768590/GPU0已完成36个embedding和72套独立noise并进入288次teacher前向，监督deadline1790963501.632387。trainer完成CPU采样和source review，固定SHA920eff43...，tmux research-E022-queue 等采集成功/进程退出后在现查空闲GPU5启动，90分钟训练deadline独立起算；当前尚无新optimizer更新，不把队列当训练完成。

E022采集终态：545.43秒、288/288 BF16调用，实际17,280SDPA/0FP4，256train/32dev状态、72唯一初始noise；collector PID1768590已退出。监督队列已接续训练PID1890988/GPU5，deadline 1790968253.118431，正在初始验证。新8prompt×2seed视频测试manifest也在看到训练输出前固定，精确文本/本地历史排除，不代表全球未见或全VBench分布。

E022首轮实际训练检查通过：初始QDQ/native开发NMSE .153120/.154799，已3个optimizer更新/13个micro backward，首步300矩阵梯度全有限非零、master确实变化；当前峰值28.270GiB。原128更新继续运行，无中途调参或新科学结论。


## D037 — 上轮是实质进展；在训练期间完成固定测试对照

2026-10-03。上一目标轮完成E020/E021闭环、报告019/020、E022现场288teacher及新主W训练启动，分类progress。当前实查PID1890988训练和PID1890981监督均live、GPU5实际99%利用，5个optimizer更新/22micro正在推进；原deadline不变。目标active，无阻塞、无投稿贡献。当前不以训练耗时转向新小审计，而在GPU0准备/执行已固定8prompt×2seed的三baseline48视频；QAD selected16视频待最终dev选中的真实权重产生后执行。训练/测试数据、选择规则和已有源不变，新的分阶段执行安排独立记录。并行核查最直接QAD成熟配方，避免将训练不足误当新机制。

## D038 — E022中期改善成立；排除一个共同生成失真的简单解释

2026-10-03。本轮有实质progress：E022完整0/32/64开发验证已保存并独立CPU重算，native pooled .154799→.123383→.104920，四prompt和四timestep汇总逐点下降。32→64的32条开发记录全部改善，不能把0→32也说成所有row单调。固定128更新、数据、学习率和选择规则继续原样执行，不因观察中期结果改选候选。

并行三baseline48视频完整结束：192DiT/38400FP4/11520SDPA，672.86秒；16独立初态、全媒体77帧和文件一致性通过。MJ22.75秒、AMT/RAFT/DINO362.93秒，48条全部独立复算，按两seed→八prompt等权。root查看全部8张replica0五帧sheet，多个BF16也有明显青绿竖条，故当前分数不支持普遍量化质量排序。

官方本地README要求FP32 VAE、旧入口采用BF16，先对固定BF16球场坏例197_r0与阅读对照133_r0的已保存latent做两dtype各一次解码；GPU2空闲检查时被其他任务占用，因此保留GPU2准备源/CPUcheck，另建仅GPU0绑定不同的版本，确认GPU0持续空闲后执行。4次VAE/0次DiT、52.82秒、13.20GiB；两个BF16 MP4 SHA精确复现，FP32仍有球场竖条，像素MAE .00484/.00210。root已看两张固定对照；不把VAE低精度认作已证根因，不批量重解码，不删旧媒体或坏例。下一步只读核对rCM官方采样/转换/文本条件；训练主线不因该诊断暂停。

ModelOpt强配方核查固定官方commit e68eb44并留小源码快照，澄清默认固定outer global与dynamic-group的区别、legacy舍入差别；E023 W-only固定global仅计划，未GPU执行，普通基线不是创新。报告021和索引已更新。目标仍active、无阻塞、无投稿核心贡献。

## D039 — 完成E022而不是停在中期；用真实条件比较原架构

2026-10-03。上一目标轮分类progress：0/32/64独立开发复算、48baseline完整评价、四次VAE诊断及报告021均已完成。本轮重新读取实际/proc/1890988，确认GPU5原128次训练仍运行（开启本轮时94次更新/376micro），原deadline不变，目标active无阻塞。继续等待其实际终态并生成预定selected16视频，不根据中期结果改训练或选模。

下一项对照直接固定E022球场197_r0、阅读133_r0的真实已保存条件与噪声，用原rCM架构/原.pt做两条四步自由轨迹，共8DiT。原UMT5本地只有不完整文件，故采用两路径共同的已保存真实embedding，明确隔离transformer运行路径，不能声称原UMT5端到端验证；不下载/重建T5、不因此阻塞。原架构源码显式要求time/head阶段FP32，而公开脚本整体castBF16在当前环境存在dtype矛盾；新入口按源码要求保留这些阶段及仿射LayerNorm为FP32，其余BF16，限定BF16 FLASH SDPA，明确这一受控兼容政策，不能将差异单归转换或称原发布二进制复现。统一FP32 VAE解码两原始新final与两历史convertedfinal，共4decode，不再随机小shape检验、不设跨实现字节或任意误差质量gate。准备后root核查空闲GPU0，15分钟/60GiB；尚未启动。E022本身及其原媒体不改。

上述参考对照现已完成：GPU0，worker2855972退出、returncode0，8DiT/480SDPA/4FP32VAE，worker95.99秒、监督101.44秒、峰值13.21GiB。独立CPU重放8步/状态链/实际BF16输入一致、20文件身份/4段共308帧完整解码通过。原架构相对converted的final NMSE球场.112151、阅读.209927，不能称输出等价；root看两张五帧对照，球场仍有同样强竖条，阅读人物保持正常主体但构图略变。由此不支持转换路径独有错误的解释，也不证明原UMT5或原发布二进制已完整复现。停止扩大当前排错网格，继续E022预定selected视频；不把共同BF16失败写成量化机制。

## D040 — E022完整视频未确认收益，转官方产品基线

2026-10-03。本轮有实质progress：原128次QAD训练正常结束，768逻辑DiT，约75分钟、28.27GiB；native0/32/64/128=.154799/.123383/.104920/.112768，按预定开发规则选择64，保留128。不因末次回升重设规则或选视频最优候选。selected16视频/64DiT/19200FP4/3840SDPA完成，190.97秒；MJ12.52秒、时序125.85秒，所有GPU worker退出。64视频完整汇总独立CPU核验，原始结果与已执行源未改。

开发NMSE改善32.22%但MJ总分plain/QAD=.071057/.071926，均差+.000868由两文本正向与六文本负向抵消，4/8文本两个seed异号。RAFT动态通过6→11/16，不复现旧小样本运动减少；flow增量91%来自科学馆/庭院/球场失真例，不将其解释为正确运动改善。QAD DINO .88815低于plain .92650，11/16轨迹下降。root已看全部8组固定五帧四臂图；共同BF16异常仍未定位。报告022及逐seed配对图完成，没有新贡献或质量等价结论。

决定不自动启动E023普通W-global训练，不围绕共同失败扩loss/诊断网格。E024准备官方FastWan-QAD强产品参照：固定FastVideo8444c089、HF621c6aeb、TAEHV011dfc211，SM120官方实现，独立torch2.12/cu130 env。systems负责环境，frontier负责固定资产，audit负责复用官方入口的薄runner，root负责协议、结果及GPU调度。源码只读核到官方当前推理3步FlowUniPC/shift3与训练文档DMD[1000,757,522]有差异，实际配置明确记录，不能静默换preset。先官方单条功能生成，再固定原16文本/seed产品评价；所有配方差异和成本单列。目标active，无阻塞，仍无可投稿核心贡献。

## D041 — 官方产品实际落地；保留采样合同疑点，不包装成新机制

2026-10-03。本轮继续progress：E024独立环境、固定公开资产下载与SHA复用、官方小内核前向、单条完整smoke、固定16视频和MJ/AMT/RAFT/DINO全部结束，CPU独立汇总通过。没有移植自研算子冒充官方。正式48DiT/14400NVFP4 GEMM/1440FP4 self-attention/1440dense cross-attention，完整81帧/480×832/16fps。实际FlowUniPC/shift3/t=[999,857,599]，未换成训练文档的DMD步骤。

MJ总分.322401高于本地rCM各臂；相对plain 6/8文本提高、相对BF16仅4/8提高。RAFT仅shuttle两seed动态，DINO.979017、AMT.994420不能当高质量；root看全部8题r0五帧，多例内容偏离提示。所有失败样例保留，未按画面筛选或重抽seed。不同权重、decoder、采样及帧数的完整产品参照不支持量化独因果或新方法claim。

首smoke包含首次CUTLASS编译，正式suite在缓存就绪后112.69秒含15.35秒构建；pipeline中位4.916秒、TAEHV中位.230秒仅diagnostic（含hook、stage同步、轨迹保存，pipeline还含TE），不写为compile稳态速度。GPU和下载进程均已退出，报告023完成。首次下载两轮0B与小kernel smoke报告字段失败原样保留；只修环境/报告，正式模型生成及评价无失败。

下一项只读核查权重发布时的官方采样合同，优先查模型卡docker f889e6b与原官方推理实现，避免在明显接线疑点尚未解决时将静态或语义失败解释为NVFP4机制。若有官方明确依据，另立采样变体直接验证，保留当前结果；不无依据盲扫scheduler、不换测试文本。E023普通尺度训练仍不启动；完整VAE控制与无观测的compile性能尚未执行。顶会目标active，无外部阻塞，仍无可投稿核心贡献。

## D042 — 恢复目标，选择一次零DiT的decoder对照

2026-10-03。目标服务已确认active。旧E022/E024均闭环，停止扩大普通QAD及precision网格。先收尾官方采样发布路由核查；并固定E024全部16 final latents 做完整Wan VAE FP32对照（E025），不按既有结果筛样，GPU解码600秒。与TAEHV FP16比较是decoder stack+dtype单独介入，不是量化单因素。结果相同则结束VAE排错，不追加网格；结果显著恢复则先修正对E024失败的解释。系统agent实现，审计agent复用评价，root决策与查看结果。仍无已成立的可投稿贡献。

## D043 — 解码与采样配置线收尾，做一次已有packet的算子干预

E025全部16对完整完成，动态判定16/16不变，MJ均差+.010158由航天器主导、仅3/8文本提高，原主要语义偏差仍在。按计划结束VAE排错，不能据此将剩余问题直接归因量化。首发官方路由同UniPC，无支持改DMD的权重绑定。转E026：旧S007块均值校正物化二次张量，附加query行可表达但复用FP4 K新增μ(Khat−Kc)项。复用E018已存真实H3 p36/s14/block0的全部56head/22539有效token，固定前六packet，仅替换第七correction，2次原生attention/0DiT。先量误差下限，不直接开发内核、不将既有代数称新颖。GPU1，600秒。

## D044 — 固定均值复用K4的代价成立，保留成本问题而非宣布融合失败

E026真实两次attention与独立CPU复算均完成。block/oracle/global pooled NMSE .002419689/.003314582/.003561302；固定μ×K4损失原block相对global优势78.39%，但仍保留部分优势。只能弱化“保持μ直接复用K4仍保持原block精度”的假设，不是允许重优化μ后的数学下界。停止追加单/双段μ编码小网格，下一步先比较保留BF16 K输入的在线correction与已有query-chunking成本；若接口改动需重写调度，先用强baseline和独立M16-stripe实测决定预算，后者不冒充融合速度。尚未实现/启动该下一实验，目标active，无阻塞，无可投稿贡献。

## D045 — 测保真校正的成本，避免先写大内核再寻找收益

2026-10-03。E025/E026完成是实质进展。E027固定同一真实H3层，比较完整物化和Qchunk4096，均包括合法预处理与原生consumer；另测BF16 M16-stripe真实成本。现有K4/SFK/correction共用3stage TMA，在线BF16需额外staging/同步，不能仅换lambda。先测两项实际成本，不把微基准拼成虚构融合时延；GPU0顺序执行、每项600秒。未开发通用后端，无新视频/DiT。

## D046 — 已知分块基线足够强，停止朴素在线M16调度投入

E027完整流程31.780→33.975ms（+6.91%）、allocated峰值3.518→2.601GiB（−26.09%），四输出对E026原block一致；独立CPU复算完成。BF16 stripe独立14.815ms，不与whole加减，peak含reference。当前不为朴素在线重写调度；并不证明所有fusion无价值。E028转CPU低成本表示诊断：c_i=global+A_iB同时改Q中心和恢复项，三层同状态，Euclidean/score/K4-error三几何谱。后者由(μ−c)(Khat−Kc)+ΔεQ Khat解释，不能只用score谱选中心；谱不代表Q4质量或固定basis泛化。E019继续parked（未执行、非科学阴性），因已知修复与缺新约束，不复活成论文方向。

## D047 — 谱集中支持一次原生同步中心干预，不以低秩本身立claim

E028 CPU13.104秒、504项谱完整；r16 K4-error尾能量0.89%/8.07%/3.63%，K-score更集中但不是实际误差代理。选择固定r16的Euclidean/K-score/K4-error三种同样本中心和block/global端点，共15原生attention（E029）；同时改Q量化及correction，不再固定原Qpacket后单独近似correction。当前无新kernel/视频质量/固定basis泛化claim，强近邻已记录。

## D048 — 同样本压缩有效，先测试固定basis而非宣称新几何

E029三层5臂15native完成65.57秒、0DiT，rank16相对freeblock误差+1.58%–3.29%，保留global→block优势91.66%–96.66%；三geometry无稳定胜者。不立K4-error几何优越claim。E030只从已有另文本p30/s14捕获一次DiT均值、固定Euclidean rank16 basis迁移p36/s14三层，3新attention；无新视频、不扫rank。consumer17行读取/组合的成本尚未知，现有每tile只读一行，不能按177→17容量比预测加速。

D048执行前补充：E030尚未GPU运行，加入factorable_fp32第二臂，以同固定B的FP32 A/center与T0+A(Tb)同步Q中心化，避免将E029 BF16中心舍入后的非严格rank17当可直接分解。三层共6probe，1DiT不变；v1计划保留。

## D049 — 固定basis转移明显回落，暂不投入完整consumer

E030完整1DiT+6probe及独立CPU复算均完成；单p30 Euclidean basis→p36保留global→block优势62.02%–75.60%，head中位50.24%–71.17%，两head比global差。FP32可分解合同只造成小幅正负数值变化，当前主要瓶颈是表示迁移。先只读核查真实RoPE/多模态/flatten结构能否解释低秩及定义有根据的selector；不盲扩rank/训练/视频网格，不将低秩结合律写成创新。全部GPU已退出，目标active无阻塞。

D049只读核查完成：实际RoPE三轴96维＋32未旋转维度，视频帧内T不变、文本/音频多轴不变，不能推出普遍rank16低频selector。源码笔记已保存；不将该直觉作为观测解释。后续优先有限比较样本自适应中心构造成本，而非直接写consumer或扩训练/视频。

## D050 — 先测经典自适应构造的真实成本，避免把oracle当免费

上一turn为progress，E027–E030均闭环。E031固定已有三层、rank16、GPU fullSVD/range0/range1，Ω固定seed20261031；9native/0DiT。分别测μ→basis/center和raw Q/K→Qcenter/T17预处理，包括必要复制/QR，但不含未来consumer组合。global/block作同scope预处理基线；不通过加减组件时间宣称端到端收益。经典方法只作为成本/精度对照，600秒GPU0，未执行。

## D051 — 自适应PCA准备仍贵，先用共享中心表检验消费表示

E031九原生输出及24个计时范围complete：fullSVD构造约211ms，range0/1约1.84/3.61ms；rawprep10.35/12.15ms比block8.43ms慢，不等于优化下界。选择E032固定K16/6次Lloyd的one-hot中心强基线，未来只挑一条T行，不先写多行producer。3native/0DiT、600秒GPU0；经典kmeans/centroid共享有强近邻，不称新算法。

## D052 — 共享表准备成本接近block，封顶一个真实consumer验证

E032三个原生输出及独立CPU复算完成，K16全部168head优于global，保留block优势55.97%–71.74%；prep8.49≈block8.43ms，prep峰值−25.7%，不是完整消费结果。选择E033最小DS选行映射，保留原TMA/barrier，独立JIT；10新旧正确性调用（允许1 BF16步而非bitexact），再3层五臂完整路径含Qchunk4096/8192。只一个原型，900秒含冷构建，不扩参数网格；若实际前沿支配则停。

## D053 — consumer点未被支配，但先排除简单粒度基线

E033全部478native与5paired control完成；候选31.5ms/2.755GiB，对full31.7ms/3.518GiB，精度仍损失。不能把无聚类中间粒度的global/block两端对照当充分证据。E034先固定K16连续Qtile均匀分组，真实paddedQ BF16中心，四臂全路径156native；无新kernel、无视频/参数网格，先问聚类是否必要。

## D054 — 简单粒度已解释大部分共享中心收益，停止聚类网格

E034完成156native/0DiT、44.38秒。coarse16 29.64–29.70ms，对codebook31.47–31.53ms，峰值同约2.755GiB；NMSE只高1.00%–1.76%，取得codebook相对global改善量93.1%–97.3%。聚类未严格被支配，但当前创新与效益不足，不扩K/迭代/几何。保留工程baseline，下一只读核查既有稀疏工作图N2入口，不写新kernel/不预注册宽泛组合claim。独立CPU复核进行中。

D054复核完成：E034独立CPU汇总46.31秒，结论一致。coarse取得codebook改善量93.1%–97.3%，但后者三层pooled仍好，不能写成严格支配。暂停的是当前聚类方法投入而非用户研究目标；goal保持active，N2只读入口核查进行中。

## D055 — 用真实同输入投影配对先测路由预算，不为假设先建稀疏系统

N2只读核查找到既有cuDNN SM120 variable-count/tail consumer，但原H3 router fixedtopk。E035先1BF16 DiT+3局部SVD-native projection，保持raw input/norm/RoPE；CPU显式contiguous128、prefixdense、CDF.9/min4诊断。无稀疏kernel调用/新训练，不冒称现成产品或新方法；只有路由事实值得解释才消费两个mask。

D055 GPU前v2修订：进一步源码证据cuDNN支持中间每块validprefix，因此采用现成H3 VSA3D128几何（prefix11/video200、211tiles），不再简化contiguous。v1完整保留未执行，不添加实验臂；输入/投影/CDF阈值不变。


## D056 — 路由边身份变化未伴随大幅预算变化，结束当前膨胀故事

E035完整capture与CPUrouter、独立六artifact复算完成。blocks0/24/48物理预算+.371/.904/.492%，有效token对+.370/.884/.468%；63.7%–80.5%行换边，但每video query平均净增.305–.672块，中位0，head正负均有。不把mask变化当时延，不以未运行consumer证明无影响。停止当前系统故事及阈值/样例网格；完整报告031。下一仅查N1：FP4 QKV跨卡之前的Q-center/K校正生产依赖，以E034粗中心现有consumer为参考，扣除globalQ、K4重建与原BF16传输基线。无新kernel/多卡任务/新claim。


## D057 — 在真实两卡边界测一次校正通信取舍

N1只读依赖核查完成：BF16 ownership天然本地，压缩先行则需要中心／统一K均值／原K校正。可用源侧T16替代远端K16，代数普通但字节账本有可测差异；不是强制传完整T177。E036固定已有block0、三臂相同coarse16、GPU0/1同NUMA，包含逆output A2A的完整路径，78native/0DiT/900秒；先CPU几何/transport检查。没有新方法claim或完整视频质量依据。


## D058 — 同合同通信收益成立，先面对global-Q强对照

E036三臂78native/0DiT和独立复算complete，两卡P2P/CUMEM，完整边界含逆output A2A。BF16/mixed/FP4+T16=28.633/23.214/21.553ms；后者比mixed快7.15%、allocated略增。三臂输出均与E034 coarse16相同，分布式Kmean一极小元素变化未改变输出。不是新方法/整模/质量claim。下一仅补现成global-Q/T1的同路径成本与原精度合同，先判断中心粒度是否只是常规精度—通信取舍；不扩网格或直接整模集成。全部worker退出，goal active。


## D059 — 同期补global-Q/T1，避免把粗中心通信当唯一强实现

现场复核E035/E036完成、workers已退出，上轮属于progress。E037固定同block0及GPU0/1，T16原样复用和原global-Q/T1两臂完整token→head→token路径，3warm+10 AB/BA，52native/0DiT/900秒。global合并Q/K sums为一次AllReduce，按原NP/N分母BF16中心；不同数值合同并列误差与成本，不要求byte一致。先CPUcheck，无新kernel/视频/参数网格。


## D060 — global强对照完成，下一投入转向完整质量

E037 global比coarse16快6.42%而局部NMSE高24.68%，两臂各自复现既有输出。没有等质量支配证据。保留E036工程结果，暂停中心/通信网格；预固定E038四动作×两seed×同native权重三attention臂共24视频，直接检验coarse实际语义/时序需求。若无清晰分离或coarse退化，停止扩大中心表示路线，不将局部阴性泛化为科学不可能。


## D061 — E038完成，停止扩大中心表示路线

四动作×两seed×三臂24视频全部完成并独立复核。coarse−global MJ均值−.012914，3/8seed改善；叠衣与汽车双seed均下降，大象异号。DINO改善不替代主质量判断；拍手中两FP4臂的手部纹理异常未被coarse稳定消除。按预定资源决策停止新增中心算法/数量/通信网格，保留E036/E037同合同工程事实；不声称所有多中心都无用、质量等价或总体统计显著。

## D062 — 只保留一个新的输出接口问题，尚未启动实验

既有逆A2A仍回传BF16 O，而原native SVD out_proj主支用Q4(O/smooth)、LR支用原BF16输入。下一先面对源侧out_proj＋ReduceScatter强基线，列清activation outer-global统计域和分片LR归约舍入，再判断packed输入加小LR状态是否有实际价值。标准分布式matmul与低秩side information本身不是新颖性；当前只有机制/近邻初查，没有新方法或新实验结论。见E038_afterward_candidate.md。


## D063 — E039按真实完整输入检验四臂输出接口

上一轮E038为progress。固定BF16 inverse A2A+native、FP8 inverse A2A+native、源行并行main/down32合并BF16 RS后一次up、native packet+down32 side四臂。第一强控制不重复up，不默认昂贵FP32大通信；输入从E035完整nativeQKV经原BF16 helper另准备22592行O，保留53真实modelpad参加global，不能用E034有效O补零替代。prepare2SDPA/0DiT complete；四臂协议已冻结、benchmark尚未执行。若普通强替代已更好则停止第四臂，不把普通矩阵重排称新方法。


## D064 — E039有完整边界收益，先检验低秩信息而非扩网格

四臂104边界/130native/0DiT complete并独立核验。FP4-side 4.897ms对BF16回传6.348ms，valid NMSE5.9324e−8；自然row8.762ms、普通FP811.090ms。保留当前实现优势，不能称胜过成熟融合FP8或新方法。E040固定同packet/main，只改LR信息来源，总2native、0DiT/attention/collective，检验decode-X廉价替代。报告035；goal active无阻塞。


## D065 — E040确认局部信息损失，但C006不足独立主线

同packet/main下side与decoded-X输出valid NMSE5.9324e−8/1.83369e−6，后者绝对误差仍小；E039重放和globals零漂移。2native/0attention/DiT/通信、7.38秒launcher及独立CPU6.67秒complete，worker1369659退出。它确认既有SVDQuant高精度LR信息的局部价值，不建立视频必要性或新机制。按用户避免简单1+1的要求及现有ABI/文献覆盖，停止C006作为独立论文主线，保留E039代码与工程收益，不直接扩多卡整模。下一回到证据驱动的问题筛选；目标active无阻塞。报告036。


## D066 — 全帧可辨性读出不支持统一动作变慢

E041两seed×三臂共744原生帧/48页完成，两agent分别逐seed检查，非盲评/非实时播放；全部物理接触不确定。低位前段仍快速往复，后段重影使完整周期只能给下界或unknown。停止把16/16较低RAFT均值当实际动作减少；也不把原生注意力一般动态下降重新包装成新发现，Attn-QAT直接覆盖。没有新增GPU/生成。下一只做具体decoder上下文/音频事件读出入口及近邻核查，不直接设计loss/新量化配方。报告037；goal active无阻塞。


## D067 — 固定音频读出止于事件可识别性，不制造同步结论

E042六份既有PCM的固定RMS/高频能量、hop2ms读出2.540秒complete，0模型/0GPU；root实际看两full/两zoom_v2并审时间轴/视觉区间转换。基线仍有多峰/宽能量团，未听音或独立语义事件标注；峰不能自动当掌面接触，未知不能填成失步。按入口规则停止当前AV同步归因，不做阈值/offset/新loss/保护层网格。P006论文问题parked，下一有限重审已有停线依据与真实未解释反例。报告038；总体goal active，无外部阻塞。


## D068 — 停线复审不重启旧候选，建立原版Wan参照

E041/E042上轮为progress。有限审查区分机制反证、未执行路径与预算停止；没有符合重访条件的具体反例。标准数学/现成ABI不是系统论文的自动否决条件，但C006仍缺成熟fusion与完整路径需求证据，现不重启。E022的共同rCM异常/E024的另一蒸馏产品不能替代原版teacher；E043固定原版非蒸馏Wan的8条完整生成，原版shards官方SHA已核，无旧PTQ模型替换。先建立实际可读参照，再决定同模型native配对；不把修实验底座当论文贡献。


## D069 — 原版参照允许完整配对，基线动作缺陷不隐去

E043全部八条生成与评价complete，800DiT/400scheduler/8decode/8TE，独立核验complete。全部固定九帧未见旧共同条纹，拍手/喷水可读；fold语义含混、car r1明显形变。按预定协议保留全部八个case和实际FP32初始噪声，不按质量后选；可继续原版非rCM的plain NVFP4/SVDQuant完整配对，但不把原版当物理动作标签。当前只读确认已有checkpoint与native loader入口，尚未冻结或启动量化实验。报告039；无可投稿核心贡献，goal active无阻塞。


## D070 — 保留更强SVD基线，残余细节损伤先作同状态诊断

E044正式v2两臂16视频/1600DiT/480000native与全量评价complete，独立CPUsummary complete。SVD−plain MJ均值+.35378、7/8为正；SVD−BF16均值−.06625、2/8为正，fineness8/8下降，DINO7/8下降。保留整个SVD配方作为强基线，不把已有LR收益当创新；较高AMT不证明运动或质量恢复。全部固定九帧可见色块/形变，但不归因decoder、时间积累或四帧周期。下一以真实同teacher状态前向/采样更新形成有限干预，先核问题再执行，不用自由分叉终态当局部噪声、不直接训练或扫参。一般decoder抗扰动和causal不均衡已有近邻，当前尚无residual claim。报告040；goal active无外部阻塞。


## D071 — 末步充分性成立，停止首位置传播解释并面对已有强控制

E045两例同teacher状态，实际native末步+四角解码完成，全部八固定九帧显示完整/rest-only在手/脸/衣物产生碎片，first-only后帧近teacher。postclamp后帧MSE full .0051593/.0063932、first-only .0000373/.0000137、rest-only .0051282/.0063707；输入后latent CFG误差本已较首片更高，不能单靠输出不均宣称decoder增益。当前末步降一阶，不是多步历史放大。停止基于首位置蔓延的解释和decoder训练；DSAQuant已直接覆盖CFG分支mismatch/晚步高频伪影，PTQD覆盖晚步提精度，不立新claim。下一只先面对完整native轨迹最后一步BF16的简单强控制；单末步足够致损不等于它能消除此前量化损伤。

首次两teacher完整后diagnostic signature stride0失败，actualcapsule保留；v2只重放末步，B与E043独立final零差，原E045final未存/shadow unavailable明确。总204DiT/1200native/104scheduler/8decode，所有GPU进程已退出，独立CPU汇总complete；报告041。目标active，无外部阻塞，无可投稿核心贡献。


## D072 — 保留末步强控制，先补匹配校准而非把残余当新机制

E046八例16媒体/816DiT/240000native与全量评价独立汇总complete，全部GPU进程退出。bf16_last相对native total/alignment/fineness/AMT8/8提高，总分均值+.077025；coherence/DINO各7/8提高，RAFT全不变。对原BF16 total/fineness/coherence各6/8仍低，固定九帧的手/衣碎片与车几何缺陷保留，不能据总体MJ略高称恢复。备用原BF16实际增加2.650625GiB allocated，当前双驻留不是等质量部署优化；不扩末步数/参数网格。

基线只读审查新核旧calib metadata真实33帧latent[1,16,9,60,104]、steps0/25/49对应t999/749/57，直接不同于当前81帧shift8末t146，与旧collector默认shift3一致。rank32/group16/E2M1+E4M3本身不是已证错误；现成r64要么rCM祖先、要么旧INT4 fake/group64，不能替换。下一优先补原版81帧/CFG6/UniPC50/shift8的匹配校准强基线，再谈残余机制；新训练/生成尚未启动。CFG common/difference变换被GCBT直接覆盖并已有Nunchaku成本结果，停止该新方法候选。patch相位只有只读入口、未测量/未立claim，先不投入GPU。报告042；目标active无阻塞。


## D073 — 匹配校准仍有主要失真，不扩大校准网格

E047完整PTQ约5小时20分完成；E048八例生成/评价/独立CPU归约全部complete，800DiT/240000native/48000SDPA/400scheduler/8decode，无新训练loss。matched相对old MJ均值+.02012、5/8提高，但相对原BF16 6/8总分、7/8细节更低。拍手两seed细节与连贯性均下降，汽车r1总分.52270→.09538、RAFTtrue→false而DINO上升，明确不能用稳定性替代语义质量。全部九帧观察保留手/衣碎片、汽车形变；帧数与schedule共同改变，八例不是独立测试，不作单因素归因或统计泛化。

保留匹配原版SVD作为强基线，同时保留旧SVD和末步BF16控制，不按提示挑最好结果。停止把匹配校准当充分修复方案，不扩rank/步数/校准网格。下一完成正在下载的官方Wan14B资产及完整原版BF16入口，固定协议后才开始新的同模型对照；不会从1.3B直接声称14B退化或研究有效性。patch/decoder/CFG宽泛机制不自动重启，新方法仍须具体残余问题与近邻区分。报告044；目标active，无可投稿核心贡献。


## D074 — 14B原版参照可用于量化对照，动作局限保留

E049八例生成/评分/独立CPU汇总全部complete：800DiT/64000SDPA/400scheduler/8decode/0TE/0native；生成1897.129秒、评价70.902秒，所有进程退出。root先完成八例九帧观察再读取分数，非盲评或完整动作标注。14B MJ均值.851667、细节.072021、coherence.650879，RAFT8/8；相对1.3B参考平均更高但DINO更低，模型/CFG/schedule共同改变，不能给尺寸或量化因果结论。无共同大碎片不等于完美语义：折叠、转弯r0、喷水来源仍有歧义，保留全部样例，不设无限语义gate。

进入本模型量化基线：E050冻结原64records/14提示政策，以CFG5/shift3完整14B轨迹采集，复用身份相同的E047实际TE embedding；不迁移1.3中间latent或checkpoint。先采集、再首block完整样本/搜索预算资源测量，不盲目承诺全模PTQ耗时。Plain逐层pack为部署适配而非方法，不能替代SVD强基线；CPU结构已过但尚无真实14B native结果。报告046，当前无可投稿贡献，目标active。


## D075 — 14B plain质量差异集中但不止单一总分，先完成匹配SVD资源依据

E051八例原生生成/评价/独立汇总全部complete，MJ .851667→.755266（4升4降），细节/coherence各3升5降；拍手r0占总净降95.8%，双seed细节/coherence都退化。叠衣/汽车平均总分略升，喷水r0总分略升却RAFT1→0。root先八例观察、再读分，指标独立复算一致；不得称全崩坏、删除离群值、按提示择优或把八例当独立泛化集。完整pipeline时长和−21.60%和allocated40.035→21.197GiB保留，但不是质量匹配的成熟serving收益，加载/转换已见峰27.354GiB。

E050匹配14B完整14轨迹/64cache及CPU独立汇总已完成，全部原worker退出。E052原监督已GPU4实际启动完整预算block0资源pilot，未整模PTQ。保持64/b4/g10/r32/LR100原早停及全部八例；依据真实成本决定整模PTQ预算，然后形成同模型SVD强基线，不因plain部分样例较好而跳过SVD比较，也不因一个最差样例另开校准网格。两项薄适配CPU/meta已准备，partial仍不是完整checkpoint。当前无新的机制claim/等质量优化/可投稿贡献，目标active。


## D076 — MJ总分仅作辅助证据，纠正评价对象与输入覆盖的误读

2026-10-04，用户质疑MJ可靠性后，核对原论文、E051实际scorer及原始输出。确为MJ-VIDEO-2B；总分含全部五方面，安全/公平性只是不单列汇报，未从总分剔除。81帧只抽0/10/…/70，整幅非等比缩放448×448；论文总体偏好准确率约64–70%，不是量化伪影验证。E051喷水r0的MJ微升与RAFT下降均属实，不能互为真值。

保留历史数字并补充报告048解释；不把分差解释成画质百分比，不以MJ单项涨跌接受/淘汰研究方向或声称等质量。下一关键质量判断须针对具体假设，使用独立样本、完整视频盲法成对人工评价与互补读出，允许平局/无法判断，且明确尚未执行。当前八例仍为诊断集。详见 `01_literature/evaluation_map.md`；本轮只读核查和文档修正，未重评分/启动GPU，调度goal返回paused且未改变。


## D077 — 为汇报整理阶段证据与两个有限候选，不重新立项

2026-10-04，按用户要求形成报告049：原生部署、attention、QAD、通信、末步干预、匹配校准及14B对照均按动机→实验→结论梳理，另保留早期阴性筛查。速度收益严格区分DiT、局部边界和带诊断pipeline；不把MJ或抽帧观察当独立质量真值。优先讨论输出patch几何/内容耦合的机制问题，但没有测得量化特有结构；普通head扰动与先前工作是必须面对的控制。通信表示作为有实测收益、创新性偏弱的系统备选，须面对成熟融合和完整路径，不改变C006 stopped状态。没有注册active新claim、没有启动GPU或完整14B PTQ；仅更新汇报口径。


## D078 — 恢复研究，以H3为主，工程优化不充作创新

2026-10-04用户恢复目标，进一步指定MiniMax-H3主要实验、Wan1.3B只快速验证，并提醒历史伪量化/粗糙kernel是快速实验工具，普通修复或融合不构成创新。据此不启动E053；本轮已准备E054融合强对照但未执行，停于预备，不再围绕旧C006扩工程。转到E055：利用已有H3 CFG1三teacher状态的BF16/plain/SVD同状态输出，比较真实残差相位统计与同一head白噪声解析null，仅决定是否值得更强机制控制。H3实际head也是2×2，但24通道且输出行序与Wan不同，不能迁移CFG放大结论。当前无active新方法claim。用户恢复授权有效；goal工具仍返回paused且没有resume操作，本轮继续推进实际工作并如实区分调度状态。


## D079 — H3相位诊断不足以支持特殊patch边界机制，停止该候选

E055分析complete0.161秒，3同状态×2量化臂、0GPU/0模型forward。独立复算逐37×24×4数组，横/纵/对角phase差六组及逐time通道均值全部低于iid真实head模板；普通head本身足以产生比观察更明显的patch内外差。白噪声null并未解释实际跨patch空间相关，但有色/各向异性普通扰动未排除，不能把这一失配升级为新机制。停止因phase统计追加hidden捕获/decoder干预/loss；不以“还没排除”延长路线。报告050已按动机→实验→结果/决定记录。E054未执行；C006不重启；H3主线与工程不当创新的用户约束继续生效。


## D080 — H3相邻误差保留方向相关，进入有限传播判别

2026-10-04。独立旧机制复核无应重启claim；新增动机来自E002条件/CFG相关性差异与用户指定H3 CFG1。E056复用E014/E015相邻teacher窗口，CPU complete0.433s、0GPU/0新forward；SVD视频去通道均值及输出比例项后cos .701306/.621988，保留92.8–98.7%能量。独立NumPy复算最大差6.11e−15。不能跨模型归因CFG或称长程累积。按预定门槛进入E057：四个BF16-on-QQ反事实及一个历史replay，检验真实两步注入的带符号交叉能量。没有新方法claim，C001历史Wan停止不改。De-biasing Diffusion方法访问仍403/验证页，细节未核实不等于新颖；普通kernel补强不作创新。报告051。目标已确认active。


## D081 — H3短窗口同向增强成立，但弱交互线性抵消门槛不通过

E057累计5次BF16 forward（历史replay1＋新control4）complete；v1在第二attempt的input prehook竞态停于forward前，v2使用阻塞D2H保持严格检查及原deadline，仅续余4次，40.147s/峰40.896GiB。四角summary CPU0.817s，SHA2a704f620fce8439d118ffa475bfd2ce9905908bead091d8896ec5ea08b020c7。SVD实际F/C正交叉项÷自身能量和.399929/.484620，I范数÷F+L范数.422165/.441016；centered同结论。正交叉不能宣称可直接消除40–48%损失。预设弱交互<.2失败，停止直接推进固定误差线性抵消/互补权重，不改阈值或扩网格；不把整个H3时间轴作用否定。相关来源、长程与感知后果未知，须有明确新预测才能再投入；无active新方法claim。报告051已更新；GPU0释放、全部新worker退出。


## D082 — 分离算术混杂，并停止把交互范数当损伤目标

2026-10-04，E058已按预写计划完成CPU分析2.024s、0GPU/0forward。四角BF16更新重放一致；SVD固定输入FP64代数更新I范数比.305083/.412913仍>.2。算术项确影响原I，但不能解释全部；保留原协议gate失败。进一步读带符号净效应，exact I使总误差能量相对F+L变化+0.167%/−6.167%，因此没有共同有害交互动机；大I也不是非线性或一切跨步方法失败证明。独立NumPy8份SVD输出/16模态角复算差9.51e−12，报告052。

独立方法筛选仅保留固定W/动态A来源的诊断候选、方法PARK。已有E009 software/native偏差能量相对native误差video.154909/audio.313393（不同p1s00），禁止未经验证把软件开关归因native；根在评估保持native GEMM的受控A残差干预，尚无GPU/新方法。新本地精度合同审查确认Diffusers FP32累加/部分模块与当前DiffSynth pruned不同，结构映射未证明，不自动重跑、不称普通修复为贡献。上一目标turn为progress（E056/E057完成并改变选择），本turn为progress（E058补机制混杂并改变损失动机），目标active无外部阻塞。


## D083 — 来源oracle未取得有效终点，不把控制停止当机制阴性

2026-10-04。E059预写native-preserving activation residual对照，6zero完整调用逐元素复现，首oracle在block0 FFN-down固定小片BF16加回RMS/补偿RMS .10338438693≥.1处停止。7attempt/6完整/0完整oracle，82.839秒；FP32公式相对FP64误差1.87e−6通过。没有来源效果读出，不能接受或否定A贡献，更不能断言W独占主导。严格执行原资源停止规则，不放宽阈值或扩精度网格；10%是局部控制投入门槛，不是科学不可辨识定理。失败样本/源/输出保留，独立NumPy复核已完成（0.420秒/0GPU，六输入及24输出数组精确一致，FP64补偿张量差0、统计差≤2.22e−16）；报告053。用户关于工程不作创新的约束持续生效，当前候选不晋级方法，目标active无外部阻塞。


## D084 — legacy基线身份成立，官方强基线等价性未建立

2026-10-04。E060 complete0.725秒/0CUDA/0forward，原state SHA重算匹配E009，200层600smooth/A/B逐byte一致；未重hash packed主体。实际saved候选2–6次，max50只是上限。当前standard生成脚本重置Q0换randomizedSVDseed，官方calibrator支持残差或完整权重初始化并carry-Q，前者初始化本身不是错误。原生产源码未绑定，不能把当前代码或候选数模式冒充历史执行证明。旧同实现结果保留，后续称legacy smooth+低秩残差基线，不能由它单独排除完整官方配方的修复空间。

分支相消/尺度支持范围两项筛选都没有可立项方法，不开kernel/scale搜索。E061预写固定smooth/rank/native的双初始化全200层输出对照，error_init/weight_init共同seed，四teacher点逐点video SSE同时较old_selected和error_init降≥20%才考虑正式matchedPTQ；先9forward，gate过再4shifted，总1800s包含22.28GB导出。它不测carry-Q、不代表完整官方复现、更不是创新。runner准备中，尚未GPU。报告054；本目标turn有E060实证与基线范围决策进展，目标active。


## D085 — 固定smooth的主方向初始化不足以支持继续投入

E061实际complete218.451683秒（两臂400层导出22.281GB+装载+9完整DiT），原native replay精确，峰37.611GiB。主weight_init/legacy SSE比.882778/.925071/.998096/1.058650；对error_init .917848/.902244/.993839/1.004678。四点均未达到预设双基线≤.8，且末点更差；按计划不补4shifted/视频/fullPTQ，不扩rank/seed/提示网格。CPU独立汇总0.561秒/0CUDA，另NumPy独立复核complete0.823秒/0CUDA，21份PT重hash、9输入与重放通过，最大相对差3.75e−16，报告055。

只停止这一固定smooth初始化对照；不能否定官方carry-Q、含LR的smooth或FP64SVD，也不能称legacy为已验证最强基线。p30/36与step5/14混杂，不编造时间机理；plain参照完整保留，不能据局部SSE选最终视频配方。此为必要基线判别、非新创新。新worker/tmux全部退出，GPU0已释放，目标active。

## D086 — SwiGLU 成对交互没有有害净作用，停止二阶修正候选

2026-10-04。依H3真实结构提出F1而非工程优化：同输入gate/up残余乘法是否有害并值得重新分配表示预算。E062预写合同、独立review，2原teacher×3预选层，完整输入真实native fc1后取512video行。v1两CPU启动失败0forward保留；v2修路径后check0.458秒/0CUDA，唯一GPU执行complete34.437秒、2完整BF16/6局部native、峰41.77GiB。六格 raw net/total 全负（−0.017%至−1.972%），同P去中心/比例及实际BF16算术保持负，0/3层达预设20%有害门槛。停止该乘性交互路线，不换层/提示/训练找阳性；不否定SiLU单支曲率或一般非线性校正。

SPEAR直接覆盖门控低秩量化补偿，NA-LoRA已有SwiGLU残余分解/敏感性；不能把泛化adapter或分解认领为创新。独立NumPy6.350秒/0CUDA，12PT freshhash、teacher实际输入/raw/velocity逐byte一致、统计差4.97e−16，报告056。BF16 c向量差15.8–83.7%但能量判断稳定，说明数值控制必要。当前无新方法claim，QK仅备选未执行，不自动立项；本目标turn是完成真实判别并改变投入决定的progress，目标active，无外部阻塞。

## D087 — QK 净改善主要来自切向，停止径向容量错配候选

2026-10-04。F2 经近邻碰撞后仅允许必要条件诊断，不认领通用功能损失或联合QK低秩的新颖性。E063冻结同smooth/shared activationpacket、fresh Q(Ws) rank0对legacyrank32，实际Comfy布局绑定并逐byte核验norm/RoPE，2原teacher×3预选层。CPUcheck0.424秒/0CUDA后唯一GPU执行complete69.667秒、2完整BF16/12局部native/6pack，峰41.97GiB。六格raw QK净改善0.454%–1.383%，径向所占四负、两正仅0.373%/2.853%；norm后改善0.384%–1.760%。原始收益本身小且主要切向，三个层均未过预设条件，停止该径向收益错配候选，不追加attention四角/训练/层提示网格。

独立NumPy3.832秒/0CUDA/0forward，12PT freshhash、两teacher输入/raw/vel及六格实际norm/RoPE样本byte一致；统计最大能量归一差5.98e−16。完整packet未保存，独立检查限执行源/签名合同，不声称CPU重现GPUkernel。此比较改变整套权重分解，不是孤立LR因果干预；norm误差也不是attention功能或视频质量。有限legacy状态阴性不否定所有QK校准，阈值是投入规则而非不可能性定理。报告057；worker/tmux退出、GPU0释放，F1/F2均停止，无active新方法claim，目标active。本轮是新增有效判别并收缩投入的progress，非基线工程创新。

## D088 — 实际深度路径的输入依赖项净作用为抵消，停止稳定净放大候选

2026-10-04。先核查CLQ/跨层联合补偿/作者相干草稿，排除通用深度累积和四角递推的新颖性；后缀SVD-null草案不执行。E064根据E014/E015完整plain与legacy配方的能量/方向差做有限来源定位，预选0/12/24/36/49×两原状态×两臂。CPUcheck0.345秒/0CUDA后唯一GPU执行complete467.429秒：6full/70local（10BB共享）、960native+pack/752SDPA/0disk，峰40.90GiB，24.546GB。六完整路径actual/raw/vel复现，四block0严格零。16非零格video signed net_echo/eout均负−0.338%至−3.386%，共同centering后−0.465%至−3.432%；没有深度满足20%正净作用，两臂均停止，不扩层/提示/后缀干预/训练。

独立NumPy175.933秒/0CUDA/0forward，104PT重hash，实际输入与E059、原始输出与E014逐byte、70local签名/对角合同及全模态统计通过，最大相对差6.72e−14。报告058。echo范数占总误差2.17%–15.87%，不能用它代表有害性；事后同Gram展开的量化有限输入响应增量却全部为正1.83%–13.44%，与qB负交叉项抵消后才为负net，因此不宣称量化更稳定/收缩。该解释亦由独立Gram复核，不改变原gate。

仅停止特定块级净有害放大动机，不否定所有输入依赖/联合校准，也不归因A4或视频质量。权重/激活来源仍未解。原worker/tmux退出，GPU0释放，无active新方法claim；普通实现验证不计创新。本目标turn完成真实source-screen并改变后续投入，分类progress；无外部阻塞，目标active。

## D089 — 补齐carry-Q配对基线，透明续跑，不将工程当创新

2026-10-04。核查E005已做RNE局部控制，不重复；实际activation residual表示与HeadQ/QUADS直接碰撞，尚无独立方法剩余。E061只换初始化，不应将其20%门槛扩为否决未检验carry-Q基线。E065固定全部200层smooth/rank32/seed、两臂各8候选，以8原校准状态的同native输出目标配对选择，再做四teacher整模读出。8校准+1legacy exact已完成，39层配对完成；1800秒监督后worker/tmux退出，最后JSON未finalize，另存终止观察而不伪造failed_stop。精确退出码未知，尾部候选开销界定为0–128，无新臂完整结果。

在新配方整模结果产生前，按实测资源速率冻结E065b续跑：同实验补剩余161层，不扩搜索；新增3600秒与前1800秒分账，新增9full/累计18。CPUcheck47.791秒/1923复用文件/0CUDA通过，GPU0 worker537031已启动，deadline1791072443.932757，fresh legacy replay通过并完成头两新层。最终有效候选25600、物理计费范围25600–25728；已准备完整输出与选中局部SSE两个独立CPUchecker，等结果再运行。报告059。当前无active方法claim或视频质量证据，目标active；续跑仍在进行，不能称整项完成。

## D090 — carry改善全部校准格，但四点video整模误差反向

2026-10-04。E065b complete3294.944184秒，200层双臂导出完整，新9full/20608候选native，原39层只读复用；两段18full、有效候选25600、物理计费界限25600–25728。峰37.6044GiB、累计115.058GB，process_exit.json记录exit_code0/非timeout，worker537031/tmux消失、GPU0 0MiB。原E065未终结JSON/半成品保持原样，未重启实验。

carry的200/200层局部SSE更低，层中位比.969830799，1600/1600校准格均更低；video整模四点carry/restart却1.129255874/1.109991657/1.004742296/1.044936389，centered也全更差。audio三点更好，一点+0.433%，不能写成全模态阴性。191层选k7且184条曲线单调不增，并不意味着更多迭代会恢复整模收益。原预写一致整模收益条件不满足，不扩本配方迭代/rank/seed或直接推进视频验证；不否定完整官方PTQ。

独立输出复核0.787228秒/21PT、差4.45e−16，局部选择271.724096秒/2612PT/400个最终安装清单绑定、差4.18e−16，均0CUDA/0forward。只重算选中局部输出与完整模型误差，不声称重演全部SVD/native/未保存候选。stdlib描述汇总0.075606秒，evaluate SHA08ac1d487b8683ab9ae7dcaeb42b46a76795435165c8fef3314bfc77c4f56e48，summary SHA1e5c2133fd8adb5a64cefa89dca3d0d5d0d8637c89c5dc2b6d05ad593da37efa；报告060。

这是一条已核验的校准目标/整模读出反转观察，尚不能分辨未见输入局部失配和组合传播，更不能命名新机制、损失或质量收益。下一先审现有缓存能否判别相同未见teacher输入上的局部优势；不为泛化“层间联合校准”立新claim。ARHQ/TwinQuant直接近邻另已核实，实际残差加权低秩迁移不当创新。研究目标仍active、无外部阻塞；本轮完成真实实验与独立验证，属于progress。


## D091 — 同一teacher输入逐层优势保留，排除简单迁移/形状/采样排序假象

2026-10-04。E066执行前补同packet真实M512控制，4BF16/3200local预算不拟合候选；CPUcheck35.382482秒后唯一GPU阶段complete350.657461秒，800pack/408SDPA/0disk、峰43.7253GiB、71.8775GB，exit0/GPU0已释放。四teacher actual/raw/velocity重放精确。full-M video每点200/200carry更好，中位比.969639403/.969812696/.970606081/.970469527，真实M512/full-M同样本排序亦全一致；最大shape比率相对变化3.71071e−5。joint/text全部改善，audio两例外均block3.fc2，不泛化全模态。

独立CPUcomplete247.513912秒/0CUDA/0forward，812PT/400导出、3200保存输出、409600样本行；样本误差与全行聚合最大相对差5.8134e−16，未重算未保存full-M通道/native。stdlib汇总0.058611秒，summary SHAf42d86fdd0f36b76e8c854af36ded8bf06421295098613590f85d3ef932c7d4c；报告061。源/产物冻结，所有任务结束，不重跑检查。

与E065b整模video四点恶化结合，得到同teacher输入逐层局部优势确实迁移、但不保证整模收益的窄结论；不直接归因跨层干扰/相消/模态或画质。固定非负逐层video/joint SSE权重无法翻转两个已有候选的排序，不否定使用加权目标重新训练第三候选。这是研究问题边界、非新颖方法。暂无充分理由开下一GPU或泛功能loss；先提出能区分方向敏感性与实际路径输入变化、会改变方法选择的干预预测。不扩迭代/rank/校准或全层扫描。用户工程非创新约束持续有效，无可投稿核心贡献；本轮progress、目标active、无外部阻塞。


## D092 — 局部候选差主要非径向，但先核验完整视频任务相关性

E067 complete551.028857秒/0CUDA/0forward，812PT freshhash/RSS2.12GiB/exit0。四状态各200层样本video两臂cos中位约.739，差值δ中正交于eR的能量占比中位85.6%，全800格80.09%–90.83%，去均值后亦正交主导；同权重形状δ差异能量≤.0325%。拒绝简单径向近似，不将其当方向致害或特殊机制。新增方向统计仅保存样本，SSE绑定已有独立结果且恒等式通过，未另程序全量独立复算。报告062；analysis SHAf414e380aef5a30855082458e07c11e28314ddd79afe15b895e3b9e05236f776。

方法论修正：完整teacher DiT SSE反转尚未证明画质损伤。与其继续纯SSE机制扫描，先进行冻结两臂的E068四视频任务相关性核验，使用原两prompt/seed/noise/embedding/decoder及历史BF16参照。若carry未一致可感知变差，不继续把SSE反转当损伤；若难区分不宣称等价，若更差才支持窄范围进一步定位。需隐藏arm标签的完整视频与人工判断，MJ/模型观感不可冒称人评。新计划并非扩E065校准搜索或改其旧成功门槛。当前仅准备、0CPUcheck/0GPU；两卡共享900s、82full、<60GiB/卡、≤10GiB，既有研究授权有效。SVDQuant/ARHQ/QDrop及MatGPTQ近邻审查无可立项残余，普通工程非创新。目标active，无外部阻塞，本轮progress。


## D093 — 完整视频生成及独立核验完成，质量尚待人评

2026-10-04。E068两臂四阶段complete、两个supervisor exit0，GPU0/1释放；共享900s窗口实际349.417秒，82full/16400native+pack/8364SDPA/160updates，4video+4audio VAE，0新BF16/TE/MJ，峰37.6044GiB/卡、1096603006B新数据。独立CPU4.256021秒、0CUDA/0forward：160新step/80历史schedule、两teacher replay逐byte、初态/相邻/末态、四视频496帧及PCM全部通过；未重演scheduler算术或任何模型，历史编码器字节重演不主张。独立SHAe3ae5d66c81bc758ffec5992d9b7d681220ed4238f7fe8e16567086c9d558672。报告063。

四新视频与两历史BF16参照已按生成前commitment盲化，6中性媒体合计26157199B，未裁切/重编码/评分。当前无人工偏好；标签映射只在private保存，不公开。收到一致退化才推进任务相关机制；改善则停止将SSE反转当画质损伤；混合/难分不推等价且不继续SSE-only修复。普通生成/核验/播放控件工程不算创新，无新方法claim。目标active，本轮真实完成视频判别材料而非仅计划，属于progress。


## D094 — 固定上游配方核查：必要强基线仍未补齐

2026-10-04。E069将上游main解析到commit69f3473f5e1c1504bae35cc50c7858ef900a9b17，保存12源码243155B。CPU AST语义审计complete0.032997秒，0torch/GPU/forward：当前H3g10指数候选9，vendored/upstream规则g10为19、默认g20为39；遗漏identity和activation-only族。旧H3选smooth未纳入LR；上游完整FP64 SVD、本地随机FP32 SVD；E065b仅固定smooth的8teacher逐linear比较，不代表完整强基线。audit SHAaa993946fbf01fc428067ba9a0d6cb59befec0cdbf7b5bbb4fd79de10480faf0。源码差异不推实际效果，也不推历史执行，旧producer SHA缺口保留。

当前H3 PTQ源已有全block评分与块间量化输出传播，不能构造“忽略gate/norm”的稻草人。官方目标按模块选择，缓存输出在yield校准前保存，不能将其use_prev等同于旧H3student重传；量化格式层级/默认初始化/截断SVD求解也要明确。结构筛查0新方法候选。下一先保证packed变长序列接入正确，再做一层资源pilot；通用TensorCache切第0维且同shape假设不适配直接[M,C]缓存。case-index wrapper仅方案，未模型运行。工程补齐不算创新。

E068人评仍待，不重复询问，不伪造质量结论；这不阻塞独立强基线工作。上一goalturn为E068完成progress，本轮可复现合同核查改变后续基线选择，也为progress；目标active，无新GPU任务。报告064。

E069独立复核complete0.027004秒，31文件freshhash，Fraction重建候选集合与顺序通过，0torch/GPU/forward，未重跑原方法；SHA6a98b15801658cb7ca2e5b56072063756b7089005a8892b1c779247d946b167d。只覆盖源码/枚举，不证明实际scale互异或模型效果；本阶段结束。


## D095 — 强基线真实入口和最大层精确SVD可执行，不构成创新

2026-10-04。E070-A两个原校准状态M22400/22464，真实cache索引接口→完整attention输入；2prefix+2wrapper在16.522474秒完成，12SDPA/0full/0quant，所有input/kwargs/output byteexact，40.2034GiB峰、1964222205B。首CPU版本因装饰函数定位误判0CUDA失败保留，v2只inspect.unwrap/独立路径修复；源与执行门槛不覆盖。独立3.474768秒/0CUDA重验4新PT、两原case、全44864行span，SHA b8e86887805c72ca4c313b487ed04084a587f6bec01a1fc9b1d918ff349ddd76。

B一次真实原始fc1[28672,5376]默认full FP64 SVD59.398311秒，总61.284249秒，峰10.4537GiB、15300421B。独立NumPy1.322301秒/0CUDA重hash该完整权重、检验top32抽样方程/Gram及BF16 A/B，SHA7c17c8d506cdbc515ef4df472a4af58d1e117301036aa3acfb09b935f90a76d7。全U/Vh未保存，不能称独立重演全谱检查。两supervisor exit0、GPU释放，原900/1800秒deadline未重置。报告065。

本轮解除两个实际入口不确定性，不证明完整评分/校准或画质，更不能把单原W分解成本推为所有候选成本。下一只准备单层单候选实际评分与导出一致性；native BF16 scale、LR读取未量化x_s、一次平滑、库内评分低精度减法需明确合同。0新研究候选，工程改进不作创新；E068人评仍待且不重问。目标active，本轮实际判别完成为progress，不因无新颖claim假称受阻或完成。


## D096 — 单候选校准/部署一致性成立，结束该接线检查

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



## D103 — 观察与研究想法的边界

**D103 / [报告072](../reports/072_20261004_observations_and_candidate.md)：回应有趣观察与idea，0新GPU。** 保留三个不同层次的事实：局部Pareto改进却整模SSE反转；FP4attention分组相位近等误差能量下改变误差向量；已看拍手中可辨性下降不等于统一减速。后两项来自attention精度/布局干预，不冒充主干W4A4失效归因。建议讨论近等能量真实干预与最终可辨缺陷的因果连接；仅问题/待验证想法，尚无执行计划或active方法claim，不能将旧停线直接复活。CLQ/PARO/跨步补偿及noise shaping已有覆盖，泛loss/重排/随机化不认领新颖；若缺陷不可辨或只有构图变化则停止。E073匿名行为读出尚待，E074数值距离不代替；用户指定基线继续冻结，不恢复官方校准工程。



2026-10-04 D104/E075：近等误差量相位因果续跑，19DiT/2视频/1200秒。CPU检查通过；GPU0被他人占用，启动前退出0GPU，保留launcher.log；仅调度改GPU1，科学协议/runner保持不变。


D104/E075完成：[报告073](../reports/073_20261004_h3_phase_suffix_results.md)。19完整DiT/2新视频，一致性控制通过；近等局部误差改变最终视频，但固定抽帧无明确缺陷修复，局部BF16 oracle亦无明确修复。不扩此窗口网格；GPU1释放，原预算未延长，所有失败保留。


**D105 / [报告074](../reports/074_20261004_h3_action_and_decode.md)：E073匿名静帧观察及揭示已完成，E076 complete。** 8例×4页全部查看，先冻结观察后读E073映射；拍手r1/叠衣两seed有主干量化运动边界模糊线索，但不是人评或确定机制；大象后段亦喷水，撤回初段缺失判断。E068 key未读。E076两臂首时间窗30局部decode、8.415秒、4.91GiB、0DiT；raw tile内已见手部重复轮廓，停止主要归因空间blend，不扩tile网格。source/plan冻结，raw与JSON保留。

**E077探索已启动：** 固定block25分界的restart/carry二段2×2组合，16完整前向、600秒、0校准/训练/视频。候选是同rank/布局的静态版本组合；CLQ及联合离散码工作已有跨层叙事，尚无创新claim。四原teacher点只有两提示簇，非独立新测试；同一混合方案若四点均优于两个端点至少5%才进入独立视频验证，否则停止这个粗粒度候选，不扫分界。此决定有限重开可直接给出部署方案的组合测试，不重开泛SSE扫描。协议见06_experiments/E077_half_composition_plan.md。



**D106 / [报告075](../reports/075_20261004_h3_composition_candidate.md)：E077 complete，16完整native前向，138.298秒，峰37.61GiB，GPU0释放。** RR/CC八次raw精确复现历史；CR四点video SSE相对RR为−5.78%/+0.095%/−0.60%/−1.89%，RC为+19.16%/+17.58%/+1.92%/−0.29%。两混合均未通过固定四点至少5%改善门槛，停止二段静态组合候选，不扫分界/分层网格。RC音频SSE四点下降约18%–35%，仅模态收益不同的探索线索，不是音质/画质或模态竞争因果证据。有限向量交互大但不是有害性证明。CLQ及联合离散码工作已有直接近邻，无新颖方法claim。独立CPU复算完成，源/计划冻结；当前无GPU任务。E073/E076已完成状态以D105为准；E068 key仍未读。


**D107/E078：用户指定探索官方SageAttention3叠加正式SVDQuant基线。** 不把组合本身作为创新，也不以旧FlashInfer/QKV-only实验替代。固定两个完整teacher输入的BF16、仅SVD、仅Sage3、组合四角，共8DiT，检验相对实际误差向量和的净放大；原主基线200linear/rank32不变。官方源码pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5，隔离DATA1构建，不升级共享环境。接入先遇CUTLASS下载停滞、系统nvcc13与torch12.8冲突；改缓存CUTLASS4.5与NVIDIA官方独立12.8.1最小工具链，SHA验证。当前编译中、模型尚未执行；各失败日志保留。协议06_experiments/E078_sage3_interaction_plan.md，执行入口scripts/research/probe_h3_sage3_interaction.py。候选由用户明确提出，继续进行接入与可解释对照；普通接入不作创新。


**D107/E078 complete / [报告076](../reports/076_20261004_h3_sage3_interaction.md)。** 用户指定官方SageAttention3＋正式SVDQuant已完成四角对照。官方pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5原码构建，8完整DiT/69.627秒/43.26GiB，200次官方attention扩展＋1smoke，B0/S0四次raw逐byte复现。video组合/仅SVD为1.1620/1.3393；相对独立误差向量和的净放大比0.7519/0.7122，故未支持视频净放大；切换attention的输出变化能量比1.3784/1.3454，尚不知上游输入vs下游传播来源。audio净放大1.1277/1.0123，不稳定。无视频/音质结论，无新方法claim。CPU独立复算0.169秒/0CUDA、相对差2.22e−16；GPU0释放，无研究任务运行。官方接入成本另计：CUTLASS下载停止、nvcc13/cu128冲突、缺依赖头均记录；独立cu128工具链及缓存CUTLASS解决，无源码/共享环境升级。E078已执行源/计划冻结，勿重跑。下一若定位，只做能区分局部attention误差与后缀响应的固定输入干预，不默认扫描。


**D108：按用户指示，后续工作baseline更新为官方SageAttention3＋现有SVDQuant，原BF16与仅SVD作为参照保留。** 不改历史标签；具体合同与E078一致。E079固定四动作×两seed，复用E038准备输入及E073/E038已有参照，新增8组合视频/160DiT。CPU两replica检查已通过，GPU0/1各4例已启动，生成＋解码各共享1800秒。随后对三臂做明确标注覆盖范围的视觉观察、并排视频与LPIPS/MAE/RMSE；不把距离当质量百分比或静帧当人评。


**D108/E079 complete / [报告077](../reports/077_20261004_h3_sage3_video_quality.md)。** 工作基线已更新为官方SageAttention3＋现有SVDQuant，BF16/仅SVD保留参照。8组合视频/160DiT/8000 Sage3调用及解码完成；全部32页顺序抽帧观察：拍手两seed有额外手指模糊/重复轮廓（r0最明显），叠衣增量不确定，汽车/大象未见明确额外崩坏。非人评/实时完整播放，不归因有害交互。8例对BF16 LPIPS/MAE/RMSE为0.44881/0.11820/0.17986，仅SVD为0.44336/0.11692/0.17510；距离不代表质量百分比。16配对CPU归约复核通过，生成/解码/评价进程均已退出。视频在DATA1 research/20261004/E079/review，候选仅为局部细结构损伤，未形成新方法；停止追加本轮GPU。


**D109 / [报告078](../reports/078_20261004_fine_structure_hypotheses.md)：精细结构机制与方法候选。** 拍手是当前最明确可见缺陷，非总体最大问题/softmax变平/有害协同的证明。只保留P011下C007（量化前PV双侧差分表示）和C008（沿运动输运的量化误差结构），均proposed、untested。有限查新剔除V中心化/保排名/重要块高精度/泛时间loss/导数匹配作为独立创新；VC-Attention、DIDB-ViT、Q-VDiT、2ndMatch为关键近邻。先固定输入机制干预，后决定工程，尚未启动新GPU。主baseline仍官方Sage3＋现有SVD；本轮0GPU、0新效果。


**D110 / [报告079](../reports/079_20261004_vc_didb_compatibility.md)：兼容性与查新更正。** VC的V-Smooth可融入Sage3 FP4，需要kernel均值补偿；ExpCast-FP8不能直接套NVFP4。未确认官方公开实现，不能许诺直接安装。DIDB是需训练的二值ViT结构，不能插入现成H3；它也直接保护attention差分，C007动机碰撞比上轮描述更强，仅同预算PV双侧表示残余待验证。无代码接入/新GPU，现baseline不变。


**D111/E080 started：** 用户授权C007双侧PV差分表示实验。两clap seed原轨迹重放40DiT，固定s14/b0,b24捕获，要求最终latent exact；局部同bit/scale对照与预设差分误差gate详E080_pv_contrast_plan.md。初版CPU检查因继承manifest输出路径guard失败（0GPU）已保留；v2两CPU检查通过，GPU0/1各900秒capture启动。执行前计划已冻结，不改门槛，不做C008或新kernel。


**D111/E080 complete / [报告080](../reports/080_20261004_pv_contrast_results.md)：** C007当前双侧PV差分表示停止。两clap seed重放40完整DiT，最终video/audio latent逐byte复现E079；s14/b0,b24全部56heads、36对query、完整keys，QKV官方pack零差异。native QK下b24差分SSE降14.29%/13.30%，但总SSE增6.23%/5.57%；V中心化32差分降30.31%/31.04%，同预算shuffle已降12.96%/9.73%，pair只额外改善1.54%/3.96%<5%。b0总SSE增48.5%–56.7%，exact QK结论一致，未过预设门槛。contrast_gain未更接近1，不能认领细节恢复。独立CPU824项复核通过，归一化差≤4.44e−16；模拟/native gap能量比≤0.0113%。捕获128.22/126.60秒、峰37.60GiB，局部成功14.55秒/13.20GiB；CPU入口失败及FP32归约验证失败均保留，新版独立FP64验证通过。0新视频/VAE/训练/kernel，GPU0/1释放，不扫参数或启动C008；无新颖方法或画质收益claim。


**D112（2026-10-05）：停C007不等于停V中心化验证。** 用户指出强对照明显有效，前次沟通混淆创新性与实际效果筛选，现纠正。E080 center32在b24局部PV SSE降40%–42%，含QK误差的同输入完整attention SSE仍降15%–16%；值得有限端到端验证，按已有方法/增强基线对待。当前双侧表示仍停止；V中心化后续验证待执行，0新GPU/视频，暂无质量收益。center精确概率质量补偿也消除P舍入质量误差乘V均值项，不能将全部收益归因V范围。详报告080末补充。


**D113 / E081 complete，E082准备（2026-10-05）：** 用户授权跟进V中心化。两因素消融复现E080：b24/g32仅V表示SSE降22.32%/25.31%，仅概率质量补偿降23.02%/21.93%，合用降40.24%/42.07%；两者均有贡献，g128/QK条件/full参考一致。b0补偿可解释84%–85%组合收益，不能推广层规律。独立CPU856项验证通过，最大归一差4.44e−16；GPU13.414秒。下一E082以g128最小私有native接入进行原生验收，通过才两clap完整20步视频；已有方法验证非创新，官方基线不改。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。


**D114 / E082 complete（2026-10-05）：** V中心化g128已完成原生验收和两clap完整视频，40DiT/2video+2audio VAE。zeroμ逐byte复现，center模拟gap≤0.68%，原生b24全attention SSE降14%–15%；两视频的32固定抽帧/clip均未见明确稳定手部修复，r1另独立审阅。对BF16 LPIPS原→center r0 .27126→.26532，r1 .36829→.42654；MAE/RMSE均增，r1构图姿态改变，不把距离当质量。保留已知中心化局部阳性对照，不替换主基线，不扩kernel/层步网格，不否定完整VC。全部检查与评价结束、GPU0/1释放；生成137.68/136.73秒/37.60GiB，解码11.92/11.47秒。记录两次实施失败及修正，不记作方法阴性。详[报告081](../reports/081_20261005_v_center_mechanism_and_video.md)。


**D115 / E083 running（2026-10-05）：** 按用户建议，隔离attention：BF16主干＋官方Sage3 vs BF16主干＋同一已验收center128，复用完整BF16参照。两clap seed共4新视频/80DiT、GPU0/1各共享1800秒；四CPU输入/源检查已complete0CUDA，开始原20步完整采样。检查200原BF16 Linear、每步0scaled_mm/52SDPA/50FP4；不新增kernel或改分组。40%–42%仅g32固定量化QK下局部PV SSE，不是完整attention或视频误差；当前均值分解是VC V-Smooth已有组成，不认领创新。见[E083计划](../06_experiments/E083_bf16_attention_comparison_plan.md)。


**D115 / E083 complete（2026-10-05）：** 用户建议的BF16主干隔离对照完成：两clap seed×官方Sage3/center128，4新视频/80DiT。200原BF16 Linear及每步0scaled_mm/52SDPA/50FP4、输入/源/时间表/VAE合同全部通过。对完整BF16的LPIPS原→center为.16658→.17749、.34304→.36946；MAE两例增大，RMSE一升一降。固定32抽帧/clip未见稳定细节收益，seed1另独立审阅且有构图变化，不把距离当质量。不能仅用SVD主干混杂解释前次未见收益，也不否定具体精度交互/其他场景/完整VC。40%仅历史g32固定量化QK的局部PV SSE；V均值分解为已有方法，不新增claim，不扩参数网格，主baseline不变。六配对独立归约/六媒体SHA通过；全部进程完成，GPU0/1释放。详[报告082](../reports/082_20261005_bf16_attention_comparison.md)。


**D116 / E084 complete（2026-10-05）：** 补齐E022原16测试轨迹的逐步同状态/自由输出/latent NMSE。288 DiT、192步哈希与48终态精确复现；CPU4192项独立校验通过。测试同状态第1–4步pooled变化−22.24%/+40.45%/+33.37%/−2.03%，开发32%降幅未稳定迁移；自由输出末步−8.33%，正交叉项增加使终态latent pooled仅−0.96%，等权+17.17%。GPU0 506.78秒/4.15GiB，无训练/视频/VAE；补测完成即停止，不恢复新idea研究。[报告083](../reports/083_20261005_wan_qad_timestep_replay.md)。
