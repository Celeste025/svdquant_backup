# 当前问题

## P001 — 跨步量化误差的统计模型是否遗漏了持久方向？
来源：S1（literature_signals.md），历史局部误差优化不能预测最终收益的阴性证据。
问题：相同低位权重重复参与多次denoising时，误差不是每步重新独立抽样。现有采样端纠错常把噪声当时间独立；在NVFP4视频生成中，忽略其跨步方向相关是否导致纠错误判？
重要性/owner：量化方法与数值采样器设计者。若相关项是主要来源，仅优化单步误差或估计单步噪声方差不够。
已知部分：误差累积、输出相关、历史平滑都不是新问题，PulseQuant/PTQD/TCEC/QuAKE是必须对照。
待检验gap：相关是否在去除简单channel bias后仍明显，是否由固定权重还是activation引起，是否在closed-loop与真实NVFP4中有可利用的稳定性。
边界：不以local-global不一致、加缓存或套Kalman本身作贡献。不先做全部模型或训练大模型。
状态：E002已完成，完整W4A4强持久方向假设在真实CFG对照下被否决，当前路线停止。

## P002 — serving统计域是否改变单请求量化语义？
来源：S4和NVFP4全tensor scale。
问题：独立请求共同amax是否导致互相影响；已有原生kernel是否提供不损失batch吞吐的隔离方式？
重要性/owner：动态batch部署人员。
已有反证：E4M3正常区间下纯2^k全局scale变化可被指数吸收；不能仅因global scale不同就认为存在严重质量问题。
边界：bitwise差异/加per-request scale本身不足以撑论文。
状态：parked，先保留数学控制，不消耗模型GPU。

## P003 — joint模型局部模态重构目标是否错配最终生成响应？
来源：E001真实packed block0配对；text仅少量token却支配输出/误差能量，SVD的全block指标与video-token指标反向。
问题：校准用合并模态的输出误差时，保护text residual是否等同保护下游video生成？单纯能量差不能回答。
重要性/owner：联合音视频扩散的低位压缩与校准设计者。
现有gap：模态加权已有MBQ，条件极值已有MixDQ，几何度量已有RGSQ；本轮未证明新缺口。先检验H3中的实际因果排序，而不是再提reweighting。
边界：LayerNorm单层不变性不是整个residual/AdaLN网络gauge；不通过丢弃文本精度获取未经评测的“提升”。
候选方向：C004因果排序错配；若成立可推导更有意义的响应目标，若不成立停止该解释。
状态：E004完成后当前路线停止；p1现象不能外推为稳定跨prompt规律。

## P004 — 原生projection量化是否改变下游attention量化的误差契约？
来源：第二阶段近邻审计；已具备真实SM120 GEMM和attention入口。
问题：低位qkv投影扰动后，FP4 attention的误差是否有不可由独立误差叠加解释、且由QKV scales切换驱动的成分？
已知：SVD+Sage组合、联合校准、一般非线性误差传播已存在，见phase2_candidates.md。仅证明组合更差不够。
实验：E006真实2×2与固定QKV scale重编码，实际norm/RoPE/cu_seqlens完全固定；计算FP32张量交互，非NMSE相减。
边界：固定QKV scale不冻结内部P；必须先自尺度重编码与native逐byte一致。小样本teacher输入不代表最终质量。
状态：E006完成并停止当前机制。12case中组合/加和误差中位0.9608、0/12净放大；固定QKV scale中位收益6.46%不足预设10%门槛。不再做剩余网络continuation。


## P005 — 输出通信与原生SVD投影接口（历史补索引）

完整问题与强对照已记录于[E039接口审视](../00_state/E039_interface_prior_review.md)、[C006范围判断](../05_scope/paper_unit_gate_C006.md)。E039有真实两卡边界收益，E040确认局部信息差，但已有SVDQuant/FlashInfer ABI覆盖核心表示，未建立质量必要性。状态：independent paper route stopped，工程实现保留。不因本条补索引重启实验。

## P006 — E038低位动作退化能否被定位为具体事件机制？

来源：S011、E038自由生成配对及E041完整原生帧。问题不是“需要更好的量化”，而是：在固定native线性权重、只改attention的受控条件下，低位画面的具体失败究竟是动作事件变化，还是可辨性退化令光流和事件读出改变？只有先区分这些，才能决定是否存在超出已有低位attention质量损伤的新机制。

重要性/owner：视频量化方法与部署评价人员。把无法看清的动作当成没有动作，会把后续干预优化到错误目标；但这一评价常识本身不构成论文贡献。

已有内容与当前gap：Attn-QAT直接覆盖attention-only动态下降，QuantWM覆盖时空token选择偏差，VC覆盖P/V误差；SSVAE覆盖decoder对生成latent误差的鲁棒性。E041没有建立超出这些先例的机制残余，因此当前gap **未成立**。未在有限查新中找到同一H3条件不等于新颖。

问题类型：measurement / mechanism screening。时间背景：已有真实NVFP4 attention和六条可逐帧复核的相同权重媒体，可在不生成新样本的情况下排除解释。

已判别解释（不是新claim）：①统一动作变慢——可读前段仍快速开合，后段unknown，当前不支持；②全部重影来自最后时间blend——非blend帧反例直接否定。更广义decoder作用尚未定位，不能凭“未排除”立项。自由分叉终态之差含内容变化，随机通道旋转对照不具有终态量化因果识别性，当前不执行。

候选claim方向：**当前无合格方向**；不为凑2–4项而把上述既有常识或待排除解释包装成研究claim。另一个尚未读出的音画事件维度只做E042入口检查，不能事先认为它已受损或值得写论文。

边界：不恢复E022的QAD运动收缩、不扩中心/聚类/decoder参数网格、不发明运动/感知loss，不把光流、归一化能量峰或模型同步分数当物理事件真值。

状态：**parked as paper problem；有限既有媒体读出继续。** [阶段037](../reports/037_20261003_clapping_readability.md)、[解码上下文反例](h3_decoder_context_readonly.md)、[音画入口限制](av_event_readout_entry.md)。


P006/E042更新：固定PCM包络已complete，读出不等于听音或语义事件标注；基线对应尚未建立，不给AV同步损伤claim。P006继续parked，停止当前样例上的算法/阈值派生，不将未知解释为物理同步机制阴性。


## P007 — H3量化残差是否具有值得研究的输出patch特殊结构？

来源：E003/E017/E038局部误差与视频读出不一致，以及H3真实高精度2×2输出头；具体假设和竞争解释于E055计划执行前固定，本条补问题索引。
问题与owner：量化设计者是否应优先保护某些输出子位置，而不是仅优化总体MSE？需要区分普通head传递、有色/各向异性误差与量化特有额外损伤。类型：mechanism screening；gap尚未建立。普通子像素伪影、关系蒸馏、decoder鲁棒性已有近邻。
候选解释：H0普通head及相关hidden误差即可造成相位结构；H1在这些控制之外有额外patch边界损害。边界：不把统计阳性、白噪声null失败或MJ涨分当机制/质量，不伪逆声称实测hidden，不立刻训练phase loss。
E055结果：3状态×plain/SVD中同距离patch内外归一协方差差都低于iid-head模板；实际跨patch相关仍强，不能说白噪声已完整解释，但不足以支持H1。状态：本次patch边界特殊损伤路线stopped；没有注册新方法claim。见报告050与D079。


## P008 — H3相干误差是否有可在单份W4预算内改变的激活来源？

来源：E056 native H3 CFG1两个窗口去偏后相邻cos .701/.622；E057实际注入正交叉40%/48.5%；E058区分第二步算术后交互仍大但净效应非共同有害。问题不是再最小化局部MSE/交互范数，而是是否存在足够的激活侧可干预成分，值得在不增权重副本/前向次数/激活历史的预算下研究时序控制。owner为低位视频推理方法设计者。

竞争解释：固定W重复投影；动态A表示残差；普通数值算术和有利抵消。20%旧gate不证明一切时序方法无效。TCEC/QuAKE、SR去偏、Ditto差分缓存均是已知近邻；尚无论文残余claim。De-biasing Diffusion精确方法覆盖仍未核实。

E059只做保留native主GEMM的activation残差oracle，总共6固定输入×zero/oracle，不称W4A16或部署收益。原software source-toggle因已有native偏差不作主实验。只有两窗口去偏正相邻点积都减半、每个teacher endpoint能量不增>10%且数值控制可靠，才设计一个具体低预算干预；否则停止当前activation时序候选，不扩网格。状态：diagnostic preparation，方法PARK。计划E059是执行协议，创新未成立。

P008/E059状态更新（D083）：六个zero控制精确复现，但首oracle在block0 FFN-down固定数值样本10.3384%≥10%的加回舍入门槛停止，0完整oracle。独立NumPy确认。来源假设未判定，按预写资源规则停止当前activation时序方法推进；不把技术控制停止记作机制阴性、不扩精度/提示网格。详报告053。

## P009 — 完整 H3 配方差是否需要输入依赖量化残余来解释？

来源S012：完整plain/legacy输出的能量与方向不一致，固定smooth初始化没有一致解决。问题是：沿各自真实深度路径，当前块的teacher局部残余与BF16传播之外，量化块随输入漂移改变的残余是否有稳定的实质正净作用？这决定下一步应先研究局部重构还是输入依赖来源，而不是泛称需要更好的量化。

Owner：视频量化方法设计者；类型measurement / mechanism screening。现有全模型原生输出和实际block入口允许有限四角定位，避免纯teacher上的局部代理。当前论文gap尚未成立：CLQ已有cross-block校准，2607.14630已有精确量化有限差分递推和跨层联合补偿；一般输入偏移/累积不是新贡献。

候选可证伪方向（未立claim）：①输入依赖项小或抵消，停止反馈放大动机；②它在两个固定状态、多个预选层去偏后仍净有害，允许进一步真实后缀干预及来源控制。第三种结果是identity/metadata/数值检查失败，只报告不可判定。即使②成立，仍未证明NVFP4独特性、A4来源或比现有block重构更好的可部署方法。

边界：H3 CFG1，两旧状态×两既有配方×五预选block；不重开F1/F2、不改旧配方、不从block SSE推最终质量、不将BF16 residual恒等carry称放大、不将数学去项视为免费修复。E059的W/A来源仍未解，本次直接保存四角后FP64相减，不做微小BF16oracle加回。

状态：diagnostic preparation；方法PARK。执行计划[E064](../06_experiments/E064_h3_depth_four_corner_plan.md)。另一个[后缀相干草案](h3_depth_coherence_screen_20261004.md)暂不执行，先用本诊断判断是否值得进一步投资。

P009/E064执行后：两臂各自路径的16个非零深度video净作用全部为抵消，raw−0.338%至−3.386%，center后仍全负；独立104PT/全模态NumPy复核通过。状态改为**本次稳定净有害反馈候选stopped，方法不立项**。量化有限输入响应能量增加却被qB负交叉项抵消，不得换称更稳定或不受输入漂移影响；W/A来源、终端方向/质量仍未确定。见[报告058](../reports/058_20261004_h3_depth_four_corner.md)。不启动后缀干预/训练，不将有限阴性推广全部联合校准。


## P010 — 固定部署预算下，严格逐层优势为何不能排序完整候选？

来源S013/E065b/E066。问题：在同native NVFP4主路、rank32与调用预算下，候选选择是否遗漏了一个可干预的具体结构，使所有teacher层级video/joint SSE更优的候选仍有更差完整DiT预测？owner为低位模型方法设计者，类型暂为measurement/mechanism screening。重要性在于避免以强局部改善误选实际配方；不是泛化“需要更好的loss”。

Current gap尚未建立为论文gap：已知分解/残差协方差/扰动重建覆盖自然方法出口，见[最近先例审查](../04_prior_work/h3_post_e066_deployment_structure_screen_20261004.md)。当前能问的是：同输入局部变化能否近似径向；若不能，是否只是普通候选重排；真正实际输入响应是否有不同于成熟扰动校准的可预测离散结构。后两项尚无证据，不立方法claim。E064的不同配方净反馈阴性不改名重开。

候选可证伪研究方向（不是已成立claim）：①保存teacher局部变化近似径向，撤掉新增方向的动机；②显著非径向但无特定有害方向/可干预结构，仍停在描述；③未来若某个实际packet切换结构能预测失配，并在同预算干预下区别于普通扰动重建，才重审机制与算法贡献。NVFP4动态global使胞元不能简化为固定阈值或总Jacobian rank32。

边界：当前只有四重复诊断点，E067新方向统计只覆盖保存的512联合样本中的video/joint行；不能借用E066全行SSE覆盖。纯径向不保证整模更好，非径向不证明方向致害；不推出视频质量、不加loss训练、不扩GPU扫描。状态needs literature/mechanism evidence，方法PARK；一次0GPU有界E067描述用于检验近似，之后仍须具体可区分预测。


## P011 — 小幅局部对比误差如何变成精细结构模糊？

来源E079拍手两seed有标签抽帧观察，E078无稳定视频超加和放大。问题：相邻结构/运动相位的差异是否比全帧误差更决定模糊，能否用固定NVFP4预算保护？Owner为视频量化方法设计者，类型mechanism→method。时机是已有官方Sage3+SVD配对可见失效；gap尚待证明而非宣布成立。边界：不做层/步组合搜索、后处理锐化或以MSE冒充质量。候选C007保护PV差分表示，C008控制运动输运后的量化误差变化；均未实验，方法未成立。完整动机、竞争解释与停止条件见[报告078](../reports/078_20261004_fine_structure_hypotheses.md)。状态needs mechanism evidence / limited prior-work check。
