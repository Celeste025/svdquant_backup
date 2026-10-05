# E072 — 可交付强基线的实施草案

2026-10-04。只读设计；本文件所列 GPU 预算尚未选定或启动；不修改 E069–E071 冻结源/产物。用户研究授权持续存在，不增加审批步骤。按 `decision-value-experiment-planner` 的 reviewer-defense/completeness 模式处理：补足成熟基线，**不是新方法实验**。E071 runner 已 complete/exit0，96.025 秒，其中 QKV 精确 SVD 52.573 秒、峰44.795 GiB、数据6.670 GB；独立checker首跑complete，24.277667秒，SHA `e86b007f147f6fe957bcf96cb82cddf62b76f8acaefa69604d779ea905bd57d1`，后续实施绑定此结果。

## 推荐与决策价值

**优先交付可实际生成视频的公开 native checkpoint 外部基线；不要默认启动全200层本地重校准。** 另一 agent 正核实公开 Diffusers/nunchaku_lite 路径，本草案不重复其来源审计。若能固定 checkpoint/运行库版本，确认实际 NVFP4 与低秩执行、底模/噪声/embedding/调度/decoder 身份，则下一任务直接做既有任务条件下的完整推理比较，不再派生入口 smoke。它应标为“公开发布配方的外部基线”，而非本地200层等预算消融；额外50 AWQ、12 refiner线性、QKV拆分与是否共享 A、实际位宽/存储/成本必须披露。312对200的数量本身不能证明底模不同：主块 fused QKV 拆分可把200变300，再加refiner12。只在这些必要身份/运行条件无法建立时转本地路线，不因README的旧私有包描述单独否定公开后端。

收束时root补充：公开Diffusers后端确有NVFP4 A4路径，但依赖的 `rootonchair/nunchaku-lite-kernels` v2 匿名读取返回401，本地没有标准HF凭据；因此**目前是成品执行依赖未取得，不是已证明仓库私有，也不是已经可运行**。本轮不承诺启动公开推理、12 GPU小时校准或完整模型校准。

本地路线的最小有效产物是 **block0四个目标层完成全部39个平滑候选、随后各自最多100次/原规则early-stop的LR校准，并导出可重载的四层 native checkpoint**。不能将其称为整模强基线；其价值是交付第一份真实校准结果并测出完整成本，决定是否承担其余49块。E072不再以“再证明一个接口”作为终点。

问题：默认候选族、LR-aware评分、精确分解与本地部署合同能否在合理成本内组成实际基线？竞争假设是可以完成可复用校准产物，或校准代价远高于直接复用公开成品的价值。**继续/停止按正确性、完成度和成本决定，不要求首块 SSE 或视频预先改善某个百分比。**

## 能复用什么；还缺什么语义

- 已有：E069固定上游commit `69f3473f5e1c1504bae35cc50c7858ef900a9b17`及39候选见证；E070完整sequence索引桥及默认full FP64 SVD；E071真实 `SmoothCalibrator.calibrate/reset/ask/score/tell`、native安装/恢复、BF16部署scale、LR未量化输入、导出重载。既有64份raw校准PT共941,086,424字节，8 prompt为1/11/20/25/46/48/105/116，steps为0/3/5/8/11/14/16/19；无需重新生成校准视频或TE。E070两份attention缓存可作已有身份见证，**不能冒充64状态、四层或LR阶段缓存**。
- 平滑：GridSearch/AbsMax/alpha=.5/beta=-2/grid20、39项保持原顺序；每项仅一次weight-init rank32精确SVD，不是39×100嵌套。raw FP32 scale与一次BF16部署cast分别记录；恒等候选必须允许，不能沿用E071“非全1”门槛。原Layer评分平局用`<=`，后到的相等候选可替换best；不能复用E065 first-tie策略。完整64 case逐例、BF16差→FP32平方和，仍由真实库评分/tell执行。
- 目标映射：H3 fused qkv整体共享一个rank32分解，评价真实attention输出；out_proj、fc1、fc2评价各自真实线性输出，保留fc1的融合gate/up行布局。此为对应上游普通串行块的H3映射；不偷偷换成统一block loss或各自独立Q/K/V rank32。
- 后续LR：实际 `QuantLowRankCalibrator` **断言 needs_quant=true**，不能照搬E071的全部quantizer=None。须提供明确native权重量化适配器并保留其真实 `_reset/_ask/_tell`：compensate=false，初始qw=0，首项分解Ws，后项分解Ws−最新qw，随后Q(Ws−BA)更新最新qw；严格变差才early-stop，平局更新best，最多100项。精确FP64 `LowRankBranch`需要隔离使用固定上游实现，不能不知不觉落回vendored随机SVD。native安装只适配执行边界，选中Q/A/B必须来自同一best项，实际latest-Q链另存身份。
- 阶段状态：上游是全模型smooth→LR/weight→activation阶段；loader在yield当前层前已算下一层输入，**不等于选完native块后再把student输出作为下一块校准输入**。首块先完成四层smooth并保留BF16平滑模型，再单独生成LR阶段缓存；不能一直复用初始BF16输入。E072可将首块LR提前完成，因为其前缀不依赖未来块，但推广时不得按“块0全量化→块1全量化”替换全局阶段语义。

主源：[固定默认配置](../../results/research/E069/upstream/examples/diffusion/configs/svdquant/__default__.yaml)、[平滑目标/阶段](../../results/research/E069/upstream/deepcompressor/app/diffusion/quant/smooth.py)、[LR递推/停止](../../results/research/E069/upstream/deepcompressor/calib/lowrank.py)、[LR阶段入口](../../results/research/E069/upstream/deepcompressor/app/diffusion/quant/weight.py)。缓存yield顺序来自本地[实际loader](../../third_party/deepcompressor/deepcompressor/dataset/cache.py:397)，其来源身份与上游固定12文件的覆盖边界需分别记录。

## 一次实际校准任务与缓存合同

1. CPU前置绑定全部64 raw PT/manifest、源权重/四层布局、39项顺序、实际M与cu段数、源E071及独立结果；元数据/静态小数组控制随正式任务执行，不另立GPU smoke。64 case按预先冻结顺序求和，所有模态与padding处理沿原完整tensor；不换成8case或512行。
2. 64次原BF16前缀捕获block0完整hidden/kwargs，随后一次64-case原块扫描取得四个真实输入。每一校准阶段的输入固定；候选只替换正在校准的目标，邻居保持该阶段对应的BF16快照。四层可在两个GPU上用隔离模块副本分工，先保存正确邻居快照再并行，不能让别的worker的临时candidate进入QKV attention评分。完成四层smooth后，从block边界缓存运行64次全四层已平滑BF16块，重新捕获LR输入。无需完整DiT。
3. **64-case参考输出必须放CPU。** E071两个独立[1,1]索引不会改变`_parse_ipts`前后样本数，因此opts_device可留None；放大到64个fc1输出会占约76.8 GiB GPU，不能照搬。可用真实TensorCache保存一个[64,1]索引batch，按sample_batch_size=1/sample_size=-1实际repartition为64个[1,1]；原代码会因1→64启用`outputs_device='cpu'`。CPU前置核完整索引顺序、数量和opts_device，评分输出仍为原GPU BF16，保留库内GPU减法/归约。原参考输出D2H须保证完成后才消费；不能为省GPU内存改成CPU评分或FP32 wrapper返回而不披露。
4. CPU mmap按case保存完整输入，GPU每次仅当前case。按M≈22464估计，四层输入合计约86.4 GiB/阶段；block边界约14.4 GiB，最大单层CPU参考约76.8 GiB。CPU前置用实际ΣM计算最终预算，不把该近似当上界。建议新产物上限256 GiB、host RSS≤256 GiB、GPU峰<60 GiB：两阶段输入约172.8 GiB＋边界＋所有候选紧凑packet/A/B（上限约30–35 GiB）可以保留；参考输出按当前层在CPU内存复用，候选全M输出不逐项落盘。不复制E071每候选6.67 GB审计包，也不落全模型200×64激活。
5. 每候选保留64-case标量分数、scale/A/B/packet与实际SVD输入/最新Q的hash、时间/峰值/停止原因；每层保存完整smooth胜者与LR胜者/恢复状态。任务最后真正重载四层导出并运行预先固定两个完整block case，保存完整输出与原candidate回放byte证据、计数和无残留hook检查；这是已校准产物验收，不另开选层/选seed接口实验。其余候选全输出未保存，独立checker只核可保存的数值与选择/递推身份，不能声称重演所有native评分。

## 成本承诺、终点和停止

确定工作量：四层smooth共156次分解/9984次候选模块评价；LR最多400次分解/25600次候选模块评价，另有基准、缓存与最终验收。以E071单QKV 52.573秒仅作尺度示意，156次约2.28 GPU小时，连400次上限约8.12 GPU小时，**尚未含64状态评分，且out/fc2等形状未实测**；这不是精确工期。若未来决定选本地路线，可考虑累计12 GPU小时、两卡时各最多6小时的备选资源上限；**它不是root已选方案或本轮启动建议**。分配任务前需保证依赖阶段可完成；不得把超时改叫early-stop或自动续预算。若余量已不足完整下一层，可在完整层边界结束并保存partial，不声称四层完成。

全200层的机械尺度则为7800次smooth分解约114 GPU小时，加LR最大20000次后约406 GPU小时，另有百万级完整case模块评价。不能仅因“补官方基线”就承担几百GPU小时；只有完整首块成本、后续可用预算、公开成品不足的具体原因与论文比较收益都明确时，才据成本决定是否执行其余49块。允许两卡分工、输出流式、相同BF16 scale/相同SVD输入的结果缓存等明确等价工程；原39项顺序与平局决策仍完整保留。不得把随机/低精度/reduced SVD未经验证称等价后替换。

有界备选是**精确分解的执行成本验证**，不是另一个模型接口smoke：复用E071已保存同一Ws及top32，预先只选一次 `torch.linalg.svd(Ws.double(), full_matrices=False, driver='gesvd')`，不跑模型、不扫driver、不重跑default参照。reduced仅去掉未使用的正交补，仍计算完整奇异值谱；driver改变后的实际速度和有限精度结果不能先验保证。若将其作为后续“等价工程”入口，数值门槛先冻结：FP64 top32重构相对差≤1e-10、Gram≤1e-8；对齐成对奇异向量符号后BF16 A/B、实际BF16 B@A残余和weight packet须逐byte匹配E071，另报告实际差值而非只比较谱。若只达到近似而非该严格部署门槛，不宣称等价、不自动替换。单点通过仍不是所有后续矩阵的逐byte证明；结合实际耗时与未被分解加速消除的64-case评分/I/O重新算资源，再决定是否值得首块完整校准。此备选旨在避免潜在百GPU小时的无效投入，仍属工程，未执行。

**接受与决策：** 公开路线可对齐即优先交付完整外部推理基线；本地路线只有四层39项完成、LR正常early-stop或到100、选中导出回放及独立证据通过，才叫block0校准完成。正确且成本可承担→保留产物、据实决定全模型预算；正确但代价不合理→停止本地扩展，优先公开成品；任一数值/计数/恢复/身份失败→保留失败与已完成产物，修实现，不判算法阴性。超时/OOM/磁盘上限→资源未完成，不缩case/grid/迭代后仍沿用原配方名称。

Necessary：完整候选族与真正LR停止、phase缓存、native数值与选中导出、实际成本。Supporting：固定两例block输出误差及旧baseline同输入描述。Cuttable：逐候选全输出、多余warmup/新shape smoke、重复E070/71。Future：200层完整导出、整模teacher读出、统一任务条件的完整视频/人评与效率。即便全模型完成，也应命名 **native-compatible H3 SVDQuant（200主线性、64-state、BF16 scale、signed-lower/two-level NVFP4适配）**；stock YAML的A尺度、额外INT4模块/覆盖、一般128样本默认与本地精度适配仍需公开，不能将适配基线或社区配方冒称逐byte官方复现。
