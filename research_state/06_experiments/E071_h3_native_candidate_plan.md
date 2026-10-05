# E071 — H3真实校准评分与native导出单候选一致性

2026-10-04。基线实施模式，非新方法探索/创新证据。E070两臂及独立核验已完成，不重复其模型检查。继承decision-value-experiment-planner：本实验仅解决完整强基线的必要部署接线风险。

## 问题、动机与决策

E070只验证无干预缓存入口，未验证库内评分的候选与原生NVFP4部署一致。具体竞争解释是：(H1)单一明确部署合同可贯穿真实reset/ask/score/tell及导出；(H2)缓存作用域、scale精度、重复平滑、LR输入或安装/恢复差异造成候选与部署不一致。成功允许推进多候选恢复/完整配方预算；失败保留定位证据，先修接线，不将其视为算法阴性。

## 最小有效设置

- 唯一目标 `blocks.0.attn.qkv_proj`，真实BF16权重[21504,5376]，固定源tensor SHA `1727c595c15e1c57f16832e961946975d1a6073f445e7b2e6ccd489554eac264`。复用E070 p1/s03、p20/s03完整attention输入，M22400/22464，x/RoPE/cu_seqlens/max_seqlen不变，不新跑prefix。
- 真实SmoothCalibrator.calibrate与继承reset/_reset/ask/_ask/_calibrate_wgts/tell/_tell，Weights、OutputsError、Layer、Manual alpha=beta=0.5、唯一AbsMax/AbsMax pair，sample_batch_size=1/sample_size=-1，degree2、develop_dtype FP32。实际population1、num_iters1，恰一次candidate ask/tell。
- eval_inputs只含[1,1]case索引；x_acts是两个完整variable-M张量，仅用于真实span，不把索引/kwargs当统计。wrapper允许baseline与candidate各按固定次序访问两case一次。原始输出保持BF16。
- w/x/y_quantizer=None；显式native executor覆写候选module安装/恢复，仅此部署边界适配。不能称库默认Quantizer/allow-LR hook路径已执行，不能用复制的自写搜索替代真实评分。AbsMax在真实_reset使用，CPUcheck实际验证跨度/候选维度，非仅AST。

## 固定候选数值与执行

真实ask产生FP32 scale，另存原值；执行与导出只使用一次cast得到的BF16 scale，必须finite/positive/非全1。这属于native-compatible格式适配，不称stock YAML逐byte复现。

一次 `Ws=BF16(W*scale_bf16)`，严格一次 `torch.linalg.svd(Ws.double())`（默认full_matrices/driver），rank32、weight-init：A=Vh[:32]、B=U[:,:32]*S[:32]，cast BF16；残余为BF16(Ws−BF16(B@A))。保留完整S、top32 FP64因子与BF16因子/scale及选定源身份；有限性、S非负降序、top32 Gram Frobenius≤1e-8，不把rank32残差当数值错误。使用既有native pack格式/有符号lower-tie/FP32 global×E4M3微块尺度，weight packet只构造一次；原bias保留。

实际NativeH3Linear.forward计算，LR必须读未量化BF16(x/scale)，而主支读它的完整tensor pack；不能叠加activation smoother/quant hooks再做一次平滑。记录完整候选/重载QKV输出、各case原生packet与LR输入证据，足以核输入同源与重载逐byte一致。用持久化export新建模块进行独立重载，不直接复用原候选对象。

真实库内score为BF16相减→FP32平方/归约，再两case累加，保留真实tell误差与best。另记录按保存attention输出计算的FP64误差，只作独立读出，不强令两种定义相等、不据此选择其他候选。核验同一评分定义的独立CPU归约采用预先相对容差2e-6（FP32并行求和差异），FP64读出采用1e-10；候选/导出张量身份仍严格byte-exact。

## 计数、接受条件、停止

真实calibrate：2次BF16 attention baseline，必须逐byte复现E070；2次candidate attention。校准结束后2次export重载QKV。合计4attention+2linear，8SDPA、4native scaled_mm、4activation packs、1精确SVD；0prefix/完整DiT/TE/VAE、新视频、拟合或网格扩展。不可用M512/QDQ线性替代完整真实attention/native执行。

接受要求：继承流程确实执行一次ask/tell；所有完整输入/kwargs身份不变；两candidate QKV与对应export重载输出、scale/A/B/weight packet逐byte一致；LR读入与pack前xs一致；原qkv对象、权重SHA及全部原hook恢复，外层finally兜底。源API/计数由执行收据记录，独立CPU不冒称重演模型。任何失败立即停止该run并保留，不放宽误差/更换case寻找通过，不追加模型调用作补救。

## 资源与复核

唯一启动，原始绝对deadline900秒，60GiB GPU峰上限，8GiB新产物。GPU0优先，启动前重查空闲；GPU6/7不动。复用E070环境和现有编译缓存，不安装依赖。named tmux＋外部timeout/supervisor，观察超时不重启。CPUcheck在无CUDA可见环境运行，执行源和本计划通过后冻结；修复新版本/新输出，不覆盖执行历史。

大产物DATA1/research/20261004/E071，报告results/research/E071。独立CPUchecker核验保存输出/kwargs/case、scale/因子与packet/导出身份、LR输入、两种评分归约及退出收据；不重复SVD或model forward。完整39候选、100/early-stop、200层、整模误差、实际视频质量与总PTQ成本仍未验证，不外推成功。E068人评待反馈，不重复询问；现有研究授权有效。
