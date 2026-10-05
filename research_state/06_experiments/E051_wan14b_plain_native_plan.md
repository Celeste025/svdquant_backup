# E051：原版14B plain原生NVFP4完整配对

模式：Exploration / baseline。当前没有论文claim，E049八例原版BF16参照已完整完成。

**问题与假设：** 在已经可读的14B参考上，真实NVFP4 plain主干量化是否可完整运行，出现何种实际视频损伤？该结果建立同模型未校准下界及原生实现入口，决定是否值得用更强SVDQuant修复、以及后续需解释的具体残余；不能以弱plain证明新方法有效。

**最小有效设置：** 固定四提示×两seed，共八例；复用E049实际FP32初噪、BF16正负embedding与CFG5/shift3/81帧480×832/50步UniPC，FP32 sampler/VAE、BF16 FlashSDPA。同一官方14B BF16 teacher的400main Linear逐层W→临时FP32→已有legacy原生packet，释放旧W；没有训练、平滑、低秩、额外attention量化或CFG调整，不整体cast FP32 globals。保持非目标参数及bias。每DiT实际400nativeGEMM/400activationpack/80SDPA，原生覆盖/scale dtype/有效SF均检查。历史RECIPE名带QAD/master不代表本实验训练或持有整模master。

**必要证据：** 八例完整50步/81帧、800DiT/320000native及activationpack/64000SDPA/400scheduler/8decode/0TE，保存实际输入/端点/计数、转换内存、固定预览和完整既有评价。与E049每例配对，不重生成或重评分BF16，不挑seed；root先读八预览再读新分数。有效完整nativeforward是实现验证，不能用meta检查替代；单层误差或少步图不替代整视频。

**预期与对应决策：** 若plain明显退化，完成强SVDQuant对照后才把残余立为机制问题；若plain已较接近参考，也需要强基线成本/质量评估，不能虚构量化收益。若运行失败，保留诊断、定位实现或资源错误，新版本修复，不把代码故障解释为算法失败。八例指标分歧与teacher语义局限须保留。

**预算及停止：** 四prompt worker各两seed，GPU0/1各串行两个worker；每worker5400秒，launcher12000秒，前两次空闲观测+立即检查，明确不在同卡叠启动。GPU2–5由E050使用，6/7他人任务不动。named tmux，任何失败停止本次owned树、保留不自动重试。完整生成后自动同GPU0/1评价（900秒）和CPU独立汇总（180秒）。采样成本含诊断/转换/解码，不宣称优化serving benchmark。

E050的真实14B匹配校准继续独立进行；本实验不替代SVD强基线，不新增loss或参数网格。
