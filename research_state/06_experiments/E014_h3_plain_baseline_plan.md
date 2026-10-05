# E014：完整普通NVFP4对照

2026-10-02，GPU前预注册。模式为exploration/强基线核查，不是新方法证据。当前完整native H3只有旧SVDQuant配方，plain只做过局部干预；这不足以决定完整部署中smooth+LR开销是否有对应的数值收益。**不再要求先发现可见的动作损伤，才允许研究性能与精度取舍。**

## 问题、假设与决策

问题：在相同主层覆盖、NVFP4舍入、attention和输入下，旧smooth+rank32 LR配方相对原始权重直接量化，获得了多少完整模型数值收益，付出多少真实开销？开放假设是其收益可能不一致，不预设plain胜出。

该对照只决定后续研究的基础配方与成本下界。SVDQuant原论文已含NVFP4有无LR消融，收益随GPTQ等配方变化；JustQuant已有通过训练移除额外算子的路线。因而“NVFP4可能不需要SVD”或“把复杂度搬到训练”都不能据此作为新贡献。文献说明见phase4_process_critique.md。

## 最小有效配置

同一原BF16 Comfy pruned H3、200个main-block linears，8个refiner linears和其余张量保持原dtype（包含原FP32 time table/RoPE，不统转BF16）。三臂为BF16、现有完整SVD配方、plain_h3_recipe。**plain从原始BF16 W重新量化，绝不从已减去BA的residual权重删掉LR。** 无smooth/无LR，原bias不变。两量化臂都使用现H3 group16、signed E2M1 ties-lower、E4M3 RNE和FP32 global；沿用已验证fastpacker及SF0合法域。不是全E2M1标准RNE的最优PTQ基线。

量化改变smooth与LR整个配方，不能把差异单独归因LR；激活分布随各自网络变化。新plain module保留主支packing+scaled_mm，无虚假零rank矩阵乘、无乘1和加0的伪开销。原文件全部不改。

固定三个输入及原文件SHA见manifest：E009原p1/step0完整raw DiT输入；E010 BF16轨迹p30/step5、p36/step14，通过原model_fn构造。前者用于与既有profile相连；后两者避免只看一个旧校准点，但已经生成过，不称未见质量测试集。每臂每输入一次完整前向，保存全部video/audio输出、真实DiT输入及packed metadata签名。BF16须与原E009输出、E010 noise_pred及原raw DiT SHA精确重放；原raw DiT与velocity符号/布局不同，不直接混比。

全部200个plain权重必须decode数值等于独立旧nvfp4_qdq(原W)，保存最大误差、原W/packet来源、零符号例外；不要求等于原BF16权重。代表层native主支与同codes QDQ参考作有限算术smoke，目的是查格式错误，不以小误差阈值证明完整模型等价。真实activation使用fastpacker既有checks。每次完整DiT记录102个BF16 SDPA、量化臂200个FP4 GEMM、零磁盘权重加载；未量化部分identity保留。

## 性能与数值评价

三臂分别独立进程，在同一空闲GPU5串行测量E009原形状。1次warmup、3次repeat、1次独立profile；各次输出必须与本轮evaluate参考SHA全等。同步计时覆盖完整resident DiT和在线pack，不含输出拷贝/hash、模型加载、TE、sampler或VAE；计时内去掉诊断hook；fastpacker仍收集全部flags，context退出时统一验证并在主forward计时之外单列，也报告含检查时间。不得退回每层同步的standalone模式或漏查。profile独立，沿用E009已经验证的策略；旧6.787s使用含检查口径，本轮只比较新测三臂的相同口径，不混用旧值。报告latency全部值/中位数、模型unique storage、steady峰值allocated/reserved、启动峰值；不将profile时间当正式计时。

独立CPU以原始输出FP64累计，逐case、逐video/audio给error energy、参考energy、NMSE和plain/SVD比值；全保留，不因某模态好看只报它。共同case输入SHA与实际调用签名一致才可比较。**这些是固定状态的速度/velocity保真指标，不是视频质量、运动能力或完整生成加速。**

决策：若plain在全部三个状态的两模态NMSE均不大于SVD，且匹配延迟更小，则替换后续数值/性能基础配方，质量主张仍需另行实际生成评测。若存在取舍，报告取舍、保留两臂，不强行合并为赢家。若SVD明显降低误差，则保留为保真基线，plain提供实测成本下界。没有人为10%改善门槛；本轮不训练、调参、追加输入或生成新视频。

## 资源与停止

首次GPU起45分钟总墙钟，最多30个完整DiT调用；200层streaming导出和有限单层smoke另计但包括总墙钟。export用空闲GPU0，三臂evaluate/bench用GPU5串行；启动前复查不干扰其他用户。当前及历史allocated≤60GiB。若来源、重放、数值合法性、执行合同或资源检查失败即停止，保留失败，不重置截止时间。CPU先验证真实输入结构、scalar hash与2D embedding；所有运行源及协议在GPU前冻结，大文件留/data1。

产物：新plain模块及export、三输入runner、独立汇总、阶段报告。实验结果无论胜负，都不能把一个基线对照升级为顶会贡献。后续仍须提出有实质残余的问题及相对于强基线的可复现贡献。
