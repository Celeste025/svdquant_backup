# E017：完整 FP4 attention 的有限配对生成

2026-10-02，GPU前固定协议。上一轮E016已完成27DiT、116文件独立核验及报告015，属于实际progress。当前没有新方法或论文核心贡献；本轮补齐两种官方attention配置的真实生成边界，不继续扩大基础设施或基线网格。

## 问题与固定比较

复用E010的p30/seed49771与p36/seed59526。它们已经看过，不再称未见测试集。沿用原20步、CFG1、video/audio shift12/3、576×1024、124帧24FPS、32kHz stereo；20步不是模型默认50步。文本embedding和两模态初始BF16 noise直接取已绑定E010 prepare，不重算TE。新两臂为E016已验证的固定SVD200 native linears＋官方block_mean/global_mean attention；各两完整自由生成，共80DiT。直接引用旧BF16与SVD+BF16-attention结果为对照，不重跑生成。

主比较是新两臂对原SVD（仅attention替换）；BF16为高精度参考。四臂各自递推，不输入teacher后续state。原pipeline.__call__、model_fn、双scheduler和step算法不改；实例级观察/placement只用于记录及在VAE前分阶段退出。两种FP4每层用各自真实QKV生成packet，50主有效长段低位、padding/refiner原BF16。继承E016实际kernel和双exact证据，不增加smoke/profiler或teacher能力门槛。

## 生成与解码检查

新check仅CPU：静态源SHA、原已full-SHA模型资产的size/mtime、prepare两个SHA/embedding/noise/tags、实际packing边界及两个完整schedule。不可把缺av的源枚举导入当模型问题；native env负责DiT，原recovered env具备av且负责VAE。

每步记录200原生GEMM、50FP4 attention、52BF16 SDPA、0disk和200fastpack合法域检查；router diagnostics=False，输出和新latent必须finite。保留真实DiT输入签名、raw输出SHA，以及全部80×2个模态step文件（before/velocity/after/timestep/sigma）、final tensors和完整链记录。CPU独立复算所有原step与连续链，不能把自由轨迹分歧当同输入量化误差。

两个denoise进程完成后，VAE-only进程按E010相同tile256/overlap64、原视频/音频VAE、H264+AAC设置解码四段；每个arm独立解码进程。保留PCM和全部视频，检查帧数/尺寸/fps/audio stream。旧四媒体直接引用并核SHA与参数。不以启动/落盘/诊断总时间或旧轮时间拼出生成加速比；固定DiT成本引用E016并注明范围。

## 评价与解释边界

同一次VisionReward模型进程评价旧4＋新4共8视频，全部原29题/权重/完整prompt与官方取帧逻辑不变，共232题。原始token、第一token解码、完整答复、实际frame indices全部保留；strict与strip.casefold两种分数并列，未知回答使可用分数null。旧四视频的回答与历史逐题对照，若有差异单列环境/重评分变化，不归因新量化。

报告每prompt各模式分数及相对SVD的全部翻转题。29题不当独立样本算显著性，6帧辅助评分不等完整运动或音频评价。人工查四变体的固定时刻接触图与时序内容，保留明确观察位置；无实际听音不得声称音频质量。PCM可做基本finite/peak/长度检查，不能替代听感。没有teacher能力筛选、追加prompt/seed/改步数或按结果修改评价题。

质量若相近仅说明这两个样例未显示明确损伤；若恶化，定位可重复的具体表现后再判断残余问题。任何结果均不把SVD＋现成attention包装成新方法，也不从两个看过样例立质量等价/泛化/音画同步claim。完成本轮后停止扩展此基线网格，不因为要找新意继续增加无假设配置。

## 预算与产物

同GPU5各阶段串行，启动前stable-idle检查；80DiT上限、4视频/audio decode、232评价问答。共享30分钟wall deadline从首个GPU阶段开始不重置，模型当前/历史allocated≤60GiB；若实现/有限域/来源失败，保留partial，不扩预算自动重跑。无新模型安装/下载/训练。native Python沿E016，decode沿E010 recovered Python，评价沿E010 mjvideo环境。运行tmux、日志results/logs，大文件/data1/models/svdquant-wjq/research/20261002/E017，小结果results/research/E017。新源码CPU就绪后统一freeze；冻结E007–E016文件不修改。
