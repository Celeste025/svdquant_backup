# H3 输出几何入口：同状态量化误差与普通 head 传递

2026-10-04。按用户指定，MiniMax-H3 为研究主体，Wan1.3B 仅作快速验证；不启动 Wan14B PTQ。本页仅做源码、checkpoint header 与既有小型输出的 CPU 核查，没有新模型前向、GPU 初始化或空间统计结论。

## 已核实的模型与配方

- 实际资产是 `/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`，40,225,724,176 bytes，532 个 tensor，共20,111,438,744个存储元素（含buffer）；为 Comfy pruned H3，不称完整33B模型。继承 E010 已记录 SHA `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4`，本轮仅读取header和stat，未重哈希40GB。
- 50主block、2 token-refiner；200个主linear采用native NVFP4，refiner和最终head保持原BF16计算。pruned实现把time embed改为1025×8的FP32表及相应AdaLN适配，不应从名称推断层数被裁掉。
- E010/E014实际视频配方：576×1024、124帧、24fps、20step、CFG1、video/audio flow shift12/3，音频32kHz双声道。20step不同于pipeline默认50step。`BasePipeline.cfg_guided_model_fn`在CFG1只执行positive branch，故Wan CFG误差放大不能直接迁来解释H3。

## 输出几何确实同属2×2重排，但行顺序不同

实际源码 `/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py` 与仓库vendored副本逐byte相同，SHA `bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80`。

最终RMSNorm和时间调制后，`final_layer.video_out`将5376维hidden投影为96维，即24通道×1×2×2。每条已有输出为 `[21312,96]`，按37×18×32个token重排到 `[1,24,37,36,64]`；空间phase对应输出行 **`4*c+2*p_h+p_w`**。Wan原入口采用phase-major顺序，不能复用其输出行索引。`model_fn`最终返回负号后的unpatchify结果；同状态误差协方差不受共同负号影响，phase均值必须注明采用raw-output还是velocity方向。

磁盘head W保存为F32 `[96,5376]`，但E014实际resident转换记录明确其计算W/bias为BF16；普通head对照必须先按实际BF16舍入W再在CPU以FP64做代数统计，不能用磁盘F32 W冒充实际head。head没有被本轮200目标linear量化。

## 现成资产够做什么

目录 `/data1/models/svdquant-wjq/research/20261002/E014/evaluate/{bf16,svd,plain}/` 各有 `e009_p001_s00.pt`、`e010_p030_s05.pt`、`e010_p036_s14.pt`。本轮在CPU读取全部9份，每份video raw output均为finite BF16 `[21312,96]`；每个case三臂保存的actual input signatures相同。p30/p36另保存实际velocity `[1,24,37,36,64]`，p1没有velocity但有完整raw output。E015继承前两状态进行下一步传播，不能当作新增独立文本或用于忽略原扰动来源。

这足以做同状态误差的空间描述性筛查，不用重新捕获整模。**没有保存最终head前的实际hidden差**，所以不能据输出伪逆声称测到了hidden误差，也不能从统计阳性直接证明head因果。三个case是三个prompt/step组合且已经看过，不可推导随时间变化或总体泛化。

## 单一最小问题与普通扰动对照

问题：**在CFG1的H3中，plain/SVD相对同状态BF16的video预测误差，在相同物理lag下是否依赖2×2起点phase；这种模式是否已被同一BF16输出head对普通token扰动的传递解释？** 这是H3实际几何入口，不是P006运动/音画事件路线，也不是已成立的新机制。

首先固定全部3状态×2误差臂；按每latent-time、每channel减空间总均值，分别统计零lag能量、lag `(0,1),(1,0),(1,1)`的四起点phase协方差，另报原phase均值；采用相同interior/support，不能把patch内距离1与patch外距离2比较。原始raw方向与velocity负号关系写入输出。

最便宜的头部几何零假设是投影输入处独立各向同性token噪声：未去均值时同token输出协方差为 `sigma² W_p W_q^T`，跨token为0。必须经实际BF16 W和同一unpatchify，按每时间片实际误差能量匹配；去空间均值对有限36×64网格的解析协方差影响也应计入。此对照不包含量化，也天然可产生phase差异，因此是必要的普通解释。

若它解释主要模式，停止把2×2结构本身当新发现；若不能解释，仍不能排除普通有色/各向异性hidden误差。下一层必要对照才是同状态真实head前hidden残差及其通道/空间统计保持的普通扰动，或保持实际输出谱的平稳surrogate；后者也只排除所选平稳解释。不能仅用已unpatchify latent上的同方差白噪声、宽泛高频能量或MJ涨分支持量化特有机制，更不由本轮统计直接开启新loss/decoder训练。

原先[Wan入口](wan_patch_phase_entry_audit.md)和[已完成近邻核查](../01_literature/patch_phase_geometry_nearest_work.md)的限制继续适用：普通子像素head的相位结构已知；本轮不扩大文献检索或新颖性声明。CPU预筛将编号E055，root冻结计划后才执行统计。
