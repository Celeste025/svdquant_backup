# SVDQuant叠加官方SageAttention3：E078

## 动机与假设

用户提出检查两种量化是否因特性相互作用而产生更大误差。区别两个问题：组合是否比只量化主干更差；组合是否比两种单独误差的实际向量之和还差。后者才支持这里定义的“净额外放大”。不同误差不必正交，因此不能简单把两个SSE数字相加作为独立叠加基准。

此前E006是局部QKV投影＋FlashInfer FP4；本次确实使用官方SageAttention3及用户冻结的完整SVDQuant主干，不拿旧结果替代。

## 实验

固定H3两个完整teacher状态：p30/s05、p36/s14。四组各2次完整前向：

- B0：原始BF16＋原SDPA。
- BA：原始BF16＋官方SageAttention3。
- S0：正式SVDQuant baseline＋原SDPA。
- SA：同一SVDQuant baseline＋官方SageAttention3。

SVDQuant保持200个主block线性层、NVFP4 W4A4、rank32、原smooth；50个主block有效长段使用Sage3，两个refiner及各自独立padding短段保持原SDPA。QK norm/RoPE与原scale不变。Sage3保持per_block_mean=True、非因果及真实序列长度，未裁掉尾token；官方preprocess原地修改K，因此wrapper给它独立副本。

B0/S0共四次完整raw输出均逐byte重现历史。BA/SA每次实际调用50次官方fp4attn_cuda.fwd，共200次，另一次非128整除长度257的smoke；head_dim128没有触发官方大head fallback。每次原始SDPA调用数分别102/52，目标native linear调用数0/200，disk_loads=0。

## 结果

以下是完整DiT的**视频预测SSE**，相对同一BF16输出；不是解码视频距离、画质或MJ分数。

|状态|仅SVDQuant|仅Sage3|组合|组合/仅SVD|
|---|---:|---:|---:|---:|
|p30/s05|346491.53|109916.57|402637.19|1.1620|
|p36/s14|53315.13|27475.47|71407.39|1.3393|

**组合确实比只用SVDQuant增加了视频预测误差，分别+16.2%和+33.9%。但本次没有观察到视频端的净额外放大。**

令eS=S0−B0、eA=BA−B0、eSA=SA−B0，则：

|状态|组合／独立误差向量和（能量比）|SVD模型／BF16模型的attention切换响应（能量比）|
|---|---:|---:|
|p30/s05|0.7519|1.3784|
|p36/s14|0.7122|1.3454|

组合相对实际误差向量和的SSE反而低24.8%和28.8%。另一方面，同样切换到Sage3，在SVD模型上造成的输出变化能量比在BF16模型上大37.8%和34.5%。这两点不矛盾：新增变化的大小与其相对原误差的方向共同决定最终SSE。非加性交互I=SA−S0−BA+B0的能量较大，仍不能单凭它断言有害。

**音频读出不同：** 组合/仅SVD为1.2577、1.5585，净放大比分别1.1277、1.0123。一个状态有12.8%额外放大，另一个接近独立叠加；两个状态不足以建立稳定的跨模态机制，也没有评价音质。

## 结论与下一步

现阶段可以回答：官方Sage3叠加SVDQuant后，两个状态的最终预测误差都增加；但“组合产生超出独立误差叠加的额外视频损伤”不获支持。更值得追踪的线索是模型量化后对attention切换的响应能量增加，而非直接声称二者相互放大量化噪声。

完整网络四角包含上游QKV分布变化、后续层传播和多层反馈，**尚不能将这个35%–38%的响应变化归因于Q/K中心化、P量化、V量化或某个局部算子**。若继续定位，最有决策价值的是固定同一attention输入/后缀的交叉干预，区分attention自身变难量化与量化后缀响应变化；不直接做层/步排列组合或宣称新方法。

两点来自不同prompt和时间步，均为既有诊断状态，探索样本很少；不外推所有H3视频，不称画质下降16%或34%。本轮0新视频/训练/校准。官方[实现说明](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell)本就没有保证对所有视频模型无损；[论文](https://arxiv.org/abs/2505.11594)中的FP4注意力设计提供机制候选，不是本地组合结果的证明。

## 资源、接入和追溯

官方代码commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5，原算法/setup无修改，源码与两个扩展SHA记录于run.json。独立DATA1目录构建，通过脚本内路径导入，未升级共享Python环境。

接入失败与修复：官方setup下载CUTLASS停滞，终止本任务clone；git自行清理未完成目录，仅失败日志保留。使用本机FlashInfer分发的CUTLASS4.5.0头文件。系统nvcc13.0与torch cu128不匹配，因此从NVIDIA官方redistrib下载nvcc/cudart/cccl 12.8.1，共约81MB且SHA全验，在DATA1独立解包；通过CPATH补入已有同cu128环境的cuBLAS/cuSPARSE/cuSOLVER头文件。匹配构建成功，没有绕过版本检查。各次日志单独保留，不把接入成本混入69.63秒模型实验时间。

E078运行含加载/替换/输出保存69.627秒，8完整DiT，峰allocated43.26GiB。独立CPU/NumPy重读8份raw张量、检查SHA并复算全部指标，0.169秒、0CUDA，最大相对差2.22e−16。GPU0已释放。事后同输入比对旧E016 FlashInfer组合输出并非逐byte一致，记录为补充，不据此比较后端速度/质量。

- [预设协议](../06_experiments/E078_sage3_interaction_plan.md)
- [运行与完整指标](../../results/research/E078/run.json) / [独立复算](../../results/research/E078/independent.json)
- [执行脚本](../../scripts/research/probe_h3_sage3_interaction.py) / [补充后端输出对照](../../results/research/E078/flashinfer_comparison.json)
- 模型日志：`results/logs/E078_sage3.log`；编译日志：`results/logs/sage3_build*.log`。
- 原始输出：`/data1/models/svdquant-wjq/research/20261004/E078/`。
