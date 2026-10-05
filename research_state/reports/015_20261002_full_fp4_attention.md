# 015：完整 FP4 attention 带来 1.33–1.37× DiT 加速，但保真存在取舍

2026-10-02，E016。固定原生 SVD 线性层后，接入两种官方 FP4 attention 均有实际整模收益。global-mean 更快、显存较省，block-mean 在后两个状态上的输出误差较小；不存在覆盖本轮全部状态的单一优胜配置。**这是成熟技术的完整基线，尚无论文核心贡献或生成质量等价结论。**

## 实际做了什么

使用 E014 的三个共同状态与原 H3 权重，固定200个原生 SVD linear，只改变50个主 attention 的有效长段；padding段和两个refiner维持原BF16语义。两种低位模式均使用自己每层的真实QKV重新计算均值、scales和correction，完整在线预处理计入延迟。global-mean仍做centering，并非去掉centering。

新环境的原BF16模型和SVD＋BF16 attention均先在三个状态上逐byte复现E014输出。随后两种FP4模式各跑三个状态。三个SVD配置在同一GPU5分别独立进程测1次warmup、3次repeat和1次profile；共27次完整DiT，323.6秒完成。没有追加输入、训练或解码视频。

所有配置共享相同的main cu边界缓存和调用后核验，避免将减少GPU元数据同步的普通收益算给低位attention。全部计时输出与自身evaluate逐byte一致。

## 同轮性能与整模显存

下表固定p1/step0，分辨率对应原576×1024、124帧设置；数字是一次完整DiT，不含生成循环、文本编码或VAE。加速比只使用本轮同环境三个SVD臂。

| Attention配置 | 中位延迟 | 对BF16 attention加速 | 稳态峰值allocated | 稳态峰值reserved |
|---|---:|---:|---:|---:|
| BF16 | 6.791 s | 1.000× | 16.806 GiB | 30.232 GiB |
| FP4 block-mean | 5.124 s | 1.325× | 17.713 GiB | 32.553 GiB |
| FP4 global-mean | 4.968 s | 1.367× | 16.806 GiB | 31.734 GiB |

三个repeat依次为：BF16 6.780/6.791/6.803秒，block-mean 5.131/5.124/5.123秒，global-mean 4.978/4.968/4.966秒。额外核验的中位开销均小于0.6毫秒，包含核验的host中位分别为6.7913/5.1243/4.9685秒。

三个配置的模型storage均11.946 GiB，启动峰值均37.604 GiB。block-mean相对global-mean增加约0.907 GiB稳态allocated峰值，但后者reserved仍高于BF16 attention；不能把单个correction张量缩小等同于整模容量按相同比例改善，更不能外推长视频OOM。

独立profile中，BF16 attention核心耗时3.340秒；两低位核心分别1.047/1.031秒，另外仍需在线均值处理、cast、correction和QKV packing。单独的correction matmul为0.174/0.036秒，不能把核心kernel约3倍的收益等同于整模加速。profile含诊断finite检查，已独立分离，未用于正式延迟分母。[实际kernel归因](../../results/research/E016/E016_profile_attribution.md)

## 全部固定状态的输出误差

下表为原始DiT输出相对本轮完整BF16模型的NMSE，按模态分别归一化。它不是视频质量分数，也不能据此评价音画同步。

| 状态 / 模态 | SVD＋BF16 attention | ＋FP4 block-mean | ＋FP4 global-mean |
|---|---:|---:|---:|
| p1/s0 视频 | 0.023338 | 0.030796 | 0.027653 |
| p1/s0 音频 | 0.041689 | 0.065641 | 0.049370 |
| p30/s5 视频 | 0.065412 | 0.073919 | 0.077019 |
| p30/s5 音频 | 0.102239 | 0.117747 | 0.124212 |
| p36/s14 视频 | 0.012059 | 0.016821 | 0.018314 |
| p36/s14 音频 | 0.014027 | 0.020106 | 0.025090 |

两低位模式在全部六个端点上都增加了误差。global-mean在p1更准，block-mean在p30/p36更准；不能只用旧校准p1选出全局配置。三个状态均已经看过，且两种模式改变的是官方完整数值配方，本轮不能把差异独立归因某一均值统计或当作泛化机制。

## 结论与下一步

保留BF16 attention精度参考以及两种FP4成本—误差配置，后续方法必须与这些完整实现比较。E006局部接口假设的阴性结果并不否定现成FP4 attention的整模部署价值；本轮已补上这一缺口。

下一阶段先依据完整生成质量决定这些基线的使用边界。最新官方源码核查确认：correction的消费已融合、compact布局已合入，公开的Q分块缓解也已存在；不能将这些现成方案包装成新意。此次未发现同数值合同、已合入的SM120即时生成correction开关，但这不等于该方向已有研究价值。[实现近邻](../01_literature/fp4_attention_correction_paths.md)。只有出现明确、未被已有方法解决的成本—质量限制，才进入新方法实验；不将组合SVD与已有FP4 attention视为创新。

[预注册协议](../06_experiments/E016_h3_full_attention_plan.md) · [运行记录](../../results/research/E016/launcher.json) · [CPU前置检查](../../results/research/E016/E016_check_bf16_original.json)。

独立CPU复算已完成：116文件、两级历史重放、真实输入/双输出、全部计时输出及实际kernel均通过；未初始化CUDA。三份独立profile均确认200个SM120 E2M1 linear GEMM；低位两臂另各有50个真实FP4 attention kernel及52次BF16 SDPA。完整FP64误差与相对SVD输出的偏移保存在[独立汇总](../../results/research/E016/independent_summary.json)和[精简表](../../results/research/E016/independent_table.json)，summary SHA `2a4856da0037f54ad98dd5b7c8177633e6781ffdf2a05a1f2d358de1c9441ce7`。
