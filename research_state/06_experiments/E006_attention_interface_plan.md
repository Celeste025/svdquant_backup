# E006 — 原生W4A4 projection与FP4 attention的接口诊断

2026-10-02，状态：只进入环境可行性核查；方法假设尚未成立。

## 问题与先例

见[第二阶段候选P2-B](../02_problems/phase2_candidates.md)。不同近似的叠加、SVDQuant+Sage、联合校准本身已有先例。本实验仅检查projection扰动是否改变attention量化网格并产生可定位的交互，不把一般softmax非线性包装成新机制。

## 第一门槛：原生算子可用

- 使用FlashInfer官方0.7.0.post1的SM120 NVFP4 attention接口；固定版本、保存本机源码哈希。
- 在数据盘创建单独venv，继承已有torch2.11.0+cu128；不改变E004/E005运行环境。安装和JIT缓存均放数据盘。
- 安装/编译墙钟预算60分钟；若无法可靠调用，park此问题，不展开一天的kernel工程。
- 先做BF16输入、native quantize+FP4 attention的小shape smoke（非模型证据），覆盖非整128长度、全零输入、两个独立sequence以及head_dim128。与torchSDPA比并确认有限值、真实SM120 FP4 kernel名称、dtype和scale语义。零输入必须输出零；不能用NaN绕过为精度结果。
- 现有Sage安装实际为1.0.6 INT8-QK/FP16-PV Triton，不能冒称FP4 attention，故不用于这个机制实验。

## 第二门槛：固定最小机制对照（待第一门槛通过后冻结详细契约）

真实H3或Wan输入上的2×2：BF16/native W4A4 projection × BF16/native FP4 attention。同一QK norm、RoPE、mask、输入token，计算张量交互I=O11-O10-O01+O00。拟复用固定BF16-QKV的scales进行counterfactual，并记录clipping；若API无法表达只作为软件诊断，不声称原生counterfactual。具体模型、blocks、prompts和steps须在看factorial结果前写定。

若交互/总误差能量中位数<0.1，或没有实际损伤，或固定scale/既有V smoothing与ScaleSearch足以解释，则停止。阳性还需BF16 continuation后仍存在，才允许提出更具体机制。无自由rollout、无新颖性或质量收益结论。

官方来源：[SM120算子源码](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py)、[安装文档](https://docs.flashinfer.ai/installation.html)。

## 2026-10-02 冻结的真实模型协议（尚未看模型因子实验结果）

前置门槛已通过：native SM120 FP4 attention四case smoke；固定scale packer七类native byte parity（随机非整块、零输入、精确中点、负零、scale下溢）。量化输入准备使用慢速reference helper，不报告性能。

### 固定样本与实现

- 模型MiniMax-H3 pruned；blocks **0、24、48**；prompt **1、20**；steps **0、19**，共12个block-case。全为旧PTQ校准数据，不能称heldout。
- 完整BF16 teacher前向捕获三个block真实输入。两种projection只有目标block的qkv_proj不同：原BF16 vs **native NVFP4 SVDQuant**（旧rank32状态，正确高精度低秩输入，RNE、E4M3每16通道scale+FP32 tensor global）。out_proj、fc1、fc2均BF16；不是全模型W4A4，也不是改变全部linear。
- T0/T1由原模型QK RMSNorm和RoPE产生，捕获实际attention入口QKV，不重写其数学。按原cu_seqlens逐段调用；每段传真实unpadded_k_len，防止原生API padding变为有效key。main sequence内不重排模态。
- 五臂：O00=SDPA(T0)，O10=SDPA(T1)，O01=nativeFP4(T0)，O11=nativeFP4(T1)，O1F=当前T1的proc/correction + 用T0 QKV scales重编码后nativeFP4。O均为out_proj之前的attention输出；真实block endpoint作为次要结果。
- 每一个真实segment的T0/T1重新编码自身native scales必须在三个codes、三个scales、FP32correction全部byte一致，否则停止该case因果解释。O11/O1F的correction必须相同。所有非有限值立即记录失败。
- 固定scale统计包括API填充位置，必须明确标注；不能将padding零scale当真实token下溢。主误差按原video位置计算；text/audio/有效非pad总体分别报告。pad不能进入主统计。

### 指标与进入continuation的门槛

FP32张量记 e10=O10−O00、e01=O01−O00、e11=O11−O00、e1F=O1F−O00；I=e11−e10−e01。保存能量及带符号交叉项，验证加法恒等式。不以四个NMSE相减替代I。

按12个case的**video行**计算（all/text/audio为次要完整报告）：

1. 交互量级 `||I||²/||e11||²` 中位数须≥0.1；低于则停止交互主导路线。
2. 实际损伤 `||e11||²/||e10+e01||²` 中位数须≥1.1，且至少8/12例>1。若大交互主要抵消误差，则不能称有害组合机制。
3. 固定QKV scales须使最终attention误差 `||e1F||²/||e11||²` 中位数≤0.9、至少8/12例改善，且降低 `||I_F||²` 中位数，才继续“QKV scale切换是可干预原因”的当前路线。否则该具体原因停止；不能含混转而声称P grid已被证明。

这些是探索阶段资源分配门槛，不是统计显著性标准。三个门槛都过，才在**预定block24、p1/p20×step0/19**做其余BF16网络的continuation；不从12例挑最大收益。continuation主目标video输出同样须median≥10%收益、至少3/4例改善，否则停止。阳性之后仍须ScaleSearch/V smoothing等强先例对照、真正heldout与自由生成；不能凭oracle teacher scales命名新方法。

预算：单GPU0，60分钟硬timeout；部分结果逐case保存。只运行这一组，不扫blocks/权重/位宽；发现实现问题只修契约并保留失败记录。源码、输入、state、模型、helper与本计划哈希入结果。主runner为 `scripts/research/probe_h3_attention_interface.py`。
