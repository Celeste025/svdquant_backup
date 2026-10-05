# E012：H3 最小条件编辑的有界数值筛查

2026-10-02，预注册于任何新模型输出之前。不是新方法、不是语义质量实验。E010已看过的p36只作开发诊断；DASH/GAMP等碰撞决定保持。目标仅判断是否存在值得进一步解释的强响应衰减，或立即停止这个具体解释。

## 固定输入与条件

使用E010原BF16 p36轨迹的step5、step14，video/audio的latent-before、各自实际timestep、原模型及原`model_fn`，CFG1。以scheduler消费的两路velocity为输出；使用E009验证的resident/native线性路径、torch BF16 SDPA。保留低秩分支精度和全部量化配置。两个时刻属于同一条轨迹，不是独立样本。

完整文本和SHA在`E012_h3_conditional_response_manifest.json`。在原p36上仅作以下替换：

| ID | 角色 | 精确替换 | 原文出现次数 |
|---|---|---|---:|
| original | 公共基准 | 不变 | — |
| direction | 目标编辑 | `from left to right` → `from right to left` | 1 |
| label_color | 第一普通编辑对照 | `gold text gleam` → `white text gleam` | 1 |
| pedestal_material | 第二普通编辑对照 | `stone pedestal` → `glass pedestal` | 2 |

原prompt中的其他方向、音频和叙事不变。四种条件均已用实际tokenizer预检为813 tokens；正式check必须再次验证token数、tags及真实packed布局一致，不截断或填充文本来制造相等。TE保持BF16，用原PromptEmbedder编码；original embedding必须与E010逐tensor SHA一致，否则停止。编码阶段不加载DiT/VAE。

## 分阶段预算与独立性

1. `check`：仅CPU，核对来源、tokenizer、cache、dtype、plan/manifest，必须CUDA未初始化。显式torch attention在导入DiffSynth前设置。
2. `prepare`：一次TE加载，编码4条件，保存原始embedding、token/tag与SHA。复用E010已完整hash的资产，核对文件stat及来源绑定；不重复下载。
3. `bf16`：两时刻×4条件×3输入（原点、下述plus/minus）=24次DiT，再各重复一次original原点，共**26次**。原点original的两模态velocity必须匹配E010缓存SHA。保存全部CPU原始输出、输入与metadata哈希及运行计数。
4. 完成全部teacher输出之后，依下述规则生成并冻结teacher选择JSON/hash。teacher gate不通过即停止，**不运行native，不追加提示或搜索对照**。
5. `native`：仅teacher gate通过后，original/direction/一个已选control，在两时刻原点共**6次**DiT。始终使用同一BF16缓存输入，不能使用E010 native自由轨迹。每次200原生GEMM/200packing合法检查、102 BF16 SDPA、零DiT磁盘加载。记录原始输出，独立CPU归约。

最多32次DiT；不生成视频、不加载VAE、不做校准或训练。GPU0单进程分阶段串行，每阶段前确认空闲。总墙钟上限25分钟，allocated峰值检查60GiB；这是检查点止损，不是防止每次分配的硬限制。所有数据在`/data1/models/svdquant-wjq/research/20261002/E012`。失败保留，不覆盖E009/E010冻结文件；源变更另开版本。

## 扰动、teacher gate和控制选择

主统计只看**video**，audio完整单列，不用于决定光照方向响应。固定每一模态展平坐标偶数为+、奇数为−，CPU BF16 `nextafter` 向相应无穷方向移动一个可表示值为plus；minus用相反方向。两个扰动都是同一个原点的一步邻值，不是连续两次更新。所有条件共享完全相同的两路扰动，保存实际相对能量、changed坐标数、SHA。非BF16或出现nonfinite则停止。

记`D_i(x)=B_i(x)−B_original(x)`，FP64 CPU归约。对每个时刻定义差分背景

`N_i=max(||D_i(x_plus)−D_i(x)||², ||D_i(x_minus)−D_i(x)||², 4||B_original_repeat−B_original||²)`。

另报单条件velocity变化能量；不能把这两个背景混称。要求target和所选control两时刻均满足`||D_i||²>0`且`||D_i||² ≥ 10 N_i`。背景恰为0时仍须非零响应，报告infinite SNR，不除0造NaN。该规则只检验对1ULP扰动的数值稳定性，**不证明BF16理解真实光照方向**。

按固定顺序label_color→pedestal_material选择第一个合格control；合格须两时刻的video teacher norm ratio `||D_control||/||D_direction||` 均在[0.5,2]，并满足上述SNR。选择只依teacher，必须在运行native前落盘与哈希绑定。无合格对照或target不稳定，status记`stopped_teacher_gate`；这是观察量/对照不足，不是量化机制阴性。

## native统计与止损

两模态分开报告每条件的BF16/native输出能量与绝对/NMSE误差。对每个编辑：`dF=B_i−B_0`、`dQ=Q_i−Q_0`，记录`g=<dQ,dF>/||dF||²`、`r²=||dQ−g*dF||²/||dF||²`、`||dQ||²/||dF||²`、绝对差分误差、差分NMSE、共同误差`||(e_i+e_0)/2||²`。核对`difference_NMSE=(g−1)²+r²`，不得将gain约1但正交误差大称作响应塌缩。

只有在**两个时刻均**满足以下数值条件时才记`candidate_requires_replication`：target `g<0.5`且响应能量比<0.5；control `0.8≤g≤1.2`且差分NMSE≤0.25；三个条件各自video输出NMSE≤0.1。上述是投入筛选门槛，不是显著性。其他结果记`stop_current_selective_attenuation_route`并解释失败项，不改报大相对误差为阳性。

即使通过，也仅支持这一原prompt的局部异常，不支持语义控制下降、视频质量、一般NVFP4固有问题或新方法。下一步必须是新预指定编辑的独立复现与真正decoded语义验证；不能直接训练已有的差分loss。若BF16控制语义本来不可靠，停止对应解释。

先前依据：[条件响应数学审计](../02_problems/conditional_response_screen.md)、[DASH/GAMP等碰撞](../01_literature/conditional_response_collision.md)、[E010报告](../reports/010_20261002_h3_heldout_results.md)。本次是一次有限异常排查，不撤销既有新损失KILL决定。
