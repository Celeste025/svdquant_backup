# E020→E022：QAD 强基线配方核对

2026-10-03；只读文献/源码核查，无 GPU、安装或运行源修改。结论：**E022 是合理的主权重 QAD 基线扩充，但还不能称已充分复现成熟 QAD。最具体、可低成本分离的缺项是外层量化尺度的训练政策；256 个固定 teacher 状态也尚不能排除数据覆盖与优化预算限制。** 保持正在运行的 E022 不变，先取得结果，不因已有 QAD 而停止主线。

本轮重点核对两个来源族：同模型 FastWan 官方发布，以及 NVIDIA QAD 论文/ModelOpt 官方实现；QUADS 仅复核与本任务的适用边界。“Scalable QAD”未找到独立题名/URL，不能当第四项已有证据。ModelOpt 后续已固定到 commit [`e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1`](https://github.com/NVIDIA/Model-Optimizer/commit/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1)（committer 时间 2026-10-02 17:48:14 UTC），于 2026-10-03 02:01:10 +08:00 重新获取并核对关键逻辑。[小范围源码快照](source_snapshots/modelopt_e68eb44_qad_recipe.json) 保存10个文件的完整文件 SHA256、字节数、永久 URL 及关键函数摘录。无法证明前次网页缓存的 `main` 就是此 commit，不作反推；以下源码链接和行号均以本次固定版本为准，也不冒充 NVIDIA 论文所有实验使用的历史版本。

## 我们实际做了什么

依据 [E022 计划](../06_experiments/E022_expanded_qad_plan.md)、[主权重模块](../../scripts/research/wan_mainweight_qad.py)、[训练器](../../scripts/research/train_wan_mainweight_qad_expanded.py)：从原 rCM BF16 W 重新开始，更新 300 个 Linear 的约 1.392B FP32 master；bias、其他参数和 BF16 attention 冻结，无 smooth/LR/rotation。32 文本×2 独立噪声×4 步产生 256 个固定 BF16 teacher 轨迹状态，4 新开发文本提供32状态。128 次 AdamW 更新、lr=3e-6、clip=1、weight_decay=0，每次四步各一个 micro，合计512次前后向，每状态访问两次。

目标是每状态 velocity NMSE，四步等权累积。训练输入来自 teacher 自己的轨迹，对学生自由生成分布而言是离线数据；不存在 rollout 更新、真实视频监督或 DMD。量化是本地 legacy-Wan：W/A 当前张量动态 global，group16，E4M3 中点向大值、E2M1 中点向较大有符号值；QDQ 真值输出 BF16，反向 identity STE。它不是标准 RNE 配方的逐字节复现。E020 的训练下降、开发上升只能说明该小配方的泛化/优化不足，不能据此归因于某个量化机制。

## 直接先前工作说明什么

| 来源 | 已核查配方与可复用部分 | 对 E022 的含义与边界 |
|---|---|---|
| [FastWan-QAD 官方发布](https://haoailab.com/blogs/fastwan-qad/) / [1.3B 模型卡](https://huggingface.co/FastVideo/FastWan-QAD-1.3B) | 同为 Wan2.1-1.3B；先做目标精度匹配的量化微调，再做量化感知 DMD 到3步。旗舰版本用 Mixkit 真实视频，另两版用 Wan14B 合成数据；公开 NVFP4 主层与低位 attention checkpoint。 | 最直接的现成质量/部署参照，不能绕过它再声称“首个可用 Wan W4A4”。但它是原 Wan→3步、81帧、低位 attention、Tiny VAE；我们的 rCM4步/77帧/BF16 attention 不相同。已读模型卡的 Training 仍是待补充，未公开可核的训练样本数、LR、总更新数、外层尺度/初始化细节；不能捏造训练规模或断言它用在线 teacher-state MSE。 |
| [NVIDIA QAD v3](https://arxiv.org/html/2601.20088v3)，§3.4、4.1–4.3 | 原始同尺寸 BF16 teacher；LLM/VLM 输出分布 KL、T=1；SFT 或 BF16 生成文本。报告收敛量约0.3B–6B文本 token，保守 LR 1e-6–1e-5，量化覆盖按模型选择。 | 原 teacher、较小 LR、真实主 W 更新、合成数据均是成熟做法。视频空间 token 数不能等价成独立文本 token 数；32个文本/64轨迹远不能据空间 token 数宣称数据充分。KL 优于 logits MSE 的结果不支持把连续 velocity 任意 softmax 后改 KL。 |
| [QUADS](https://arxiv.org/html/2607.15810v1) | MoE RL 的 rollout/trainer 对齐；非对称训练端 W fakequant、A不量化，配合 rollout 残差激活补偿。 | 其 importance ratio/策略梯度失败机制不同于我们的监督 NMSE。不能照搬 W4A16 训练并省略其推理补偿，也不能把 QDQ/native 差异首次发现作为贡献。 |

FastWan 的公开 checkpoint 可以作为独立外部产品基线，但不能直接替换 rCM 权重继续同状态误差比较。其发布的1.78秒包含步数、attention、解码与工程变化，不是我们主层 QAD 的质量匹配速度收益。

## 最明确的成熟配方差异：outer global 与 block scale 必须分开

以下结论来自当前官方 ModelOpt 默认 `max` NVFP4 路径，不是所有 NVFP4 训练的统一规定。

| 项目 | 当前 ModelOpt 默认 PTQ→QAD | E022 |
|---|---|---|
| W outer global | PTQ 收集 W `_amax` 后保留；正常训练前向优先读 buffer | 每次由更新后的 FP32 master 全矩阵 amax 重算 |
| A outer global | 校准集收集 A `_amax` 后保留；不是每个 batch 自适应 | 每个实际 A 独立全张量 amax |
| group16 E4M3 SF | 根据本次 W/A 组内 amax 动态计算，使用保留的 outer global | 动态计算，但 outer global 也随当前张量变化 |
| STE /尺度导数 | 默认 `pass_through_bwd=True`；global amax 不求导 | identity STE；也不求尺度导数 |
| 默认初始化变换 | `max` 配方，无额外 SmoothQuant/rotation 要求 | plain 原 BF16 W，无变换 |

可复核路径：[`nvfp4.yaml`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt_recipes/configs/numerics/nvfp4.yaml) 19–23行为 group dynamic；[`QuantizerAttributeConfig`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt/torch/quantization/config.py) 478–485行顶层 `type` 默认 static，627–642行默认 identity STE；[`TensorQuantizer`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt/torch/quantization/nn/modules/tensor_quantizer.py) 736–751行优先读取保存 `_amax`，890–920行把它传入 dynamic block quant。因此“所有 scales 都冻结”与“type dynamic 就全动态”都不准确。

**校准图**：默认 [`max` preset](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt_recipes/configs/ptq/presets/model/nvfp4.yaml) 无平滑。[`max_calibrate`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt/torch/quantization/model_calib.py) 317–369行先启统计、直接访问 W，再执行校准前向；1135–1174行显示采集期间关闭 quant，结束才加载统计并启 quant。因此从原 BF16 模型做默认 max 校准时，A 统计来自未量化图，并非已有 QDQ 误差传播的图；已有变换的 checkpoint 或 layerwise/AWQ 配方另当别论。

**饱和及舍入**：[`fp4_kernel_hopper.py`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt/torch/kernels/quantization/gemm/fp4_kernel_hopper.py) 73–89、130行采用固定 outer global 与动态组 amax；[`nvfp4_quant.py`](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/modelopt/torch/kernels/quantization/common/nvfp4_quant.py) 105–125行先将组尺度裁到 E4M3最大448，E2M1 magnitude 最大6，33–63行是 ties-to-even。旧 legacy 规则不能因名为 NVFP4 就与之混同；固定 global 后还必须保留有限饱和，避免 FP8 cast 溢出成 NaN。固定版本的 Hopper fakequant 对正且有限的 global 使用正常组尺度；零组尺度输出零，小的非零尺度正常量化；仅非法 global 使用单位组尺度。**版本更正：前次网页缓存所见“很小有效组尺度替换1”的行为不属于此固定版本，现撤回该当前版本描述。**

所读默认路径没有 EMA/delayed-amax，也没有必须学习尺度的要求；当前源码虽然有独立 LSQ/静态组尺度分支，不能把可选支持说成默认 QAD 必须项。**identity STE 忽略 scale derivative 本身并非我们漏掉的成熟关键步骤。**

## 一个应优先补的具体对照，而非新方法

E022 完成后，优先做 **W-only 固定初始 outer global** 对照：每层从同一原 BF16 W 取一次 global，组 SF 继续随 master 更新，A 保持 E022 动态；数据、shuffle、128更新/512micro、lr3e-6、NMSE、量化覆盖、舍入、开发选择规则全部相同。训练与导出必须使用同一个已保存 global；同后端原生验证仍保留。超出范围需显式饱和并记录比例，不允许 export 偷偷重新取 global。它只隔离 W 尺度政策，不称完整 ModelOpt 复现；先不混入 fixed-A、EMA、LSQ 或平滑网格。

这是可直接落到现有 packer 参数与 checkpoint buffer 的普通基线补充，不需要新算子或 serving。按 E020 约7.8秒/micro，额外512micro保守约67分钟，加开发导出评估仍是现有90分钟级任务；不需要重新采集 teacher 数据。官方 [QAD YAML](https://github.com/NVIDIA/Model-Optimizer/blob/e68eb44ee5c7fdb6a86885f457f1cdba0598e0c1/examples/llm_qat/configs/train/qad_nvfp4.yaml) 提供20,000训练样本、batch2×accum2、5% warmup、cosine等现成实践；这些说明我们的 constant-LR短程配方不是唯一成熟选项，但本次不同时改调度器来破坏单因素解释。

尚未被排除的实质限制是：有限独立文本/轨迹覆盖、teacher-state 拟合与学生自由轨迹的分布偏差、原生部署误差与 QDQ 训练目标差异，以及尺度政策/普通优化配方的影响。目前没有证据证明其中任何一项不可由充分训练和成熟配方解决。若固定 W global 改善，只能说该稳定化基线有效；若不改善，也只排除这一受限对照。继续增加数据、训练、精度匹配与自由生成质量验证具有研究价值，但它们以及已知 DMD、平滑、尺度稳定化本身都不是新的顶会贡献。
