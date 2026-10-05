# Wan14B 原生 W4A4 / SVDQuant 入口核查

2026-10-03；仅源码、JSON和safetensors头，不载权重、不运行GPU、不改E049。官方资产已完成：[download.json](../../results/research/asset_wan14b/download.json)确认1,095 tensors、14,288,491,584个F32元素，40层。头部实际主干覆盖为 **400 Linear / 14,050,918,400权重元素**：320个`[5120,5120]`、40个`[13824,5120]`、40个`[5120,13824]`；cross-K/V也已在模型内投影至5120，不是直接4096输入。这里没有新的14B量化checkpoint，也不重复外部checkpoint检索。

**Plain native：算子可复用，整模安装方式不能照搬。** [wan_mainweight_qad.py](../../scripts/research/wan_mainweight_qad.py:31)的TARGET_NAMES、`_targets`、export/install的300声明均固定30层，应在新适配中由40层得到400个同后缀名字，并核非目标state保持原值。`PlainPackedWanLinear`与[NativeWanLinear.main_from_packet](../../scripts/research/wan_native_nvfp4.py:195)没有hidden1536等限制：K/N均满足32对齐，group16、uint8双FP4码、E4M3 scale的128×4物理swizzle、独立FP32 tensor-global及BF16输出可沿用；81帧图像M32760与text M512也沿既有同形状路径。需保留Wan legacy E4M3 ties-up / E2M1 signed ties-larger及zero-SF检查，不能改称标准RNE；注册后不能整体cast模块而把FP32 global改BF16。

关键实际阻碍是**全FP32 master的构造峰值**。`install_qad`先以targets字典保留全部原Linear，再逐个建立FP32 master；只改30→40时，原BF16整模26.614GiB＋主W master52.344GiB已至少 **78.958GiB**，超过72GiB卡，尚未计临时张量。普通未训练baseline应逐Linear执行“原teacher BF16 W→临时FP32→现pack→packed buffer/CPU artifact→释放原W”，不先安装整模QAD图；原下载F32 W也应先经过teacher的BF16 cast，避免偷偷改变量化输入。底层packer/GEMM和bias、非目标state保存可复用，无须新kernel。此处是部署适配，不是QAD训练或SVDQuant的替代强基线。

**SVDQuant：校准/加载骨架可复用，必须重新获得14B五文件。** [infer_rcm_wan_4step.py:22](../../scripts/infer_rcm_wan_4step.py:22)的`load_quantized_transformer`可借其attention类型注册、smooth/branch加载和`ptq(load_dirpath=...)`，不能借rCM采样主函数；当前parser仍指定1.3B YAML，应新入口显式14B model path/name及相同量化配置。`wan_native_nvfp4.convert_wan_transformer_to_native:338`固定300须改为400并逐W保持保存codes/scales roundtrip、LR先于activation-QDQ的hook顺序及parent smooth。新`model/scale/wgts/branch/smooth.pt`都必须来自14B；1.3B低秩、平滑、scale或E047 checkpoint不能迁移，rank32仅是可选复用配方。

[ptq_wan_matched_calibration.py](../../scripts/research/ptq_wan_matched_calibration.py)已验证直接DiT→`DiffusionModelStruct`、关闭旧gated扩展及保留Diffusers0.40 `(cos,sin)`广播；需解除其30/300注册断言和1.3B模型/CFG6/shift8标签，保留rank32、g10、64records及batch4等明确合同。缓存格式可以复用，但应采集14B自身CFG5/shift3轨迹输入；旧1.3B latent即使形状合法也不是14B匹配校准，TE身份相同不改变这一点。

缓存确实按layer迭代、CPU收集激活并yield后清理（`dataset/calib.py:413`、`dataset/cache.py:419`），Wan随后使用上一层输出（`nn/struct.py:1705`）；**这不是权重按层offload**，现PTQ入口仍将整个模型放GPU，并保留本层64样本缓存/校准临时值。E047 1.3B实际PTQ为19,228秒、GPU peak allocated25.410GiB，仅为已有经验，不能给14B造时长或显存预测；14B完整batch4/G10/LR100早停的GPU/主存峰值均未测。

下一最小入口问题已具体化：plain需要避免全master峰值并完成400模块真实完整forward；SVD还缺14B匹配缓存与五文件，首先需要在上述完整既定样本合同下取得实际layer校准资源证据，再决定完整PTQ预算。两条路径都不要求改attention、CFG双前向或FP32 sampler/VAE；plain完成也不代表SVD强基线完成。本轮未实现或启动任何一项。
