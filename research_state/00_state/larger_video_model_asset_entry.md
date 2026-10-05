# 更大视频模型本地入口：有限资产核查

2026-10-03。只读目录、JSON配置/索引及文件大小；未加载模型、未重哈希大文件、未下载或运行GPU。先查既有记录：H3的资产/规模和高精度参照已有充分核查，下文仅引用；Wan14B未找到既有本地完整性结论，才做限定目录检查。

**结论：指定资产根内没有可立即加载的原版 Wan2.1-T2V-14B；已有可复用的大模型是本地剪枝版 H3，不能把它当完整未剪枝官方H3。** 未扫描其他用户目录或全盘，因此不是“服务器任何位置都不存在14B”的断言。

| 已查实际资产 | 事实与入口 |
|---|---|
| `/data1/models/svdquant-wjq/models` | 顶层10个模型目录，无14B目录。检查其中全部Wan家族的config及权重索引：基础Diffusers、rCM转换版和FastWan-QAD均为30层、12×128 heads、FFN8960的1.3B；`rcm-Wan`只有`rCM_Wan2.1_T2V_1.3B_480p.pt`（2,838,262,729 bytes）。没有隐藏在这些产品目录内的14B Transformer索引/分片。FLUX目录是图像模型，不作视频跨模型资产。 |
| 原版 `Wan2.1-T2V-1.3B-Diffusers` | Transformer两分片、TE五分片索引所列文件均在；原版身份与payload完整性已有[入口审计](vanilla_wan_entry_audit.md)和[E043来源收据](../../results/research/E043/base_transformer_provenance.json)，不重复验证。另一个无Diffusers后缀的原版目录曾确认DiT/UMT5下载不完整，不能作更大模型入口。 |
| rCM / FastWan-QAD | 是蒸馏/量化产品，非原版14B。FastWan-QAD目录的5,676,070,784-byte Transformer不意味着模型有更多参数；其配置仍为1.3B。相同TE/VAE与大体积text encoder也不能算另一个视频DiT。 |
| H3 | 已有[模型与格式核查](native_h3_next_baseline.md#4-实际存储预算)：实际DiT路径为`/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`，532 tensors、40,225,668,192 payload bytes，混合BF16/F16/F32；200个主干Linear含19,267,584,000权重元素，BF16计算模型常驻约37.460GiB。不得称完整33B官方模型。本次只看官方形状的`MiniMax/MiniMax-H3/FL2VA`目录：包含TE14个分片、processor、video/audio VAE；没有另一套未剪枝DiT文件。TE的`of-00014`是分片数，不能作Wan14B资产证据。 |

**H3参照边界已有结论，不重看媒体。** [E010](../reports/010_20261002_h3_heldout_results.md)已经完成真正BF16与native的两提示完整生成，采用20步而非pipeline默认50步；文字/产品提示覆盖有限、音频当时未评价，不能视为广泛质量参照。[E013](../reports/012_20261002_behavior_readiness_stop.md)完成两条50步BF16，因读出门槛及远离行为不可靠而停止；不是“BF16不能生成运动”或量化阴性。特别是[E038](../06_experiments/E038_center_video_plan.md)的`bf16`标签只指attention，主Linear仍是native SVD，不能拿其八条当全BF16 teacher。

**最短复用与缺项。** H3已有 `run_h3_native_paired_video.py` 的TE/噪声、resident BF16 DiT和VAE分阶段入口，`h3_native_nvfp4.make_h3_resident`及E009导出可复用；恢复环境和原生环境路径均在原计划/收据中，无须重装。但旧runner绑定旧提示及协议，新跨模型问题仍需另行固定输入，不能修改旧已执行源。原版Wan14B则首先缺本地Transformer权重/config/索引与对应来源收据；当前不具备“只换model_dir即可跑”的事实基础。本轮未下载或拟建新14B实验，也未改变E047/E048运行链。

## 2026-10-03 19:21 上海时间附记：官方14B资产下载已启动，尚未完成

以上保留的是启动下载前的审计结论。随后研究任务授权准备原版14B Transformer；这不是启动跨模型实验。官方仓库为 [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)，固定 revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`。12个权重分片共 **57,154,077,760 bytes（53.229 GiB）**；索引列1,095 tensors、payload 57,153,966,336 bytes。官方配置为40层、40×128 heads、FFN13824；已下载的前两分片头均为F32，**全部分片的dtype与完整性仍须等待最终头部/索引校验，不能先称已验证BF16存储或完整pipeline**。

独立目标为 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-14B-Diffusers-38ec498c`，范围仅Transformer及README/model_index/scheduler和组件小配置；不下载TE/VAE大文件。缓存、partials及临时目录全部在 `/data1/models/svdquant-wjq/research/cache/wan14b-assets-38ec498c`，不使用根HF缓存。启动时DATA1空闲1,270,046,707,712 bytes。现有1.3B目录的TE五分片、VAE与tokenizer/config共13项已经与该14B revision身份对照匹配：大文件引用E024先前实际完整SHA256及本次文件大小，小文件本次核Git blob SHA1；见 [component_reuse.json](../../results/research/asset_wan14b/component_reuse.json)。这只证明资产可复用身份，不证明14B完整生成已经运行。HF模型卡声明Apache-2.0、树中无独立LICENSE；另存官方Wan代码仓库固定版本许可证，来源边界见 [license_source.json](../../results/research/asset_wan14b/license_source.json)。

唯一会话 `tmux wan14b_assets`，supervisor PID3654981 / worker PID3654983，19:18:36启动，**22:18:36硬deadline**。两路有界curl，CUDA隐藏、0模型/张量加载；成功后逐分片核官方LFS SHA256，并检查safetensors头的dtype、shape/offset/payload与分片索引一一对应。失败或超时保留已有分片与partial，不自动降低模型规模或覆盖原模型。

入口 [prepare_wan14b_assets.py](../../scripts/research/prepare_wan14b_assets.py)，源SHA256 `c04e1d11d8a066ec9545ff9ac0f9a94d41bbdd6ff0bd698397187b34284d4a0e`。启动收据 [launch.json](../../results/research/asset_wan14b/launch.json)；运行/终态收据 [supervisor.json](../../results/research/asset_wan14b/supervisor.json) 和 [download.json](../../results/research/asset_wan14b/download.json)；日志 `results/logs/wan14b_assets_download.log`。截至本附记仍为running，不能将目录存在当作下载完成。根节点接手低频监督，完成后再做有限独立核验；E047/E048运行链未修改。


## 2026-10-03：官方量化资产的有限核查

仅核查SVDQuant/DeepCompressor/Nunchaku维护方的一手发布与一个易混淆近邻：未发现可直接复用的**原版Wan2.1 T2V-14B NVFP4 W4A4**权重及与之绑定的校准帧数/schedule/rank/加载格式说明，不代表全网不存在。[Nunchaku官方HF目录](https://huggingface.co/nunchaku-ai)与[transformer API](https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.models.transformers.html)未提供此Wan入口；DeepCompressor固定tree `69f3473f5e1c1504bae35cc50c7858ef900a9b17` 的[通用NVFP4配置](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/nvfp4.yaml)和[默认rank32](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/__default__.yaml)不是Wan14B发布权重的协议证据，其[公开Nunchaku转换入口](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/backend/nunchaku/convert.py#L399)限定Flux。

[LightX2V Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4/blob/main/README.md)列的是4步蒸馏I2V-14B/T2V-1.3B，不能替代原版T2V-14B对照。因此当前仍没有已核实的即用14B量化基线，后续需自行PTQ/转换或先验证其它发布产物；不能将当前原版Transformer下载完成等同于量化基线已准备好。本次查阅未下载额外大权重、未用GPU，E047/E048保持原协议。


20:19下载检查点：同一监督3654981/worker3654983实际存活、两收据running，已运行约1小时1分钟。4/12权重分片已完成并通过官方LFS SHA256；含小元数据的已校验文件共19,621,848,703 bytes，另有1,706,610,688 bytes未完成分片。全体头部/offset/index验证在下载全部完成后执行，当前不能称完整模型资产已就绪。


## 22:06 完整资产与生成入口更新

原下载/监督均complete/rc0，9926.867秒，原PIDs3654981/3654983已退出。12分片及元数据57,154,198,566B；完整header/index确认1,095张量、14,288,491,584元素全F32，payload57,153,966,336B。root独立读取实际headers、小文件SHA和全部当前size，与生产者及官方LFS收据一致；未重哈希大权重/载tensor，[completion_review](../../results/research/asset_wan14b/completion_review.json)。因此此前“尚未完整下载/完整dtype待核”已被本记录取代，历史进度仍保留。

E049已通过隐藏CUDA的八实际noise/embedding输入检查，随后在GPU0–3加载原版14B BF16、公共FP32 VAE，0TE。22:06四worker真实存活、12/12分片加载完成，尚未完整生成/质量结果。采用HF Diffusers CFG5/shift3、50步与81f480×832，16fps本地约定；参见[E049计划](../06_experiments/E049_wan14b_reference_plan.md)。无14B原生NVFP4/SVDcheckpoint就绪的主张。
