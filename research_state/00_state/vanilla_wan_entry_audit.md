# 原版 Wan2.1 1.3B teacher 入口只读审计

2026-10-03。结论：**本地官方 Diffusers 格式基础模型可直接作为独立 teacher 入口；不需要补下载或使用 rCM 转换器。** 本次只读文件头、配置、小片权重和环境，没有生成、加载完整模型或重哈希大文件。

## 资产与来源边界

根目录 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers` 中：

| 组件 | 实际完整性检查 |
|---|---|
| Transformer | 两个 FP32 safetensors，4,998,781,576 + 677,289,072 bytes；825 tensors、1,418,996,800 参数；index 与全部 header keys/所属 shard 完全一致，各 shard 最大 payload offset 恰到 EOF。30 blocks、1536 hidden、12×128 heads、8960 FFN。 |
| UMT5 | 五个 FP32 shards，242 tensors；index/header 一致且各 payload 完整。不是不完整的原始 UMT5 `.pth`。 |
| VAE | FP32，194 tensors、126,892,531 参数，507,591,892 bytes，payload 完整；配置含 Wan 的逐通道 mean/std。 |
| Tokenizer / scheduler | T5 tokenizer 的 spiece/model/json/config 齐全；UniPC config 为 flow prediction、1000 train timesteps、flow sigmas、solver order 2。磁盘默认 **flow_shift=3**，不能默默沿用。 |

TE 五 shards、tokenizer 两大文件、VAE 已在 [E024 资产 receipt](/home/wjq/workspace/svdquant-exp/results/research/E024/assets_manifest_attempt3.json) 以 `verified_local_symlink` 对这些同源文件验证官方 LFS SHA；本次继承该事实，不重复扫描。本轮 root 随后独立完成基础 transformer 两完整 shards 的官方 LFS SHA 验证，绑定 HF revision `0fad780a534b6463e45facd96134c9f345acfa5b`，两项全部相等；见 [E043 provenance receipt](/home/wjq/workspace/svdquant-exp/results/research/E043/base_transformer_provenance.json)。本 agent 继承该 receipt，没有再次重哈希。三个分散权重各读取 1024 元素，统一 cast BF16 后，与 rCM 分别有 744/697/903 个不同，与 FastWan 有 965/851/815 个不同；不是这两个蒸馏模型的同一权重。此小片检查不是全模型等价证明。

另一个 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B` 只有原版 DiT/UMT5 的 `.incomplete` 文件（591 MB/798 MB）和完整 VAE，不具备原始 Wan CLI 的完整资产。有限检查 workspace/third_party 未发现独立 Wan-AI 官方 checkout；本地 NVlabs/rcm 有 Wan 派生模型与 diffusion 入口，但不是必须绕行的基础 teacher。既有 `verify_rcm_wan_transformer.py` 和旧数值报告只覆盖 rCM 转换，不能证明原版 Wan 原架构与 Diffusers 全 rollout 逐位等价。

## 最短完整生成路径

使用 `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python`。本次 CUDA 隐藏 import 成功：torch 2.11.0+cu128；Diffusers 0.40.0 实际来自 `/home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/diffusers`。没有加载模型。torchao 可选导入警告不妨碍三个所需类的导入。

直接 `AutoencoderKLWan.from_pretrained(BASE, subfolder='vae', torch_dtype=float32, local_files_only=True)`；再 `WanPipeline.from_pretrained(BASE, vae=vae, torch_dtype=bfloat16, local_files_only=True)`，不覆盖 transformer。`UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8)`；480×832、81 帧、50 步、CFG 6、固定原 negative、16 FPS、不扩 prompt。原 [本地官方 README](/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers/README.md:148) 推荐 1.3B 的 CFG 6、shift 8–12 和 480p；同页 Diffusers 例子另用 CFG 5/15 FPS，应明确本轮采用前者采样推荐与指定 16 FPS，不混称例子完全一致。实际 [WanPipeline](/home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/diffusers/pipelines/wan/pipeline_wan.py:568) 使用 FP32 latent/state，模型输入 cast BF16，CFG 每步两次前向，最终按 VAE 配置一次 denorm；保留原 pipeline sampler，VAE 独立 FP32。旧 `infer_wan_bf16_vs_w4a4_one.py` 经过 DeepCompressor、仅 33 帧并默认 shift 3，不宜直接当此次入口。

## NVFP4 复用与当前限制

E007 `convert_wan_transformer_to_native` 的 300 Linear 覆盖与原版结构兼容，但要求**先加载对应模型的**历史 PTQ 图、300 项 saved residual/scale、正确 LR-before-QDQ hook 顺序。不能将 rCM/E020/E022 权重载入基础模型后称为原版量化。实际另有 `ckpts/wan2.1-1.3b-real-nvfp4-s16`，五项 model/scale/smooth/branch/wgts 均可解析到存在文件，`wgts.pt` 确有 300 项；其校准来源、当前 loader/fastpacker 合同与数值未在本次重验证，先只作为可复用候选。E007 的 saved-weight roundtrip、动态激活 legacy Wan 舍入、平滑/LR、native GEMM 实现可复用；原 rCM 实验输入、采样和 SHA 参照不可复用为基础模型证据。

目前没有基础 teacher 完整新视频结果，不能保证动作可读或排除共同 TE/VAE/实现问题。BF16 DiT、FP32 VAE 与磁盘 FP32 transformer 也不是全 FP32 teacher；它是官方常规推理精度入口。与 rCM 的差异同时包含权重、采样步数、CFG、shift 和时间长度，不是单因素机制实验。

本 agent 的 GPU 快照：0/1/5 为 0 MiB、0%；2/3/4 各 345 MiB、0%；6 为 52,153 MiB、7 为 72,479 MiB。root 随后 12:20:21 更新：0–5 全部 0 MiB、0%，6/7 仍属其他任务。共八张 RTX PRO 5000 72GB Blackwell；任何正式启动仍须即时复查。本次无 GPU 工作。

E043 薄入口已另建 `scripts/research/run_vanilla_wan_reference.py`，CPU `--check` 完成、未加载模型。原生环境没有 PyAV，首次 import 失败已保留；现用已安装 imageio + imageio_ffmpeg 自带 FFmpeg 编码和完整读回媒体，无环境修改。
