# 原版 Wan2.1 T2V-14B BF16 完整生成入口（prepared，未运行）

2026-10-03，仅CPU源码/配置检查；不载权重、不启动GPU。结论：当前原版 `WanPipeline` 可通过显式注入14B Transformer及已有公共组件组成完整参照，不需要改Diffusers模型数学或使用PTQ loader。**尚无合格原版14B SVDQuant checkpoint，本入口只针对BF16 teacher。**

**资产和环境。** Transformer固定为 [Wan-AI/Wan2.1-T2V-14B-Diffusers@38ec498c](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)，目录 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-14B-Diffusers-38ec498c`（下称W14）。config/index确认40层、40×128 heads、FFN13824、patch `[1,2,2]`、16 latent channels；1,095 tensors分12片。下载/全SHA/头部核验终态看 [supervisor.json](../../results/research/asset_wan14b/supervisor.json) 和 [download.json](../../results/research/asset_wan14b/download.json)。只准备Transformer和小元数据，W14不是独立完整pipeline目录。

复用 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers`（BASE）的UMT5、tokenizer、FP32 VAE；[component_reuse.json](../../results/research/asset_wan14b/component_reuse.json)的13项身份与14B仓库相符，大文件沿用E024全SHA+当前size，小文件本次核Git blob。native Python `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python` 即已验证的torch2.11/cu128、Diffusers0.40、imageio/FFmpeg环境。缓存放DATA1，无需安装。

**最薄加载路径。** 新worker复用[E043媒体/记录辅助函数](../../scripts/research/run_vanilla_wan_reference.py)，但自行组装组件，避免把1.3B Transformer加载后再替换：

```python
dit = WanTransformer3DModel.from_pretrained(W14, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
te = UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True)
tok = T5TokenizerFast.from_pretrained(BASE, subfolder="tokenizer", local_files_only=True)
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
scheduler = UniPCMultistepScheduler.from_pretrained(W14, subfolder="scheduler", local_files_only=True)
pipe = WanPipeline(tokenizer=tok, text_encoder=te, vae=vae, scheduler=scheduler, transformer=dit)
```

constructor见当前`pipeline_wan.py:131`，未实际载14B。相同E043文本/negative/512长度可复用保存的embedding和FP32 noise，不再载TE；新prompt则一次编码后将TE搬CPU、两seed复用，明确记录常驻边界。

**新wrapper必须改的几处。** E043 `prerequisites:81` 的30层改为所选14B配置；不要继承旧native 300 Linear/60 SDPA断言：按当前40个block、每块self/cross attention，预期80 SDPA/DiT（实际计数确认）；若以后讨论同覆盖PTQ是400个候选主干Linear，当前并无其权重或运行证明。50步CFG>1仍是100 DiT/50 scheduler/1 decode每视频，与层数无关。manifest/来源/目录、采样参数和worker预算另定。CFG6/shift8是1.3B推荐：14B固定HF scheduler是shift3，固定HF README的Diffusers示例为CFG5；其他选择须显式登记。若保持480×832/81帧，FP32 noise `[1,16,21,60,104]`及32,760视频patch tokens仍适用；720p需同时修改噪声/媒体几何。

**精度与资源边界。** 保留原pipeline的FP32 latent准备/更新（本地`pipeline_wan.py:562–568,635`）、BF16 DiT输入/两支CFG及cache_context（605–632）、FP32 VAE与一次原mean/std denorm（656–668）；FLASH context仅包DiT，不包FP32 VAE。不要全pipeline `.to(bfloat16)`。存储前两片头为F32，其余待下载终检；按全F32 payload推算的BF16 DiT权重约26.61GiB，TE约10.58GiB、FP32 VAE约0.47GiB，共约37.7GiB权重，不是实测峰值或可用性保证。E043的26.52GiB峰值、约183–188秒/视频及1500秒worker预算均不能外推到14B；应在空闲72GiB卡上另定预算，实测全长activation/decoder峰值与耗时。下载完成并确定新协议后再正式完整生成；本轮不加GPU smoke或新量化实现。
