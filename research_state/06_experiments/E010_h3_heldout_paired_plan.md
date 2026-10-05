# E010：两个未校准 prompt 的 H3 BF16/native 配对生成

2026-10-02，预注册，尚未启动 GPU。目的只是在真实自由轨迹和解码视频/音频上闭合 E009 baseline；两例不构成质量 benchmark、泛化证据或质量等价声明。

## 固定样本与采样协议

源表 `/home/wjq/workspace/178866172854036`，124 条；SHA256 `19dc31db08d18bee157879b7d9823fdc00621a2a0cf28cd2d09c0c853666af45`。完整文本、seed、文本 SHA 和资产清单已锁入 `E010_h3_heldout_manifest.json`。

| Prompt | 原 seed | 原始内容与选择理由 |
|---|---:|---|
| 30 | 49771 | 5 秒中文字动画、胶片噪声与环境底噪；检验高对比细节/时序文字变化 |
| 36 | 59526 | 5 秒液体微距→产品展示，液体声、铃声与气流音效；检验流体/细节与场景转换 |

排除 PTQ/calibration prompts `{1,11,20,25,46,48,105,116}`。先按原时间标注≤5秒筛选，再排除已存在旧生成的 p26/p42/p51；不看 BF16/native 结果选择。本地 `results/calib`、`results/samples`、`outputs` 及 research artifact 检索未见 p30/p36 旧样本。这里的“heldout”仅指旧 PTQ state 的校准集合，不能证明相对预训练语料 unseen。

仅 BF16 与 **E009 已验证的 compat nativefast** 两臂，不加旧 QDQ 陪跑。固定 576×1024、124 帧、24 FPS、20 steps、CFG=1、video flow shift=12、audio shift=3、CPU BF16 初始噪声。124/24=5.1667秒；两个原 prompt 的5秒脚本不需要截取十几秒叙事。**20 steps来自原 `collect_minimax_h3_calib_standard.py`、`infer_minimax_h3_svdquant_standard.py` 及8份缓存manifest，pipeline本身默认50；本次不是“默认50步质量”的报告。** CFG=1只跑positive分支，每臂每prompt恰好20次DiT，两个模态各20次原scheduler.step。

这两例不覆盖人物动作/口型/长语音。若BF16自身不能正确生成文字或产品，不把双方共同失败归因量化；禁止据结果换prompt、换seed或只展示较好一例。

## 已核实的本地资产与环境

| 资产 | 路径 | 文件数据量 |
|---|---|---:|
| Pruned DiT | `/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | E009同一份 |
| Text encoder | `/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors` 至 `00014` | 66,714,912,872 B / 62.13 GiB |
| Video VAE | 同FL2VA根目录 `video_vae/source/model.safetensors` | 10,415,548,320 B / 9.70 GiB |
| Audio VAE | 同FL2VA根目录 `audio_vae/model.safetensors` | 605,429,308 B / 0.564 GiB |
| Processor | 同FL2VA根目录 `processor/` | tokenizer、merges/vocab、image/video配置齐备 |
| Native export | `/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/` | E009原200层export |

CPU已核对16个safetensors的header与完整文件长度匹配；processor可离线实例化为Qwen3VLProcessor/Qwen2Tokenizer，T2VA text tags全1；PyAV18.1.0具备libx264/AAC编码器。CPU检查不等于新的TE/VAE GPU执行已验证。prepare在任何GPU模型加载前，对所有model shard及processor作完整SHA，后续阶段绑定该清单、size/mtime、源快照，任何变动停止。

使用 recovered Python `/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python`。强制 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`、`DIFFSYNTH_SKIP_DOWNLOAD=True`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，本地绝对ModelConfig路径。必须显式H3_MODEL_ROOT为上表DiffSynth目录：common默认 `/data1/models/svdquant-wjq/models/MiniMax-H3` 不存在。当前默认 `data/minimax_h3_prompts.jsonl` 只有p2且seed0，本轮完全不用该文件。

## 实现与阶段隔离

新入口 `scripts/research/run_h3_native_paired_video.py`，原pipeline、common、scheduler、E009源码均不修改。每个阶段独立进程，源文件/manifest/本计划快照随artifact保存。

1. `--phase check`：纯CPU验证固定prompt/seed、原资产header、E009正确性与性能前置；导出manifest SHA及四项SDPA backend开关必须与E009参考一致；离线tokenizer、原NoiseInitializer双模态noise复现、两个20-step时间表。要求CUDA未初始化。
2. `--phase prepare`：只加载text encoder与processor（disk wrapper，vram_limit=30GiB），逐原 `MiniMaxH3Unit_PromptEmbedder.process` 编码两个prompt并缓存CPU embedding/tags。原 `NoiseInitializer` 在CPU创建每prompt的video与audio初始BF16噪声；**两个模态各自用相同seed创建独立generator**，不误用串行同一个generator。完成后进程退出释放TE。
3. `--phase denoise --arm bf16/native`：每个进程只装一份resident DiT，复用E009 `make_h3_resident` 与native installer/fastpack。直接调用原 `MiniMaxH3Pipeline.__call__`；传 **`prompt=None, text_embedding=cached`**，否则原接口会重新编码并拼接embedding。原unit链、双scheduler、`cfg_guided_model_fn`、`model_fn`及 `step` 算法不重写。只用实例级placement guard在VAE加载前停止，并观察原step输入/输出。两臂共用冻结embedding与初始noise，自行递推，不喂teacher后续latent。
4. 每个DiT调用实际统计BF16 SDPA（由真实cu_seqlens推导期望次数）与native200 GEMM；每次native collector退出后验证200flags，非有限域仍拒绝，SF0 affected calls仅计数。保存全部40个video/audio step文件、sigma/timestep、前后latent及预测；失败时保留已完成step与错误，不改帧数/阈值续跑。
5. `--phase decode`：确认两个denoiser进程报告complete，再在VAE-only进程按原 `decode_video(...tiled=True,tile_size=256,tile_overlap=64)` 与 `decode_audio` 解码全部四组latent，依次换组件释放显存；沿用原temporal stitching，不引入Wan的core/halo算法。统一24FPS/32kHz stereo H264+AAC；重新读取实际媒体帧数/尺寸/音轨。每视频存含 `{video:absolute_path,prompt:full_text,variant,prompt_id,seed}` 的caseJSON，以便随后有限辅助评估。

## 预算、止损和产物

由E009同shape单步8.260s/6.787s粗估，两prompt×20steps×(BF16+native)约 **10.0 GPU分钟** denoiser；prompt token数略变，绝不是生成速度实测。TE两次编码、VAE四次解码尚无本轮实测，预留额外10–25 GPU分钟，整体目标 **20–35 GPU分钟**，单卡外层硬预算60分钟；阶段检查时发现当前或历史峰值allocated>60GiB即停止，不降shape。这是检查点止损，不是分配发生前的硬限制。text encoder、DiT、VAE互不常驻同一进程。超过预算/缺依赖/非有限/源漂移保留partial，不自动加样本、下载或修改量化定义。

数据：`/data1/models/svdquant-wjq/research/20261002/E010/{prepare,denoise_bf16,denoise_native,decode}/`。小报告：`results/research/E010_h3_{check,prepare,denoise_bf16,denoise_native,decode}.json`；logs在 `results/logs/`；已存在报告/目录默认拒绝覆盖。交付全部四个视频、两个配对音轨、shared-noise/embedding证据、每step轨迹与失败记录。不给两样本打“质量等价”标签，也不以辅助评分替代观看/听音与范围限制。

首个待root审阅的执行命令（**本计划撰写时未执行**）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/run_h3_native_paired_video.py --phase prepare
```

随后依次 `--phase denoise --arm bf16`、`--phase denoise --arm native`、`--phase decode`，均named tmux+logs、每次启动前重新确认目标GPU空闲。入口已强制离线与torch attention；仍须在launcher中加入recovered/bin到PATH以使用对应runtime工具。
