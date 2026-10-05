# E024：官方 FastWan 薄 harness 实施备注

2026-10-03，只读本地源码；无 GPU、安装、下载或推理验证。实际 checkout 为 `/data1/models/svdquant-wjq/third_party/FastVideo-8444c089`，HEAD `8444c0897a8b96848eb85b6e5750ef486f79fc92`。下文路径相对此 checkout；HF 小文件来自 `/tmp/fastwan-readiness`，固定 revision 沿[readiness记录](/home/wjq/workspace/svdquant-exp/research_state/00_state/fastwan_qad_external_baseline_sources.json)。**可直接复用官方入口，但有输出、计时和 scheduler 文档差异需要显式记录。**

- **构造与调用。** [入口](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/examples/inference/optimizations/FastWan_QAD_TAEHV.py:129)的 `build_generator(args)` 只读取 `model, baseline, no_compile, distilled_model, taehv, num_gpus`。首次调用用 `baseline=False, no_compile=True, taehv=False, num_gpus=1`；`model` 指本地完整 Wan base，`distilled_model` 必须是新 QAD transformer 的**绝对文件名**，不是目录。若 `model` 已是完整 QAD snapshot 则 `distilled_model=''`。默认 HF resolver 寻找不存在于公开清单的 `generator_inference_transformer/...`，不要触发它。环境显式 `FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER`、HF/Transformers offline；该环境变量并非此示例自动设置。

```python
result = generator.generate(request={
    "prompt": prompt,
    "sampling": {"seed": seed, "height": 480, "width": 832,
                 "num_frames": 81, "fps": 16,
                 "num_inference_steps": 3, "guidance_scale": 1.0},
    "runtime": {"return_trajectory_latents": True},
    "output": {"output_path": absolute_mp4, "save_video": True,
               "return_frames": True},
})
```

- **保存与解码。** 上述参数是实际 typed-request schema（`api/schema.py:151–219`）；已有目标文件会自动加数字后缀，保存路径应以 `result.video_path` 为准。单 prompt 返回 `GenerationResult`：`samples`、`frames`、`size`、`video_path`、`trajectory`、`trajectory_timesteps`、`generation_time`、`logging_info`、`extra`（`api/results.py:12`）。full Wan VAE时 `samples` 是 CPU NCTHW `[0,1]` pixels，`frames` 是 uint8 RGB帧；启用 trajectory 可保留三个 **step-after** latent及对应实际 t，最终 latent 为 `trajectory[:, -1]`，不是再跑一个采样器（`stages/denoising.py:499–538`）。TAEHV时 `build_generator` 改为 `output_type='latent'`，`samples` 才是normalized NCTHW latent；官方 `TaehvDecoder.decode` 用FP16并自行输出THWC uint8。**原示例main即使 `--no-taehv` 也设 `save_video=False, return_frames=False`，末尾只在TAEHV分支写MP4，因此必须在harness显式保存。** build固定完整Wan VAE为BF16；FP32解码若另做应作为明确变体，不能静默改后仍称原配方（入口135–170、260–295）。

- **scheduler合同与已知张力。** HF小文件 `hf_model_index.json` 是 `WanPipeline`，`hf_scheduler_scheduler_config.json` 是Diffusers UniPC/flow_prediction/flow_shift3；实际 [WanPipeline](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/pipelines/basic/wan/wan_pipeline.py:28)会替换为仓库 `FlowUniPCMultistepScheduler(shift=3)`。官方示例明确3步/CFG1，普通preset是81帧480×832，而不是我们rCM77帧4步。读取worker的 `pipeline.get_module('scheduler')` 的实际class/config/timesteps/sigmas；`TimestepPreparationStage.forward` 在85行调用原 `set_timesteps`，request不传自定义sigmas。**同commit的[QAD训练文档](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/docs/training/attn_qat.md:90)却明确学生DMD rollout `[1000,757,522]`（同目录stage2 YAML也如此），与当前推理入口的UniPC不同。** 这是公开训练配方与推理接线的张力，不能据此确定公开HF checkpoint的真实训练轨迹；按当前入口复现并记录，不自行切DMD。HF README不在小文件缓存中，本次没有独立重读它，以上判断依据实际JSON、入口和仓库文档。

- **薄观测放在worker。** 默认executor是multiprocessing，模型不在parent；parent普通monkeypatch不会证明worker路径。现成 `generator.executor.collective_rpc(callable)` 可发送可导入的顶层函数，函数首参worker wrapper经 `__getattr__` 可访问 `worker.pipeline`（`worker/multiproc_executor.py:293`、`worker/worker_base.py:72`、`utils.py:811`）。只在此读取scheduler、模块名单/精度、安装和移除计数hook；保留官方forward。DiT上按 `NVFP4QATQuantizeMethod` 枚举300目标，检查 `_fp4_weight/_fp4_weight_scale/_weight_global_sf` 且dense weight已移除；eager计 `apply` 或实际 `fastvideo_fp4::mm_fp4`，其源码最终调用FlashInfer `backend='cutlass'`（`layers/quantization/nvfp4_qat_config.py:107–140`）。加载时pack另计，不能混进forward次数。

- **attention receipt与预期调用。** worker调用 `attn_qat_infer_receipt()` 应得到 `arch=sm_120 kernel=fastvideo-kernel-cutlass scheme=sage3-fp4-sm120`，并枚举每层实际 `attn_impl`/backend；源码SM120默认 `per_block_mean=True, single_level_p_quant=True`。30个self-attention可走此FP4路径；**cross-attention LocalAttention显式只允许FLASH_ATTN/TORCH_SDPA，selector会回到dense backend**（`models/wan/transformer.py:169–175`、`attention/selector.py:316`）。故3步CFG1的静态预期是3次DiT、900次FP4 linear、90次FP4 self-attention，另90次dense cross-attention；运行计数仍需实测。receipt本身不能替代次数证明。`--baseline`只移除线性量化，未关闭环境中的FP4 attention，因此不能单凭此flag称完整BF16。

- **计时取实际边界。** `generation_time`从executor整条pipeline开始，到输出CPU拷贝结束，包含TE、denoise和full-VAE（若启用）；**原示例叫它denoise不准确**（`entrypoints/video_generator.py:785–863`）。`result.extra['e2e_latency']`另含frame processing/保存，外部wall还可含request准备。`FASTVIDEO_STAGE_LOGGING=1`才填充 `result.logging_info.stages[name]['execution_time']`；其中 `prompt_encoding_stage/denoising_stage/decoding_stage`可直接取，原机制每stage同步，应注明diagnostic timing。TAEHV decode在generator外单独同步计时。首次0 warmup/1次eager是功能smoke，不能称稳态；原main warmup还硬编码2步（237行），不要把warmup算作3步验证。所有失败留存；最后在finally调用 `generator.shutdown()` 清理worker。
