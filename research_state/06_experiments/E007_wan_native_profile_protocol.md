# E007 — 完整 Wan native baseline profile 协议

状态：frozen / ready to execute（2026-10-02）。只测既有部署实现，不作研究贡献或生成质量证据。

固定 E007 原样本 `0001-00000-0.pt`、31,200 视频 tokens/512 text tokens、rCM Wan1.3B 30 blocks；torch2.11.0+cu128、Diffusers0.33.1、相同 BF16 torch `SDPBackend.FLASH_ATTENTION`。不加载 text encoder/VAE，不运行4步rollout。

三 arms：BF16、历史 QDQ、native fast。**同一 GPU0 串行三个独立进程**，命名 tmux 与独立日志，启动前检查GPU空闲。每个进程1次完整 warmup、3次无profiler重复、1次独立profiler。GPU运行须由root确认；本文件落盘不自动启动任务。

每次输出（包括warmup和profiler）必须直接匹配 `E007_wan_native_correctness.json` 对应完整 endpoint SHA；native对应 `stages.native_full.endpoint_sha256`。native warmup精确通过后不重跑慢packer。失败即保存并停止性能结论，不能换阈值继续计时。BF16复用warmup与原cache比较，不额外执行两轮cache验证。

native以旧loader重建语义、300 saved weights roundtrip后转换，逐300 quantizer调用root的 `validate_quantizer_contract`，再赋 `pack_activation_fast`。冻结 `E007_fastpack_real_parity.json` 的source hash，若当前source不符则停止。依赖完整E007和真实三activation+七synthetic的byte-parity门槛完成。

计时的每次forward都包在 `collect_fastpack_checks()` 内，CUDA结束event与同步在scope内；scope退出验证全部300 flags后才接收数据。报告CUDA event、forward host（不含flags验证）、flags验证耗时，以及**含验证的总host wall**各自中位数/min/max和原始3次数据。flags kernel和Python收集成本包含在forward，检查成本不隐藏。timing区间内无逐层metrics、CPU capture、SHA或输出保存。

显存：独立列startup/load/conversion峰值；warmup后reset peaks，记录3次稳态allocated/reserved峰值和前后allocator状态。常驻storage统计按data_ptr去重，包含registered parameters/buffers及hook.branch内LR A/B、hook.processor smoothing tensors；它不代替真实allocator指标，也不包含CUDA库context。保留旧BF16 LR，不融合或共享down，不改变旧recipe。

独立profiler才安装 `record_function` 标签：main BF16 GEMM、native GEMM、smooth、pack、LR（包含down/up和output add）、attention。这些语义子范围互不嵌套；完整forward为父范围，不再与子项相加。导出CUDA kernel聚合、CPU runtime launch/sync聚合与operator self CPU time。未分类工作保留other；device工作总量不是host关键路径，也不能在有重叠时按比例机械推算潜在加速。

Chrome traces和endpoint snapshots放 `/data1/models/svdquant-wjq/research/20261002/E007/profile/`；仓库 `results/research/E007_profile_{bf16,qdq,native}.json` 保存聚合、SHA与路径。源码、协议、模型、checkpoint、输入与前置报告SHA记录。加载/JIT不进入稳态latency，单列startup和warmup成本。

启动入口（每个arm单独进程，必须串行）：

```text
/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python -u scripts/research/profile_wan_native_nvfp4.py --arm bf16
/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python -u scripts/research/profile_wan_native_nvfp4.py --arm qdq
/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python -u scripts/research/profile_wan_native_nvfp4.py --arm native
```

环境设 `CUDA_VISIBLE_DEVICES=0`、历史env/bin进PATH、`CUDA_HOME=/usr/local/cuda`、`SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq`。单进程timeout 1800秒，三个进程总计最多90分钟；失败保留输出与日志，不覆盖，不并行占CPU改变latency。暂不写吞吐加速结论，等待三arm同契约结果。
