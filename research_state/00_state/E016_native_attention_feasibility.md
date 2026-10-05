# E016：现成 SM120 NVFP4 attention 完整对照可行性

2026-10-02，CPU-only 只读核查。**结论：最小替换可行，ready for implementation；本轮未写 runner、未安装依赖、未运行 GPU。** 这是已有官方技术的完整强基线，不是新方法。

## 已核实环境与真实输入

直接使用 E006 环境 `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python`。本次在 `CUDA_VISIBLE_DEVICES=''` 下成功于同一进程导入 E014 runner、H3 native wrappers 和 FlashInfer attention：torch `2.11.0+cu128` 实际来自 `/home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/torch/__init__.py`，与 E014 一致；Triton `3.6.0`，FlashInfer `0.7.0.post1`，`F.scaled_mm` 与两级 scaling API 均存在，CUDA 未初始化。无需 CuTe、vLLM、新环境或额外安装。

E006 已在此环境同时实际运行原生 FP4 projection 与 `nvfp4_attention_sm120_fwd`，有 SM120 E2M1 GEMM 和 `nvfp4_attention::attention_kernel_ws<...float_e2m1...>` profiler 证据；12 个真实 H3 block-case 完成。它验证的是 `per_block_mean=True` 局部路径，**没有验证 False 或当前200层 legacy量化图的完整运行**。新实验必须保留这两个边界。

本次 CPU 读取 E014 三个真实输入、调用原 packed builder 得到：

|输入|原 main cu_seqlens|有效长段 N|原 padding 段|FlashInfer 内部 N_pad|
|---|---|---:|---:|---:|
|p1 / s0|[0,22384,22400]|22384|16|22400|
|p30 / s5|[0,22227,22272]|22227|45|22272|
|p36 / s14|[0,22539,22592]|22539|53|22656|

每层 main Q/K/V 都为 `[M,56,128]`，由原 Comfy QKV 顺序、QK RMSNorm、RoPE 生成。原 helper 按 cu 分段，不跨段计算；没有 causal 或额外 attention mask。两个 refiner blocks 的 cu 为 `[0,text_len,text_len]`，text_len 分别 658/501/813，因此各仅一次非空 BF16 SDPA。原路径合计 50×2+2=102 次；替换后应为 **50 次官方 FP4 attention + 52 次真实 BF16 SDPA**，原 padding 不成为长段有效 key。

## 最小实现路径

1. 复用冻结 E014 loader、resident conversion、200层 SVD packets、fastpack、原 model_fn/input 构造及 RuntimeAudit。在新文件里为50个 main attention 的原 bound forward 添加轻量作用域标记，再用作用域路由 `comfy._sdpa_varlen_attention`；QKV/norm/RoPE/out_proj 原 forward 不重写。refiner不打标记，直接调用原 helper。不要按“长度较长”猜层身份。
2. 对被标记 main helper，检查 cu 正好等于当前输入的既定三项，再取 `[0:N]`，转为官方所需 contiguous `[1,56,N,128]`。调用 `quantize_qkv(q,k,v,per_block_mean=mode)`，随后 `fwd(*packet,sm_scale=原softmax_scale,causal=False,per_block_mean=mode,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)`。裁 Q 输出回 N，转回 `[N,56,128]`。布局转换、centering、QKV packing、FP32 correction、原生 kernel 和裁剪回填全部计入完整前向，不能预pack假称部署速度。
3. 原 padding `[N:M]` 切片调用**原 helper**，以局部边界 `[0,M-N]` 映射相同独立序列；局部 cu 可为 CPU int32，因为原 helper 仅 `.tolist()` 读它，不必额外设备分配。结果回填原位置。refiner保持原 cu/原 helper。路由只改变50段的实现；没有捕获异常后偷偷回退。
4. `RuntimeAudit` 继续核200个实际 `F.scaled_mm`、原BF16 dtype与零disk；低精度路径另计50个quantize/fwd对，BF16 SDPA必须52。原BF16 attention臂仍102。独立 profiler 须同时出现200个主层 E2M1 GEMM和50个官方FP4 attention kernels。检查refiner未替换、原softmax_scale=`128**-0.5`、输入/输出layout及长度；非法值或计数不符停止。

## 必须保留的数值与来源合同

- 在这个**同一 E006 环境**先做原BF16整模三状态重放，raw/velocity与E014逐byte一致；再做SVD+BF16-attention三状态，亦须与E014 SVD逐byte一致。只验证teacher不够：否则环境导致的原生主层漂移会混入attention差异。计入计划的3+3调用，不加探针。
- True/False 两臂均从自己当前层、当前轨迹的真实 QKV 重算 mean/scales/correction；不能共享teacher或另一臂统计。
- 官方 False **仍先对未padding K做centering，且对padding后的Q取一次全序列mean**；True亦在padding后按128组处理Q。False不是no-centering。保持官方顺序，`unpadded_k_len=N`排除API新增key padding，不能把原M或N_pad误当N。True/False是两套完整官方数值配方，不能将差异单独归因某一个统计量。
- 固定 `allow_tf32=False`、原BF16 SDPA backend flags、原headscale以及H3 legacy linear recipe。所有三个状态、两输出模态均报，不以某一个结果选臂。

## 执行规模与风险

按root方案：原BF16 teacher重放3次；SVD固定三attention臂各3次eval；三个SVD臂各1warmup+3repeat+1独立profile，共 **27 DiT**，同一空闲GPU、30分钟共享deadline、最多30calls/60GiB。所有输入、模型/packet与官方源码hash固定，原模型fullSHA沿用E010 inventory+size/mtime；GPU运行前先完成新源码CPU检查。

False尚未GPU执行，首个完整eval本身就是合法性门槛，不另开benchmark网格。True默认FP32 correction在p1为 `[1,56,175,22400]`，约0.818 GiB，False为 `[1,56,1,22400]`，约4.79 MiB；但两者仍有contiguous QKV、FP32 K等临时缓冲，因此必须测整个模型峰值，不能只报correction大小。已安装/JIT cache可降低启动成本，不保证完整模型速度或30分钟一定完成。若既有ABI、尾部mask或重放不成立，停止并记录，不进行复杂移植或自写kernel。

## 已读取的来源

- 原 H3：`/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:_comfy_attention_forward`，`minimax_h3_dit.py:_sdpa_varlen_attention`，`pipelines/minimax_h3_audio_video.py:model_fn_minimax_h3`（refiner cu 构造）。
- 本机官方接口：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/nvfp4_attention_sm120.py`，SHA `040738c6d6af5584ca9b375bb59468d7011506862ff0a162f1f51a72149abefc`；`_preprocess_qkv`、`nvfp4_attention_sm120_quantize_qkv`、`nvfp4_attention_sm120_fwd`。支持SM120/121、BF16/FP16、head_dim64/128，API明确支持True/False与真实unpadded K length。
- 既有运行证据：`results/research/E006_h3_attention_interface.json`（complete、12case、实际kernel），`scripts/research/smoke_nvfp4_attention.py`、`launch_E006_attention_smoke.sh`（原cache/CUDA环境变量），以及冻结E014三状态manifest/evaluate。
