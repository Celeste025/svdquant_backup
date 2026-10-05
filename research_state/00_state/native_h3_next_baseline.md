# H3 完整 native NVFP4 baseline：最短实施方案

2026-10-02，静态审计。未运行新 GPU 计算、未安装依赖、未修改已验证源码。目标是把用户主模型带回可测的真实部署基线；既有 SVDQuant/fusion 不作为研究贡献。

**建议先做 A：保留本地 DiffSynth 的 pruned H3 前向，替换 200 个主干 linear，torch 原生 NVFP4 GEMM + 原 BF16 LR。并行只准备 B 的隔离依赖和一个 fused-up contract smoke。不要把换 serving 框架、换 checkpoint、换 rounding、换 attention 同时叠入第一条 baseline。** 上游 vLLM-Omni 确有可复用 H3/SVDQuant 实现，但当前 SM120 与我们的旧 state 都不能无条件直接加载。

## 1. 可直接复用什么

| 来源 | 已核实的可复用部分 | 本次不能省掉的验证 |
|---|---|---|
| `scripts/research/probe_h3_native_contract.py`（E005） | 真实 qkv/fc2 捕获；H3 legacy/RNE pack；独立 nibble decode；torch `F.scaled_mm` 两级 scales；实际 SM120 kernel | 只覆盖 block0 两层、两 step，未覆盖全部 200 层或 full-DiT propagation |
| `scripts/minimax_h3_svdquant_common.py` + E003 | packed token/raw-call、Comfy-pruned 模型类、正确的 pre-QDQ BF16 LR 输入、显式 torch attention | 保留原源码，不直接沿用其 disk-onload 策略作 resident timing |
| `wan_native_nvfp4.py` | PackedNVFP4 容器、128×4 swizzle、native GEMM、独立 decode、内存清理及 manifest 思路 | Wan activation packer不可直接用于H3：global clamps/算式和两级 rounding不同；Wan hook转换也不适用于H3 AutoWrappedLinear |
| E007 profiler/trace 后处理 | 独立进程计时、warmup+3 repeats、flags收集、模型/输入/source SHA、严格 Chrome `cat=kernel` 统计 | 禁止重用 raw profiler 的重复 GPU annotation 聚合；H3输出为video/audio，需要分别校验 |
| 已装 FlashInfer 0.7.0.post1 | `mm_nvfp4_svdquant` SM120 CuTe fused-up 与 unfused oracle；smooth+pack及高层linear源码 | CuTe尚未安装、尚未JIT；官方API可复用不代表本机已经运行 |

H3 recovered 环境 `/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python` CPU probe：torch `2.11.0+cu128`、`F.scaled_mm` 存在；无 FlashInfer/CuTe/vLLM；本地 Nunchaku包存在但没有已核实H3 adapter。历史 E005 已在此 torch 路径实际执行 SM120 GEMM。

## 2. 上游成熟路径：有源码，但不是当前 checkpoint 的直接入口

vLLM-Omni [PR #6162](https://github.com/vllm-project/vllm-omni/pull/6162) 于2026-08-27合入通用offline SVDQuant loader，MiniMax-H3 FL2VA是首个验证模型。可参考其[checkpoint schema](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/svdquant.md)和[真实 linear 实现](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/svdquant_config.py)，不用重写serving。该实现当前明确只启用SM103；[后续RFC #6493](https://github.com/vllm-project/vllm-omni/issues/6493)仍把SM120和native fusion列作后续阶段。不能把B300数据迁移为本机速度承诺。

代码级阻碍：`_SUPPORTED_CAPABILITIES={(10,3)}`；其BF16 LR消费**原始x**，我们的corrected state消费`BF16(x/s)`；其`input_global_scale_inv`初始化为1，serialized `wtscale`为BF16，旧H3为每输入动态FP32 gA及FP32 gW。转换`A/s`、改global存储或直接解除hardware guard均不是现有验证的等价重放。另有 pruned AdaLN curve table、state键名及token布局与官方FL2VA权重映射尚待验证。因此第一阶段借其schema/测试设计；全DiT正确性稳定后才设计exporter和一个SM120 provider适配，不先安装整个serving栈。

Nunchaku当前检索到的[H3 feature issue #948](https://github.com/nunchaku-ai/nunchaku/issues/948)已以inactive关闭，未找到可复用且已验证的H3模型adapter。此证据不证明永远不支持，但不足以把Nunchaku整模型接入列为最短路线。官方vLLM-Omni的[H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)可作为后续generation/serving参考；本次只测原cache DiT endpoint。

## 3. 旧 state 的确定契约

state：`results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt`，301,607,239 bytes，format `minimax-h3-svdquant-standard-v1`，200 layers；rank32、g16、num_grids10。保存`smooth/final_a/final_b`及诊断字段，**没有已保存的packed residual weight或其scale**。这与Wan不同：导出时必须从原BF16W依照旧GPU算术重建 `R=BF16(BF16(W*s) - BF16(B@A))`，再执行旧quantizer配方一次。此为确定格式导出，不是重新校准；但CPU的`B@A`不能未经验证替代旧GPU BF16路径。

运行时：`x_s=BF16(x/s_BF16)`；主支`Q_A(x_s) @ Q_W(R).T`；LR为两次独立BF16 linear使用原A/B，最后BF16相加。200个目标linear无须拆QKV或SwiGLU：H3原本就是fused QKV和fused fc1，各有一个global，无Wan三QKV不同global拼接问题。

H3 global为`g=max(max(abs(x)),1e-12)/2688`后再clamp到1e-12；group ideal为`max(amax(group)/6,1e-12)`；E4M3转型是RNE，E2M1显式升序codebook argmin在中点向较小数值。Wan fastpacker的分步常数乘法、SF中点向大值和E2M1规则均不同，不能只改函数名复用。可以复用Triton骨架与swizzle，另写H3专用recipe及测试。

E005证明8个测试组合legacy独立decode与旧QDQ逐元素相同；同codes native-vs-BF16-QDQ主支NMSE约2.43e-6～6.35e-6。它不证明200层传播仍小。已有Wan fullnative-vs-QDQ差异提醒必须记录完整H3 endpoint，不能用单层小误差代替质量判断。

zero/subnormal：H3的clamps不能自动杜绝极端非零group的SF下溢；必须保留计数、逐元素legacy roundtrip和finite guard，遇到不可表示旧行为则停止该layer导出，不用epsilon或canonical零静默改变定义。RNE另列为新数值arm，不覆盖legacy基线。

## 4. 实际存储预算

本次只读safetensors header和CPU mmap state，未分配GPU。设备为RTX PRO5000 **72GB**，NVML显示73,415 MiB/卡。

模型：`/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`。532 tensors实际payload40,225,668,192 bytes，混合BF16/F16/F32。按现有加载到BF16计算，DiT权重约37.460 GiB。不可误称这里是在跑完整33B未prune官方权重。

| 项目 | bytes | GiB（约） |
|---|---:|---:|
| 主干200个linear，19,267,584,000参数，旧BF16W | 38,535,168,000 | 35.889 |
| 同200层packed FP4 | 9,633,792,000 | 8.972 |
| g16 E4M3 block SF | 1,204,224,000 | 1.122 |
| 200层BF16 LR A/B | 298,188,800 | 0.278 |
| BF16 smoothing | 3,225,600 | 0.003 |
| 非目标tensor按BF16预算 | 1,687,709,488 | 1.572 |
| native合计，未计小global/metadata/padding | 12,827,139,888 | **11.946** |

非目标包含token_refiner的另外2blocks/8linears，不能误算成208个已量化目标。4类主干weight shape各50份：qkv `[21504,5376]`、out `[5376,7168]`、fc1 `[28672,5376]`、fc2 `[5376,14336]`，全部对齐满足现有native接口。

这些是resident模型tensor预算，**不是实测峰值显存**。E005峰值11.344 GiB来自disk/offload局部实验，不能当完整BF16常驻峰值。p1 packed M22400的最大fc1输出约1.20 GiB，临时buffer、attention/norm、allocator仍需记录。先排除text encoder/VAE；每层导出后释放BF16W/R与原offload wrapper引用，不允许保留整套CPU→GPU旧weights往返而标作resident native benchmark。

## 5. A：最短正确性闭环与分工

建议新增3个文件，不改common/E003/E005/Wan已验证代码：

1. `h3_native_nvfp4.py`，约250–450行：普通NativeH3Linear，保存codes/SF/gW/smooth/A/B为真实registered buffers；使用E005 native GEMM容器与独立decode；API暴露`pack_input`、`main_from_packet`、`forward_legacy_qdq`；保留slow reference接口，快packer可后接。LR显式消费x_s，不复制Wan hooks。
2. `export_h3_native_nvfp4.py`，约150–250行：逐层通过旧`load_from_disk`读取W，在GPU重建R，导出legacy codes和SF，逐200层与common.nvfp4_qdq(R) roundtrip并保存SHA/shape/数值域。export只跑一次，模型load计时不混入forward；非目标权重按原source加载。
3. `bench_h3_native_nvfp4.py`，约250–400行：复用E003完整raw kwargs/视频音频分区、torch BF16 SDPA guard，构建resident teacher/QDQ/native，记录完整endpoint和逐block诊断（诊断与计时分开）。不自行实现pipeline/serving。

可由systems负责1+2，audit负责3及独立roundtrip抽查；root负责H3 fastpack小kernel或协调B的依赖预检，三者只约定packet/manifest接口。若只两人，先不并行B，避免接口拖慢。

预估工程2–4小时；导出和初次correctness GPU预算30–60分钟，实际以分层进度为准，不承诺50blocks实时速度。先p1 step0，只有通过后追加同p1 step19；这仍是calibration-cache样本，不是heldout。快packer另预计1–2小时开发、≤20分钟小型GPU parity。

前置门槛依次为：

- 原模型与旧state/source SHA固定，明确`MiniMaxH3DiTComfyPruned`，200目标及非目标列表完整；teacher在resident图与旧disk图的同rawinput输出逐元素一致，否则先排查offload/attention变化。
- 导出200W独立解码与旧QDQ逐元素相同；smoothed input、A/B/bias语义固定。无残留AutoWrappedLinear每层disk load，全部packed buffers真正GPU常驻。
- block0四个真实linear同packet native主支vsBF16 decode main，先用NMSE≤1e-4排除scale/pack错误；全block另报BF16、legacyQDQ、native误差，不要求它们bitwise相同。
- 第一个完整50block native forward全部finite，实际调用200次NVFP4及原BF16 attention，保存video/audio endpoint和逐block差异；未达数值门槛不开始性能结论。无已建立完整native参考时先保存新SHA；之后快packer必须严格复现其SHA。
- 只有快packer真实四层输入+synthetic ties/zero/subnormal/tail与legacy codes/global/SF逐byte一致后，运行三arm独立进程1warmup+3repeat、单独profile。将模型驻留、公平attention、无磁盘onload列为硬条件；不拿slow软件pack数据评价硬件NVFP4潜力。

## 6. B：隔离 FlashInfer/CuTe fused 路径

已装源码路径：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/gemm/gemm_svdquant.py`。SM120走`cute-dsl`，要求CUDA≥12.9；本机CUDA_HOME下nvcc13.0，但torch为cu128，完整JIT兼容性仍待验证。包metadata基础依赖`nvidia-cutlass-dsl>=4.6.2a0`，cu12/cu13 extra都要求≥4.7.0a0。后续可在新隔离env先解析**精确候选`nvidia-cutlass-dsl[cu13]==4.7.0a0`**并固定wheel hash；该确切wheel可用性及torch兼容性本次未验证，不能写成已安装成功。不得浮动升级torch或在recovered env原地pip。

优先仅调用`mm_nvfp4_svdquant`：仍由A提供相同legacy packed x_s、W、down=BF16(x_s@A.T)，每次device上求`alpha=gA*gW`及`l1=BF16(B/alpha)`，传同codes的fused/unfused各一次。先隔离fused-up收益和额外数值变化；当前[官方API数学](https://docs.flashinfer.ai/api/gemm.html)是`alpha*(residual + down@l1.T)`，不能直接传B。

动态alpha使B/alpha不能只离线算一次；额外[B列数,32]重缩放需计时、检查溢出并量化BF16重舍入。fused epilogue把main/up在更高精度相加后一次输出BF16，也不同于旧两次GEMM后BF16 add。即便samecodes，不能要求/宣称bitwise等价。

暂不直接套`svdquant_linear`：它把smooth作为BF16 reciprocal multiply并折入A，down用原x；旧图为BF16除法后再用A。`nvfp4_quantize_smooth`融合后避免BF16中间值也会变化；其hardware RNE不同于H3 E2M1向较小值ties。想采用这些路径必须单列新的数值arm，对比明确软件RNE oracle、完整denoiser与随后heldout生成质量；不需要预先大QAT，但不能继承旧checkpoint所有质量结论。

B仅做qkv/fc2、rank32、M512和真实M22400两档，JIT/有限验证预算45–60分钟；import/JIT/SM120支持不通就停B，继续A，不开始自写fused GEMM。真正合入full-DiT之前，要在**同一环境**重放A，确认只有backend变化；用同codes+同LR分别报main、LR和完整linear误差，然后full-DiT propagation。serving仍用现有DiffSynth，后续需要调度/多请求才接vLLM-Omni。

本方案证明范围仅为工程可行性规划。Wan native慢说明需要完整profile和成熟fusion对照，不提供“新方法”的证据，也不能外推H3必然慢或快。
