# Wan1.3B 既有 checkpoint 的原生 NVFP4 部署可行性

2026-10-02。范围：只读源码/环境/CPU mmap 检查；没有启动新 GPU 工作，没有修改环境、E005/E006 或实现完整模型。定位是工程复现和真实速度/显存 baseline，不是方法贡献。

## 决策

**已有 checkpoint 可以导出 packed residual weights，并保留 smoothing/LR 参数；不需要先重跑 PTQ。可以构建完整 DiT 的原生 W4A4 linear baseline，但 FlashInfer 三个高层 SVDQuant API 不是目前可直接替换旧 hooks 的等价实现。** 最短路径先用现有 CUDA NVFP4 quantizer + 已验证的 PyTorch SM120 `F.scaled_mm` + 独立 BF16 LR；确认数值后再比较 CuTe DSL 融合。所谓完整 DiT 指其 300 个目标 linears 使用 native residual GEMM，attention、norm、embedding、time modulation 等继续原有精度，不能称所有算子都是 FP4。

有两个独立门槛：

1. **运行依赖：** 已安装 FlashInfer 0.7.0.post1 的源码确实实现 SM120/SM121 fused SVDQuant；但 E006 隔离环境缺少 `cutlass` / `nvidia-cutlass-dsl`，CPU probe 的 `is_cute_dsl_available()` 为 false。这些 API 当前不可调用，不能因为 E006 attention 已成功就推断 GEMM 也已成功。
2. **量化契约：** 旧 Wan 的 smoothing 精度、E4M3/E2M1 tie rule、动态 global scale、LR 输入与输出舍入都必须独立核对。即使全部接口可运行，也不能直接声称与历史生成结果数值等价。

## 已核实的环境与 API

隔离 Python：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python`。torch 2.11.0+cu128，FlashInfer 0.7.0.post1，apache-tvm-ffi 0.1.14.post1，Diffusers **0.40.0**。历史 PTQ 环境 Diffusers 为 0.33.1，版本差异须先解决；不得将换 Diffusers 的行为差异记到量化上。建议后续单独 Wan native 环境或严格固定兼容版本，同一个环境跑全部对照；不在正在执行 E006 的环境上原地升级依赖。

`gemm/gemm_svdquant.py` 的实际 dispatch：

| API | SM120 路径与契约 | 现状 |
|---|---|---|
| `mm_nvfp4_svdquant` | `cute-dsl` fused 或 `cute-dsl-unfused` oracle；SM100 CUTLASS 后端不能强行用于 SM120 | 缺 CuTe DSL，未 JIT/未跑 |
| `nvfp4_quantize_smooth` | SM120 CuTe DSL 融合 BF16 channel multiply 与 NVFP4 pack，scale vector 16、128×4 swizzle | 同上 |
| `svdquant_linear` | smooth+pack，然后独立 BF16 down GEMM，再 residual+up fused GEMM；不是一个全融合 kernel | 同上 |
| `flashinfer.quantization.nvfp4_quantize(..., backend="cuda")` / `fp4_quantize` | 存在不依赖 CuTe DSL 的 CUDA pack 路径 | 源码可达，普通 GEMM pack API 尚未在本审计实测；E006 attention pack 是另一接口 |
| `torch.nn.functional.scaled_mm` | E005 已实测 SM120 FP4、两级 FP32 global + E4M3 scale | 可作为首个 GEMM baseline，无需 CuTe DSL |

SM120 SVDQuant guard 要求 CUDA≥12.9；FlashInfer `get_cuda_version()` 首选 nvcc，在 `CUDA_HOME=/usr/local/cuda` 下本机 CPU probe 为 **13.0**，因此 torch 编译版本 cu128 不直接导致此门槛失败。仍需一次有限 JIT 验证才能确认整套 runtime 兼容。包 metadata 基础依赖列 `nvidia-cutlass-dsl>=4.6.2a0`，`cu13` extra 列 `nvidia-cutlass-dsl[cu13]>=4.7.0a0`；后续应固定确切安装版本与缓存，不安装浮动最新版后直接测性能。

`mm_nvfp4_svdquant` 接受 A `[M,K/2]`、B `[N,K/2]` contiguous uint8，E2M1 双 nibble；E4M3 scale 为 uint8 128×4 swizzle，不是线性 layout。N、K 必须整除 32，LR rank 为正的 32 倍数（官方验证 32–128），BF16 down `[M,r]` 与 up `[N,r]` contiguous，bias `[N]` BF16，输出 BF16。Wan 的三种实际矩阵 `[1536,1536]`、`[8960,1536]`、`[1536,8960]` 和 rank32 均满足静态 shape 门槛；M=31200 的 tail 要在首次真实 smoke 中验证。

官方 API 可供复核：[mm_nvfp4_svdquant](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_nvfp4_svdquant.html)、[svdquant_linear](https://docs.flashinfer.ai/generated/flashinfer.gemm.svdquant_linear.html)、[nvfp4_quantize_smooth](https://docs.flashinfer.ai/generated/flashinfer.gemm.nvfp4_quantize_smooth.html)。网页版本标签/缓存并不完全一致，本结论以本机 0.7.0.post1 源码为准。

## checkpoint 实际保存了什么

路径：`/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16`，模型是 rCM 4-step Wan2.1-1.3B，30 blocks。

- `model.pt`：825 tensors；300 个目标 linear 的 **已反量化、已平滑、已减 LR 的 residual BF16 weights**，以及 bias 和未量化模型参数；不是 packed FP4，也不是原始全精度 W。
- `scale.pt`：900 entries，每个目标 weight 有 FP32 global `scale.0`、以 FP32 容器保存的 E4M3 block values `scale.1`、zero。300 个 zero 全为 0。global 范围约 1.63487e-5～6.771997e-4。
- `wgts.pt`：300 个模块配置，抽查 q 的 scale/zero/dynamic_range/range_bound/quant_range 都为 None；它不是 packed 数据。
- `smooth.pt`：210 个 channel vectors + `proj.fuse_when_possible=False`。此 checkpoint 在 activation 上保留显式 smoother。
- `branch.pt`：210 组 BF16 A/B，rank32。self-attn QKV 共用 A，B 为 `[4608,32]`，按 q/k/v 切三段；cross-attn KV 共用 A，B `[3072,32]` 切 k/v；其余单独 branch。不能按每个 linear 名直接要求一一对应的 branch key。

CPU-only 用 mmap 抽查 block0 `attn1.to_q` 和 `ffn.net.2`：固定保存的 global/block scales 后恢复最近 E2M1 code，再重建并转 BF16，分别 **2,359,296/2,359,296**、**13,762,560/13,762,560** 元素与保存的 BF16 residual 完全一致，max_abs=0；两层 block scale 转 E4M3 再回 FP32 全等且无 zero。这个检查只覆盖两矩阵，不声称全部 300 个已验证。正式导出必须逐层做该 roundtrip，不重新从 residual 的 absmax 求 scales，否则可能改变 checkpoint。

静态存储量：300 weights 共 1,391,984,640 参数，BF16 为 2,783,969,280 bytes；packed FP4 695,992,320 bytes，E4M3 SF 86,999,040 bytes，另有非目标 model tensors 60,278,912 bytes 与共享 LR 78,643,200 bytes。约从 2.72 GiB 参数存储降至 0.86 GiB，加少量 globals/smooth/padding；这是布局预算，**不是实测峰值显存**。完整 77-frame 480×832 的 attention/activation、allocator、workspace 仍可能主导峰值。必须释放 BF16 residual 和 GPU export 临时副本才会兑现该存储减少。

## 必须保持的旧执行语义

`infer_rcm_wan_4step.load_quantized_transformer` 先用 smooth cache 重建 graph，再调用 `load_diffusion_weights_state_dict` 注册 branches 并覆盖 `model.pt`，最后注册 activation quantizers。不要重复给 `model.pt` 乘 smooth 或再减一次 BA。

对于已经到达 linear 的输入，旧流程等价于：

```text
x_s = bf16(x / bf16(s))                 # 相应 smoother，部分在 parent attention 上
branch_input = x_s                      # AccumBranchHook prehook 保存
x_qdq = QDQ_legacy(x_s)                  # 后注册 ProcessHook，返回新 tensor
main = bf16_linear(x_qdq, W_saved, bias)
y_old = bf16(main + bf16_linear(bf16_linear(x_s, A), B))
```

`ActivationSmoother` 默认 develop dtype 是输入 dtype，因此 BF16 x 会先将 smooth vector 转 BF16 再除法。Q/K/V smoothing 可能挂在 **parent attention**；cross KV、out、FFN 又有不同挂载位置。最短 native adapter 应在原 graph 完成重建后只替换目标 linear 的 activation quantizer/residual/branch计算，**先保留 parent smoothing**；此时 native linear 接收到的已经是 x_s，额外 `pre_quant_scale` 应为 ones。不要 indiscriminately 清空所有 hooks，也不要把 smoother留着同时再次传 `1/s`。

### 六项具体差异

1. **Wan tie rule 与 H3 不同。** DeepCompressor 的 C++ `nearest_neighbor` 在 `bnb=false` 默认时用 `x+x < low+high ? low : high`，精确中点向较大数值；同一 codebook route 也用于 E4M3 scales。H3 历史 `argmin` 向较小数值，E005 的 tie 数量/影响不能直接外推 Wan。FlashInfer 原生 E2M1 `cvt.rn` 是 RNE，E4M3 conversion 也须检查。保存的 weight codes/scales 可保持，动态 activation 则必然要做本模型合同对照。
2. **global reciprocal 方向和动态性。** 记 saved weight decode global 为 gW，activation decode global 为 gA；native pack 的 `global_scale` 是编码乘数 G=1/gA，常用 `2688/amax(x_s)`；G 不是 GEMM 的 alpha。主支 `alpha=gA*gW`，每次输入变化都要在 device 上重算，计入时间，不能 `.item()` 同步或默认设 1。全零/scale underflow处理也需匹配；legacy `QuantScale.remove_zero` 会修改聚合 effective scale 为1，不能假定与原生 zero-SF规则相同。
3. **融合 LR 的 alpha 会影响 branch。** `mm_nvfp4_svdquant` 数学是 `alpha*(residual_unscaled + down @ l1.T)`，所以必须传 `l1=bf16(B/alpha)`。既有 dynamic gA 意味着此 l1 不能只离线预处理一次；每次更新 `[N,32]` 很小，但多一次 kernel 和 BF16重舍入，必须计时/校验。直接用 B 或固定旧 alpha 会给 LR 乘错尺度。若改为固定 activation global，就已改变量化 recipe，需要用校准集确定值并做 heldout 质量验证，不是格式转换。
4. **融合 smoothing 与 LR down 不逐位等价。** SM120 helper 实际是寄存器内 `bfloat2_mul(x, pre_quant_scale)`，仍有 BF16乘法舍入，只是不物化中间张量。`bf16(x / bf16(s))` 与 `bf16(x * bf16(1/s))` 未必相同；把 reciprocal折入 A 后用 `x @ bf16(pre_scale*A.T)` 又改变 GEMM输入和舍入。不能把这些变化说成“纯 kernel 替换”。
5. **epilogue 与 bias/add 顺序。** fused up 加到 FP32 accumulator 后一次转 BF16；旧路径是 main（含 bias）先转 BF16、LR down/up 分别 BF16、最后 BF16相加。native 与 BF16 dequant operand GEMM 本就还存在 scale/global应用次序差异。验证时至少分开同codes主支误差、branch误差、完整linear误差。
6. **QKV 合并会遇到不同 gW。** 虽然 QKV 可共享 x_s/packed A/LR down，三份 saved weights 有各自 tensor global。一个拼接 B + 一个 alpha 不自动保持它们原scales。第一版保持三次原生 residual GEMM；如后续合并需证明 E4M3重缩放可精确表达，不能无声重量化。

## 最短可执行路径与是否需要重标定

**路径 A，先完成可测 baseline（推荐）：**

- 一次性导出 300 weights 的原codes、swizzled E4M3 SF、FP32 gW、bias、共享 A/B 和 hook mapping；未量化 tensor从model.pt原样保留。全300 roundtrip通过才部署。导出可CPU分块，运行时不保存 BF16 residual 副本。
- 保留旧 BF16 smooth；native wrapper内部从 x_s 求 gA，以 FlashInfer `backend="cuda"` pack RNE activation，调用两级 scale 的 `F.scaled_mm`，独立 BF16 A/B branch。bias/add按固定契约实现；把 activation RNE 与旧 tie rule 的数值变化标明。
- 在同一输入上另有明确软件 RNE QDQ oracle，与 native 主支先对齐；原有 DeepCompressor完整结果作为第二参考。这样先去掉多路Python QDQ/dequant与BF16大 residual GEMM，保留少量 elementwise 与两次小 BF16 LR GEMM。
- 该路径不需要重新训练 smooth/LR 或重标定 weight；需要测量激活舍入造成的偏差。若应用要求严格复现 legacy dynamic量化，须额外实现同 tie/zero/FP32除法规则的 GPU packer；这属于部署兼容工程，不能用 H3 packer直接顶替。原生 FP4 GEMM仍不可保证 BF16-QDQ bitwise相等。

**路径 B，可选性能候选：** 在独立 Wan native 环境补 pinned CuTe DSL，有限 JIT 后用 `mm_nvfp4_svdquant`，仍先给它既有 x_s、down、每次重算的 `B/alpha`。这可以单独量化 fusion收益，不必同时改 smoothing。只有 A 的完整DiT正确性通过、性能说明 LR/up 是瓶颈时，再尝试 `svdquant_linear` 将smooth/down变换折叠；做配对误差与质量检查。无证据必须重校准；若融合后误差不可接受，再评估重新校准，不能承诺“不用改任何数值”或直接覆盖旧checkpoint。

**需要重新确定校准配置的变更：** static activation global、重新计算weight scale/clip、改变 group size、重新求 smooth/LR、将QKV tensor globals合并到不精确的共同SF。这些均超出本次格式导出。旧 checkpoint 质量结果不能直接继承给它们。

## 下一实验的最短闭环（按优先级收敛）

**第一步：先获得完整 DiT 正确性闭环，允许 packing 慢。** 固定历史 Diffusers 与同 attention，原loader重建后保留 smoothing，把300目标linear转换为 packed static weights + torch原生FP4 GEMM +独立BF16 LR。activation 首版按DeepCompressor真实FP32算式生成 scales/codes（可用软件reference），不要求快；它只用于核实全部30blocks的hook/参数/scale布局和输出，**不报告该慢packing实现的加速比为最终baseline**。经过完整DiT endpoint之后，再让同一个wrapper切换快速packer；不要先同时改smooth、LR共享计算和融合epilogue。

**第二步：让 activation packing 真正可计时。** 两条路线用途分开：

- FlashInfer普通CUDA量化API可以提供最快工程起点，但其RNE/fast-math/zero处理不是DeepCompressor的默认数值契约。它只能作为明确标记的RNE native baseline，并与旧recipe软件reference配对校验；启用 `FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` 也不自动修复ties规则。
- 若目标是保留旧recipe，最小Triton实现可限定为两个阶段：device上求与旧代码**相同算式次序**的FP32 global；每个16元素组计算FP32局部amax、用向较大值的E4M3 midpoint规则取SF、以相同FP32 effective-scale除法做向较大值的E2M1编码、输出swizzled SF及packed uint8。不能把 `amax/6/448` 无声改写成 `amax/2688`，也不能把division换近似reciprocal后宣称字节一致。E4M3正数可用指数/尾数逻辑实现向上tie，E2M1用7个正midpoint和有符号规则；不是大codebook broadcast。具体实现、全零和underflow处理仍待与legacy CUDA extension逐字节验证。

首个Triton packer不融合smooth，直接消费原BF16 x_s；保持旧branch输入与dynamic global。需要包含全零、E4M3 subnormal/zero、所有E2M1/E4M3中点、真实q/fc2、M31200和M512尾块测试。必须先确认native-vs-reference差异只剩GEMM算术，而不是packing错误；该实现属于已有量化配方的部署，不作新贡献。

**第三步：再考虑融合。** 高层`svdquant_linear`带来reciprocal smoothing、LR权重折叠和epilogue舍入变化，不能直接套。先拿路径A完整profile决定是否值得补CuTe依赖/融合工程。

### 共享 LR 是否已经省掉重复 down

**没有。** `calibrate_diffusion_block_low_rank_branch`确实把self-QKV/cross-KV的`branch.a`指向同一module，但每个目标linear分别注册一个`AccumBranchHook`。每次posthook调用各自`branch(self.tensor)`，而`LowRankBranch.forward`每次都执行`b(a(input))`，没有输入memoization。因此旧self-QKV执行3次down，cross-KV执行2次down；210是branch cache分组数，实际300目标linear的hook仍分别执行。它不是已优化的共享计算。

第一版native仍保留各自down调用以便校验；随后可在同一次parent attention forward内明确共享**相同x_s对象/数值**的BF16 down，scope严格限单次调用，验证结果和hooks生命周期，不做跨step缓存。对应weight residual仍分别GEMM，因为gW不同。QKV合并到一个packed GEMM需处理不同weight globals；这不是仅cat weight就能保持旧码值/scales的操作。是否做共享down由profile决定，非建立全DiT baseline的前置条件。

## 半天以内最小 benchmark（建议，未执行）

固定单卡 GPU5，启动前 `nvidia-smi`，命名 tmux+`results/logs`，硬墙钟 **4小时**。不触碰 E006 GPU0；不用 autotune扫全模型、不跑 VBench、不先做视频解码。

| 阶段 | 墙钟预算 | 交付/止损 |
|---|---:|---|
| CPU导出/环境固定 | 30分钟 | 检查300层原weight BF16 roundtrip、shared branch切片、同Diffusers版本；失败停止模型替换 |
| 普通CUDA pack + torch native GEMM contract | 30分钟 | block0 q、fc2真实BF16 upstream；完整M=31200及cross KV M=512；zero/ties/tail synthetic；同codes软件参考+profiler SM120证明 |
| 可选CuTe依赖/JIT门槛 | 最多45分钟 | 一个rank32 shape和bias/tail能跑、fused vs unfused数值通过；失败直接保留路径A，不做一天kernel工程 |
| 一block与30block集成校验 | 60分钟 | 同输入torch SDPA、baseline/candidate分别进程；先block再DiT endpoint，保存每层差异避免最后才发现hook错位 |
| 测时与显存报告 | 45分钟 | 分层完整linear、完整DiT、完整4步denoise，保存所有执行成本；无质量指标冒充 |
| 分析/日志/hash | 30分钟 | 总计≤4小时；没过完整DiT则只报告已通过层/block，明确未完成部署 |

样本固定现有rCM contract：batch1，480×832，77帧，4steps，sigma_max80，guidance0，seed42，同一预指定prompt。latent patch token约31200、text512；通过实际 capture再次确认。prompt embedding预先一次生成并保存，text encoder/VAE在三个backend测量中都移出GPU；另行报告是否包含它们。

三条主对照：原BF16、既有DeepCompressor QDQ checkpoint、native checkpoint。若CuTe路径通过则为第四条。所有路径固定 attention backend/SDPA dtype、torch设置、分辨率、timestep、输入和weights；不能将原BF16和native放不同软件版本比较。独立进程记录 startup/load/export/JIT成本；稳态timing排除首次编译，但把每次的 smooth、amax/global reduction、quantize/pack、GEMM、LR、bias和临时分配全部包含。

测量层级：

- q/fc2的完整linear：预热3次、重复20次（如QDQ过慢则限制总时间并保留次数）；另看cross KV M512。CUDA event与host同步wall time都报，不只报预pack GEMM。
- 完整DiT：同一缓存raw input预热1次，重复3次，记录中位数和范围；最大allocated/reserved、NVML进程显存另列。保存native相对QDQ/BF16 endpoint NMSE。
- 完整4-step denoise：每路径同seed完成一次暖运行后3次重复（预算不足至少1次并标明）；报告4步合计时间及最终latent差异。到此仍是**denoiser吞吐**；只有加入text encoding和VAE decode才可声称完整video pipeline latency。

首层同codes native-vs软件主支误差预期是舍入尺度，但不能沿用E005数值作为本实验结果。可先用NMSE≤1e-4做packing/scale粗门槛，过门槛并不证明生成质量。对原legacy结果另报变化；若误差跳变或层间突然放大，先查global方向、bias、double smooth、LR输入、shared B切片，而不是重校准掩盖实现问题。

## 需要新增/修改什么（未实施）

建议新增 `scripts/research/export_wan_native_nvfp4.py`（CPU导出与roundtrip）、`wan_native_nvfp4.py`（wrapper与明确hook移交）、`bench_wan_native_nvfp4.py`（复用rCM schedule、matched inputs/profiler/time/memory）。先不改历史 `load_quantized_transformer`；新runner通过现有loader重建语义、捕获实际hook并转换，运行时加载导出轻量状态。长期可单独新loader避免DeepCompressor初始化成本。不要改common历史QDQ或E005/E006。

本次源码指纹：

```text
scripts/infer_rcm_wan_4step.py
9212636afac8d7c98a82f3fb98c6efb0a0ed366ea83db65866f64cfcf99e02fb
third_party/deepcompressor/deepcompressor/csrc/quantize/quantize.cu
df5086015798a594699a17b7a3743c0e9b96e5a75d8c7fe1e289643e40ca0b8f
FlashInfer 0.7.0.post1 gemm/gemm_svdquant.py
cec596b89f998d49a4ecf4947517963f4dd845e903be8533da18bf5e44fbc6da
```

证据边界：本报告的SM120 fused支持是源码与官方接口证据，不是成功kernel调用；CPU exact-recovery只覆盖两层，不是全checkpoint证明；存储数字是静态容量，不是速度/峰值显存结果。当前没有实现部署，也没有产生新的GPU benchmark。
