# NVFP4 系统可行性审计 — 2026-10-02

**结论：当前环境可直接调用真实 SM120 NVFP4 tensor-core GEMM；旧 H3 的 BF16 QDQ 不能证明真实 W4A4 加速。** 已完成两个小矩阵与两级 scale 的 correctness smoke，并捕获 native kernel 名称。额外的 global-scale 控制仅是小型 CPU 合成实验，不是视频模型研究发现。本轮未安装包、未修改全局环境、未加载模型。

## 1. 硬件、软件和当前可用路径

- 8 × RTX PRO 5000 72GB Blackwell，驱动 580.126.20；实测 compute capability `(12,0)`。
- 初次快照 GPU 0/1/5 空闲，2/3/4 已用 53.8/50.6/48.1 GB，6/7 已用约72.6/72.5 GB且100%利用率。资源状态会变化，不能沿用“0–5全空闲”的旧记录。只使用 GPU5 做秒级 smoke，恢复核查时 GPU5 为0 MiB/0%。
- 拓扑：0–3位于NUMA0，4–7位于NUMA1；组内 NODE、跨组 SYS，未见 NVLink。多卡通信成本不能套用NVLink服务器数据。
- `svdquant-ptq` 和 `mjvideo` 环境均为 Python3.12.13、PyTorch2.11.0+cu128、Triton3.6.0。PTQ 环境的torch架构列表含sm120，Diffusers0.33.1。
- 上述两个环境未装 Nunchaku、FlashInfer、SageAttention、FlashAttention、TorchAO；PTQ环境也未发现Transformer Engine或CUTLASS Python。仅核查本项目环境，没有遍历其他用户环境。
- 系统nvcc13.0.88，另有CUDA11.8；torch wheel运行时为CUDA12.8。未来编译扩展需固定工具链，不能视为天然ABI兼容。
- 仓库旧Nunchaku的`setup.py`仅显式sm86/sm89；`src/kernels/zgemm/gemm_w4a4.cuh`使用`mma...s4.s4`/`u4.s4`整数指令，且本机未安装其扩展。不是当前可调用的SM120 NVFP4实现。

| 路径 | 本机状态 | 适合做什么 |
|---|---|---|
| `torch._scaled_mm` / `F.scaled_mm` | **真实SM120 FP4已实测通过** | 原生linear correctness与模型集成的最短路径 |
| H3 `nvfp4_qdq` + `nn.Linear` | BF16 QDQ | 算法初筛，不作为NVFP4吞吐/内存证据 |
| 仓库旧Nunchaku | 未构建，旧INT4路径 | 不能仅改架构编译参数就称NVFP4 |
| FlashInfer `mm_fp4` | 当前官方支持SM120，本机未装/未测 | 以后在独立环境评估kernel效率 |
| FlashInfer SM120 NVFP4 attention | 当前官方提供，本机未装/未测 | 非首次实现机会，仍需核对H3 packed/mask语义 |

[FlashInfer mm_fp4官方API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)列出SM120 dispatch与scale layout。[SM120 NVFP4 attention官方API](https://docs.flashinfer.ai/generated/flashinfer.nvfp4_attention_sm120.nvfp4_attention_sm120_fwd.html)已提供packed Q/K/V、转置V/scale、QK correction与causal选项；这不代表H3的packed text/video/audio、变长mask可直接替换。CUTLASS也已有SM120 dense/sparse NVFP4示例：[官方changelog](https://docs.nvidia.com/cutlass/4.3.1/CHANGELOG.html)。

## 2. 已复核的真实kernel证据

源文件：`scripts/research/smoke_nvfp4_systems_20261002.py`；原始JSON：`research_state/00_state/systems_smoke_20261002.json`。恢复后已重新读取，文件完整。没有重新跑相同测试。

```bash
nvidia-smi --id=5 --query-gpu=index,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=5 /data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python \
  scripts/research/smoke_nvfp4_systems_20261002.py \
  --output research_state/00_state/systems_smoke_20261002.json
```

输入直接生成packed E2M1 codes与非均匀E4M3 block scales，不涉及activation quantizer。独立unpack/dequant后分别进行FP32与BF16 reference GEMM，TF32关闭。

| API/scale | M/N/K | native对FP32 reference NMSE | native对BF16 QDQ NMSE |
|---|---|---:|---:|
| `_scaled_mm`，global=1 | 128/128/256 | 2.714e-6 | 0 |
| `_scaled_mm`，global=1 | 257/256/512 | 2.720e-6 | 0 |
| `F.scaled_mm`，block+tensor两级 | 128/128/256 | 2.725e-6 | 1.2404e-5 |

Profiler捕获：

```
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

这是SM120 E2M1×E2M1、UE4M3 block scale、FP32 accumulate/BF16 output kernel；名称含bias/relu不表示本次请求ReLU，正负输出均由数值对照验证。

两级scale使用global=0.0137/0.00231时，53.44%的BF16输出元素与BF16 QDQ不同，NMSE仅1.24e-5。QDQ先将`code × block_scale × global_scale`舍入到BF16，而native执行次序不同。这是实现契约差异，**不能据此声称模型质量显著下降或形成论文发现**。global=1时完全一致也不能外推所有shape/值域/accumulation策略。

## 3. 格式与rounding契约

1. NVFP4每16个reduction-axis元素共享非负E4M3 scale，value为E2M1，另可带FP32 tensor scale。不是INT4，也不是group32/E8M0的MXFP4。每byte打包两个值，低nibble为先出现元素。
2. 本机A为row-major `[M,K/2]`，B由row-major `[N,K/2]` transpose成column-major `[K/2,N]`。block scales须128×4 tile swizzle及补零；直接flatten逻辑 `[M,K/16]` 会错配scale。smoke中的`swizzle()`用非均匀scale验证。
3. 本机API校验packed K为16倍数，即逻辑K为32倍数，N为16倍数；M=257已通过。更广shape、padding、stride、性能尚未验证。[cuBLAS官方布局文档](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)说明16元素/E4M3和tile padding；实际wheel支持以本机测试为准。
4. 两级API是`F.scaled_mm(..., scale_a=[block,global], scale_recipe_a=[BlockWise1x16,TensorWise], ..., swizzle_a=SWIZZLE_32_4_4)`，完整例子在脚本中。不能把`scale_result`误用成NVFP4 global input scale。
5. Native conversion明确指定rounding/finite saturation；通常采用nearest-even应使用明确RNE实现及边界测试。[CUDA FP4 conversion官方API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html)
6. 旧H3 `scripts/minimax_h3_svdquant_common.py:183`先量化再返回原dtype，模型通常BF16。sorted 15-level codebook的`argmin`在ties选择数值较小侧：-0.25→-0.5，0.75→0.5，1.75→1.5；与RNE的0、1、2不同。非tie且有限值时nearest codebook一致；不能把所有旧结果都判为无效。需先统计实际midpoint频率、端到端影响，再决定重跑。此次未改历史实现。

## 4. 共同global scale：先排除纯缩放假象

定义block理想scale `a_b=max(abs(x_b))/6`，global `g>0`，有效scale `s_b(g)=g*R8(a_b/g)`，重建`Q_g(x_b)=s_b(g)*R4(x_b/s_b(g))`。这里R8是E4M3，R4是明确RNE的E2M1。

**四种情形必须区分：**

- 所有x同时乘c，并相应令global乘c：理论上`Q_(cg)(cx)=c*Q_g(x)`。忽略浮点溢出/零scale/floor后，这是缩放协变；纯共同倍率不会自动造成outlier损失。
- 目标x不变，仅因其他请求使共同global变大：只改变目标block scale在E4M3格点的phase。在normal区间，若倍率为2的整数次幂且对应值仍可表示，E4M3指数可精确吸收，重建不变。
- 任意非2次幂倍率会改变E4M3 rounding，继而改变一些E2M1 codes。这可能让输出变动，但不等价于误差单调增大；应比较对原输入/任务指标，不能仅以“和原量化输出不同”作为失效证据。
- 只有跨请求幅值差足以把较小请求的`a_b/g`压入E4M3 subnormal/zero，或其他非缩放结构变化，才有直接动态范围损失机制。E4M3最小normal为2^-6、最小subnormal为2^-9。实际模型中该情况是否出现、是否频繁、是否损伤质量尚无证据。

CPU控制：`scripts/research/probe_nvfp4_global_scale_20261002.py`，结果`research_state/00_state/systems_global_scale_20261002.json`。仅8×256标准正态合成输入，明确RNE；无模型、无GPU原生多请求推理、无延迟测量。

| 仅global倍率 | 相对原QDQ的NMSE | 相对原输入的NMSE | 现象 |
|---|---:|---:|---|
| 1 | 0 | 0.00962718 | 基线 |
| 1.5 | 0.00687717 | 0.00962958 | 6.49% codes改变，但重建误差几乎未变；scale均normal |
| 2、16、1024 | 0 | 0.00962718 | 指数精确吸收，无变化 |
| 1e6 | 1 | 1 | 全部block scale下溢为0；人为极端控制 |

所有x与global一起乘上述倍率时，除浮点机器精度外均严格协变。**不能以1e6极端控制推断真实视频模型存在batch catastrophe，也不能用1.5时输出变化宣称严重outlier。**

## 5. SM120原生GEMM能否隔离per-request scale

可以，但要区分表示能力与现成单调用接口。

- **分开调用：** 每请求独立量化/打包，分别使用已验证的block+TensorWise recipe，可直接隔离global。代价是更多GEMM launches、小M效率、分段padding；未测服务延迟。
- **单GEMM吸收入block scales：** 请求i有`g_i, s_ib`，选共同G后将block scale改为`R8((g_i/G)*s_ib)`。若乘积可精确E4M3表示，则原E2M1 codes不变且数学值保持；2次幂且不越界是充分常见条件。任意倍率通常再次舍入，因此不能宣称无损，subnormal/zero仍可能出现。这也说明global大本身未必导致精度恶化。
- **单GEMM加输出row scale：** 把各请求按自己的global预归一化并打包，主GEMM使用共同global，按请求在输出端乘`g_i/G`。实数线性代数上可隔离，但在BF16输出后再乘会多一次舍入；需FP32输出或融合epilogue才能更贴近分请求契约，并且bias/residual/低秩branch的顺序须正确。当前未实现或验证这条优化路径。
- 本机torch `_meta_registrations.py:6862–6878`列出的NVFP4组合为BlockWise1x16单级，或BlockWise1x16+TensorWise两级；没有BlockWise1x16+RowWise组合。**因此不能把每请求FP32 global向量直接传入当前API当作已经支持。** 这是对现有torch接口的限定，不是SM120硬件不可能支持自定义epilogue。

值得做的下一步仅是：实际请求在合batch前后的scale normal/subnormal/zero比例、真实midpoint与codes变化、任务误差变化；并加2次幂控制、固定global控制、per-request oracle。只有观察到常见真实分布中的不可吸收结构且影响任务质量，才考虑形成研究claim。不要先在这个机械现象上投入大benchmark。

## 6. 真实加速的限制和最低验证标准

- 计费activation absmax/global、FP8 scale生成、FP4 packing/swizzle、主GEMM、低秩branch、bias/residual/norm/attention，再测transformer step和完整生成。Python QDQ的15码本临时张量及大量kernel不是速度baseline。
- 权重离线packed，激活高效生成；NVFP4约4.5bit/value（4bit值+每16值1byte scale，另加global/padding）。BF16 QDQ checkpoint没有上述内存收益。
- rank32/64的两次BF16小GEMM及访存须计入质量—延迟Pareto；融合要针对SM120，不照搬SM100。大的video token M、H3 fused QKV/SwiGLU N/K、padding和连续性决定效率，256方阵smoke只证明可调用。
- 多卡PCIe/NUMA通信、不规则稀疏的索引/packing和occupancy损失必须测；官方已有structured sparse NVFP4，简单组合不是贡献。
- 最低可信链条：独立software reference→native per-layer parity→完整block→固定prompt/seed paired denoising和最终视频→真实延迟/显存。随结果保存rounding、scale粒度、global、padding、排除层、attention dtype。不能把理论吞吐或小矩阵kernel速度等同端到端加速。

本轮证明范围：两shape的packed native GEMM、一次两级global-scale numerical smoke、CPU缩放代数控制。未证明任何模型质量改进、服务batch失效、端到端加速、FP4 attention实际可运行或投稿级novelty。
