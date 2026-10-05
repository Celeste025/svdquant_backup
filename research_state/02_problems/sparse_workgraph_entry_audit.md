# N2 现有稀疏工作图入口：有限只读核查

2026-10-03；无安装、下载、GPU、源码修改或新门槛。**补查已确认：本机有明确支持 SM120、D128 BF16、真实尾长、外部可变 count 的 cuDNN BSA consumer；API 与 dispatcher 已 CPU 导入成功。** 无需安装或新写 attention kernel 即可准备有限路由适配；首轮 GPU JIT/正确性尚未验证。H3 的累计概率路由与模态布局组合仍是新 adapter，不是现成产品全路径；这项工程缺口不是科学阻塞或机制阴性。

**本机新增入口。** E024 隔离环境 `research/envs/fastwan-qad-20261003` 已安装 FastVideo 固定 `8444c0897a8b96848eb85b6e5750ef486f79fc92`、fastvideo-kernel **0.3.5**、Torch **2.12.0+cu130**、Triton **3.7.0**。kernel 的 `block_sparse_attn(q,k,v,mask,variable_block_sizes)` / `block_sparse_attn_from_indices(q,k,v,q2k_idx,q2k_num,variable_block_sizes)` 可消费外部工作图；非 SM90 默认选择既有 Triton-64，明确支持每 query tile 不同 KV 数量、最后 KV tile 的有效长度。源中无 SM120 排除，D128 使用普通 BF16 `tl.dot`，因此是无需新 kernel 的候选；**本轮未运行它，不能写成 SM120 已验证可调用**。CPU 隐藏 CUDA 后导入 kernel/H3/BSA 入口遇到 Triton `0 active drivers`，CUDA 未初始化；这是导入期 autotune 驱动依赖，不证明 GPU kernel 失败。[本地 dispatch:447](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/fastvideo_kernel/block_sparse_attn.py:447)、[实际 consumer:703](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py:703)。SM100a/103a 专用路径不能当作 SM120 支持；可选 FA4 未在此环境安装。

| 现有 router | 真正的预算规则与可用边界 |
|---|---|
| FastVideo H3/Wan VSA；H3 `_pool_tiles` → `_build_block_mask` | **固定 top-k**，`ceil((1-sparsity)*video_blocks)`；H3 prefix query 全 dense，exempt 模式保留全部 prefix columns，再选各 video region 固定 k。输入可改变边的身份，但给定 metadata/模式每行物理块数固定，不能把它称 top-p 预算膨胀。图的 KV union/缓存局部性仍可变；本机没有相应通信或时延证据。[H3 mask:468](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/video_sparse_attn_h3.py:468)、[pool/router/consumer:753](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/video_sparse_attn_h3.py:753) |
| FastVideo BSA `_select_kv_blocks` | **真实累计概率**排序选块，默认 threshold=.9/min4；但完整实现还裁 Q、重构输出，consumer 是逐 head/query gather + varlen FA，FA 不可用时退 Python reference。当前没有 `flash_attn`；metadata 要时空各轴被 `(4,4,4)` 整除，rCM `(20,30,52)` 和 FastWan `(21,30,52)` 不满足。因此不能直接用它的完整耗时代表成熟 block-sparse 系统。[router:132](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/bsa_attn.py:132)、[消费/回退:177](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/bsa_attn.py:177)、[精确 tiling 限制:615](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/bsa_attn.py:615) |
| NABLA `nablaT_v2` + Torch FlexAttention | **真实累计概率 + 固定 STA union**，本次 CPU import 成功；但入口属于 Kandinsky 的 fractal 顺序，要求 `S%64==0`，没有 H3 混合模态/tail 有效键语义。把 H3 padding 当真实键、截断有效 token 或令 STA=0，都不是原模型既定路径。SM120 实际编译/消费仍未验证。[router:32](/data1/models/svdquant-wjq/third_party/FastVideo-8444c089/fastvideo/attention/backends/nabla.py:32) |

普通 SLA 的 `get_block_map` 也是 fixed-topk；SageSLA/Sparge 所需 `spas_sage_attn` 仍未安装。旧 Sparge pinned router 的尾块语义问题未因 dense FP4 kernel 可用而消失。未继续全盘查找或泛搜文献。

**最小数据与决策。** E018 三层 p36/s14 已有完整 post-RoPE QKV `[22539,56,128]`，可用于现成 consumer 的有界 smoke；但它们来自 SVD-native 自由轨迹，**不是同输入的 BF16/NVFP4 投影配对**。E006 未保存 QKV 本体；E009 qkv artifact 只有 `x_s` 和原输入 SHA，不能乘回 smooth 冒充逐字节原输入。要检验投影因果，最短是复用 E014 固定 teacher 输入和 E006 局部重放脚手架，只捕获一次 BF16 block0 输入，再在同输入上作 BF16/native qkv、原 norm/RoPE；各存一份 QKV，之后跨环境仅重放 tensors，避免 FastVideo 重载整个 H3。

**补查：优先的已安装 cuDNN SM120 入口。** 隔离环境 `fastwan-qad-20261003` 的 `nvidia-cudnn-frontend 1.30.0` / CuTe DSL `4.8.0` / CUDA bindings `13.4.3` 提供：

```python
from cudnn.block_sparse_attention.api import block_sparse_attention_forward
out = block_sparse_attention_forward(
    q, k, v, q2k_block_index=indices, q2k_block_nums=counts,
    sparse_block_size=128, softmax_scale=128**-0.5, layout="bhsd",
)
y, lse = out["o_tensor"], out["lse_tensor"]
```

输入 QKV 为同设备 FP16/BF16 `[B,H,N,128]`（亦可 `bshd`），最后维连续；非因果。外部索引 `int32[B,H,ceil(Nq/128),capacity]`，计数 `int32[B,H,ceil(Nq/128)]`，因此可直接表达真实 top-p 不同 query tile 的不同 KV 工作量；不是固定 top-k。caller 必须保证有效索引不越界、不重复，`1 <= count <= capacity`；本次不依赖未经验证的空行行为。也支持 block64；默认 SM120 是64，须显式128。`block_sizes` 可选 int32 `[ceil(Nk/block)]`、`[B,...]` 或 `[B,H,...]`，表示每个物理 KV block 内前缀的有效长度，不是任意 token mask。`kv_splits=1`，不启用 `pack_gqa=True` 或 `use_clc`。[公开参数与检查](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/cudnn/block_sparse_attention/api.py:174)

`_bsa_attn_fwd_sm120` 用真实 Sq/Sk 的 ceil-block metadata；SM120 D128 BF16/FP16 是显式分派。提供 `q2k_block_nums` 时走内置 **general SM120** kernel，不走仅适用于固定 count/无 block_sizes/Sk 整除128 的 FA4-style 分支，所以无需补 `flash_attn`。最后 KV block 自动以真实 Sk mask，LSE 写回检查真实 Sq，输出按真实 Sq 分配；没有要求截掉有效尾 token。H3 `[1,56,22539,128]` 对应177块，最后 KV 11个有效 token，可直接保留；任意模态 tile 重排仍需明确自己的 prefix/packing 合同。第一次运行会 `cute.compile`，本次未执行编译或 GPU，不能宣称性能/数值已通过。[dispatcher:831](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/cudnn/block_sparse_attention/_interface.py:831)、[动态 loop count:169](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120.py:169)、[KV tail:321](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120.py:321)、[Q LSE tail:405](/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/lib/python3.12/site-packages/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120.py:405)。

CPU 实查 `CUDA_VISIBLE_DEVICES=''` 下 `api` 和 `_interface` 均导入成功，前后 `torch.cuda.is_initialized()==False`。本地三份源码 SHA256：api `f482f0a47f7882f64dd60e13f6bc056fdf55a21cc0a43f914b867da45b66010c`；interface `a669b4a11dc7072cced10ce4fba6ff81ce04be4b8a6956446578475e6edc523b`；SM120 blk128 `770f278088ab6136d333e24420506e8d3309a7c40340a357b78344b676579c3a`。

**当前决策：存在值得做有界合法性测试的实际入口。** 优先 cuDNN 的外部 variable-count consumer，将既有累计概率 selector 与明确的 H3 pooling/prefix/tail 合同接起来；显式称新 adapter 即可，不应因尚无完整产品集成停止科学问题。后续若获准，先同一真实输入上比较 BF16/native 投影的工作图，再分开量度 router/consumer/完整路径；已有 E018 只够验证消费链，不能代替因果配对。无需新 kernel、安装、整视频或任意10%门槛；在看到实际工作量和系统成本前，不把量化×稀疏本身立为贡献。
