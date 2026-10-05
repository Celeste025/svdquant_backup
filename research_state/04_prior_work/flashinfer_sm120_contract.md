# E006 FlashInfer SM120 NVFP4 attention 静态契约核查

2026-10-02。只读已安装 FlashInfer 0.7.0.post1 官方源码；未安装、编译或运行 GPU，未写 E006 脚本。结论：**packed `fwd` 能表达固定 Q/K/V scale 后重编码的干预；现成 `quantize_qkv` 不能接收外部 scale，需要小型 fixed-scale packer。P 的 online scale 无法从此 API 冻结。**

源码根目录：`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer`。本地版本优先于 GitHub main；末尾记录 hashes。

## 1. 实际 Q/K/V 格式与处理

[Python 入口](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py) 的本地同名文件关键位置：`_preprocess_qkv` 第 123–165 行；`quantize_qkv` 第 208–255 行；`fwd` 第 390–540 行。

| 项目 | 此版本实际契约 |
|---|---|
| 输入 | contiguous CUDA FP16/BF16，Q `[B,Hq,M,D]`，K/V `[B,Hkv,N,D]`，D=64/128；Hq 必须可被 Hkv 整除。 |
| K smoothing | **先对未 padding 的 N tokens 求均值并减去**；随后 pad 到 Npad=ceil(N/128)×128。BF16 输入时减法结果仍为 BF16，不能把该 rounding 当成 FP4 scale 效应。 |
| Q smoothing | **先 padding，再**按 128-token block 求均值并减去（默认）；`per_block_mean=False` 则在 padded M 上求全序列均值，不能在 arms 间切换。 |
| QK correction | FP32 计算 `q_mean @ k_centered.T`，形状 `[B,Hq,Mpad/128,Npad]`（全局 Q mean 时第三维为 1）；它使用未 FP4 量化的 centered K。传入 `fwd` 后加到 QK score 再 softmax。 |
| Q/K scale | 每 token 连续 16 个 head channels 一个 E4M3 scale；**没有额外全 tensor/global scale 参数或 amax reduction**。 |
| V scale | V 先转置为 `[B,Hkv,D,Npad]`，因此每 channel 连续 16 个 tokens 一个 E4M3 scale；**没有 V global scale，也没有 V mean smoothing**。不要套用其它 NVFP4 API 的全局 scale 描述。 |
| packed bytes | Q `[B,Hq,Mpad,D/2]`；K `[B,Hkv,Npad,D/2]`；Vt `[B,Hkv,D,Npad/2]`，均 uint8。两个 E2M1 值一字节，低 nibble 对应较前元素。 |
| scale tensor | API shape 看似 `[B,H,R,C/16]`，但内部按特殊 scale swizzle 存储，不能直接按该 shape 广播解码。 |
| masking | 只有 causal bool 和一个共同的 `unpadded_k_len`；无 custom mask、cu_seqlens 或每 sequence 长度。默认不传逻辑 N 会把 K padding 作为有效 key。输出仍 Mpad，要裁回原始 M。 |

CUDA quantizer：`data/csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_quantize.cu` 第 134–243 行（Q/K）及 245–382 行（Vt）：

- s = E4M3_RN(max(abs(x_16))/6)，再把**已舍入的 s**转 FP32；r = 0 if s=0 else 1/s；code=E2M1_RN_SATFINITE(FP32(x)×r)。
- native conversion 位于 `data/include/flashinfer/math.cuh:124`，用 `cvt.rn.satfinite.e2m1x2.f32`。软件 pack 必须覆盖 ties-to-even、正负零与饱和，不可沿用旧 midpoint-tie 实现。
- scale=0 的微块编码为 0；若 teacher scale=0 而受扰动数据不为0，固定尺度 arm 仍必须保留该规则，并单独报告这类退化微块。

### 固定 packer 必须复现的排列

把 Q 或 K 的物理矩阵看作 R×C，C/16=cblocks。逻辑 scale 行 r、微块列 c 的字节偏移为：

```text
offset = (r//64)*64*cblocks
       + (c//4)*256 + (r%16)*16 + ((r%64)//16)*4 + c%4
```

Vt 使用同一公式，但 R=D、C=Npad。实现先从 native scale buffer **反 swizzle**到逻辑位置做归一化，再按相同公式存回；固定 arm 可直接复用 teacher 原始 swizzled buffer，不应重新舍入。

K 还需额外的 token permutation。物理行 t 在每 32-token 组内读取原 token：

```text
u = t % 32
pi(t) = (t//32)*32 + (u//8)*2 + ((u%8)//2)*8 + u%2
```

即 K packed 的 t 行以及相应 scale 都来自 `k_centered[pi(t)]`。Q 不做此 permutation；Vt 只转置，不做 K 的行 permutation。`qk_correction` 保持 Python 生成的自然 token 顺序，不能自行把其 N 轴再按 K permutation 重排。

## 2. P 两级量化：固定 QKV scale 无法冻结它

本地 JIT 宏固定 `QBLKSIZE=128,KBLKSIZE=128,CTA256,DQINRMEM,PINGPONG_MATH_ORDER,PINGPONG_EARLY_RELEASE_K`，没有启用额外 `DIRECT_P_QUANT_SOFTMAX` 等实验宏。

`softmax.cuh:54–58` 固定常数 `1/(448×6)`；当前 n64 slot 路径在 364–410 行。用自然指数等价表示，内部保存的未归一化概率被乘上 **2688**，其 FP32 行和也采用同一倍率，所以最终相除抵消该共同倍率。P 微块 scale 由当前 QK score 的局部 max 与 running row max 动态产生，约为 `448×exp(local_max−running_max)`；归一化 code 输入约为 `6×exp(score−local_max)`。P scale 在 mainloop 781–787 行转 E4M3，P 在 797–805 行转 E2M1。

一个容易混淆的差别：**P 的 FP4 code 输入用浮点 AbsMaxP 归一化，E4M3 scale 另行转换**；它不像 Q/K/V quantizer 那样先将 scale 舍入后再取逆。内核还有在线 row-max 更新、既有 O 重缩放、FP32 row-sum 和近似 exp2；这些都随着实际输入变化。不能把 `O11−O1F` 解释为“P scale 不变时的纯误差”，也不能仅通过替换 Q/K/V scale tensor 构造 frozen-P arm。

`return_lse` 可以固定为 false 以减小干预面；若返回 LSE，源码已减去上述 2688 倍率，使其是普通 scaled-score logsumexp，不能再人为减一次。

## 3. 可直接实施的最小五臂协议

先选 **Wan 的一个 dense self-attention sequence，D=128**，避免把 H3 packed 多 sequence 错当一条全连接序列。若选 H3，必须由真实 attention caller 切出每个合法 sequence 并逐段调用，保留原 attention 分区；不允许加长跨 sequence 可见范围。

同一层输入通过两种 projection 产生 T0 与 T1：

- T0：BF16 projection 后，经过模型原有 QK norm/RoPE，抵达 attention 入口的 Q/K/V。
- T1：native W4A4 projection 后，经过完全相同的 QK norm/RoPE，抵达同一 attention 入口。后续 output projection 保持 BF16，以免加入第三个因素。

用固定 `per_block_mean=True` 分别求 `(proc_i, correction_i)`；`proc_i` 可直接复用此版本 private `_preprocess_qkv` 的输出，并固定其源码 hash。Packer 的插入点必须在 preprocessing **之后**。

| Arm | 计算 |
|---|---|
| O00 | torch BF16 SDPA(T0)，固定非因果、无 dropout、显式 scale=1/sqrt(D)。 |
| O10 | 同一 torch SDPA(T1)。 |
| O01 | 官方 quantize_qkv(T0) → 官方 packed fwd。 |
| O11 | 官方 quantize_qkv(T1) → 官方 packed fwd。 |
| O1F | **同一个 proc_1**，使用 T0 的 Q/K/V scales 重新编码；将新 codes + teacher scales + **correction_1**传给原样 packed fwd。 |

O11 与 O1F 之间只改变输入 Q/K/V scale 选择；两者使用同一组 smoothing 后的实际受扰动数值和同一 correction_1。**禁止把 native(T1) 的 codes 配上 teacher scales；禁止复用 correction_0。** 两种做法都改变了重建值的含义，不能回答问题。

FP32 计算 `I=O11−O10−O01+O00`，`I_F=O1F−O10−O01+O00`，故 `I−I_F=O11−O1F`。最后一项是固定 QKV 输入尺度干预的总因果影响，包含它对 kernel 内部 P 的传导。由于不同 T_i 的 means 自适应变化，I 本身包括 smoothing/FP32 correction 的交互，不能预先都归因于 scale；第五臂只定位其中一个可控部分。

最小必需验证：

1. 对 i=0、1，fixed-packer(proc_i, native_scales_i) 的三个 packed tensors 必须与官方量化逐 byte 相同，三份 scale buffer 也一致；运行 fwd 应重现各自原输出。不是近似 cosine 相同即可。
2. 所有 arms 显式传相同 `unpadded_k_len=N`、`causal=False`、`sm_scale=D**-0.5`、`per_block_mean=True`、BF16 out dtype、相同 return_lse 选项；最终只比较前 M 行。
3. Q/K/V 各记录 logical scale mismatch rate、scale=0 微块率、`abs(proc_1)>6*s0` 的元素率/能量、saturating-code fraction；K 指标先 inverse permutation，V 按其 token-group granularity统计。code==6 不等同 clipping，二者分开。
4. 正常 O11 与 O1F 的 means/correction 做 hash 或逐元素一致校验；smoothing mean 范数与 correction 变化单独记录，不能与 scale switch 合并。

如果第五臂明显有效，再做第二级（非最小首轮）：两种 projection 都使用 teacher Q/K means 作为固定坐标系，每种输入仍重算 `q_mean_teacher @ (K_i−k_mean_teacher).T`，分别比较动态/固定 scales。不能冻结整个 correction 来假装固定均值。该第二级还须验证 BF16 smoothing 的舍入影响，暂不建议在首轮加上。

## 4. 允许和不允许的结论

允许：输入 QKV scale 变化是否贡献了组合交互；在真实 native FP4 attention 上的固定-scale 因果效应。准备 codes 的 packer 可以是慢速 reference 实现，因为真正被测 fwd 不变，但 **不能由此报告完整量化+attention 的吞吐收益**。

不允许：声称冻结了所有量化网格；把 P 误差分离出来；把全局 V amax 当本实现机制；由自定义 packer 近似不匹配产生的差异讨论研究结果。若固定 QKV scale 不起作用，只否定这一输入尺度机制；内部 P/code-cell 或普通 softmax 非线性仍可能造成 I，但需要另一个问题与实验，不能含混延长 E006。

## 核实源码 SHA256

- `nvfp4_attention_sm120.py`: `040738c6d6af5584ca9b375bb59468d7011506862ff0a162f1f51a72149abefc`
- `jit/nvfp4_attention_sm120.py`: `bcae2efa6c36cc8aa4f0c2c8ccd803af3b0a55f8300e227157680c4e5296d115`
- `data/csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_quantize.cu`: `86abc2a339444943c16baaab8943d212d30af4c1428371f99e844b6795a68aca`
- `data/csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_binding.cu`: `9a847598ef4a36cc078a73722c2826d16b8bc489acfb031cac5f9957b4c5619c`
- `data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/consumer/softmax.cuh`: `57fa09d6e5884abe2b9ce94f492d1a222ce6e6bddbc3ab364818a6d913d12cb6`
- `data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/mainloop.cuh`: `0c36b7ecf8ed705ca9b852560a4dba17840b0bf494f50fd8511d3f8f77268f5e`
- `data/include/flashinfer/math.cuh`: `905bd156f57a1667e30be4a890906a628dbe414a650ce51c900c0bb1a65a135f`
