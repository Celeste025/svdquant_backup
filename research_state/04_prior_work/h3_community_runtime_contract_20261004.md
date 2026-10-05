# H3 calibrated-8×20：部署 A4 合同与公开 loader 边界

2026-10-04。有界只读核查：正文读取 6 份直接 primary 源文件；另做仓库 HEAD 定位和未成功的匿名 kernel 可达性请求。0 权重下载、0 安装、0 远程代码执行、0 GPU。未修改历史结果或共享 current_state。源文件与访问记录见[本次 manifest](/home/wjq/workspace/svdquant-exp/results/research/h3_community_runtime_contract_20261004/manifest.json)。

**结论：公开 Diffusers 的部署调用合同是 NVFP4 A4，不是由 sidecar 的 `int4` 字段决定的 INT4 A4。公开 loader 源码确实存在；其必需 Hub kernel 在本轮匿名访问中返回 401，未取得 kernel 源码/二进制或 SM120 分发清单。因此当前应保留为外部产品对照候选，不能记为本机可运行、已复现的完整强基线，也不能用它直接替代同权重、同目标范围的自建基线。** 401 不能区分仓库缺失、私有/受限或服务状态，不证明永久不可用；没有尝试绕过访问限制。

固定 checkpoint revision 为 `1d59adb350cc915b2bf96d5a41d7abd238842772`，producer revision 为 `09658748fdb63b6150a7621566a46fcb2d3b830f`。本轮用 `git ls-remote` 固定公开 Diffusers HEAD 为 `8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd`。这一当前 loader 不是 checkpoint 历史生产环境证明。

| 本轮直接读取的源 | 精确位置与事实 | 能支持的结论 |
| --- | --- | --- |
| [calibrated-8×20 config](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/calibrated-8x20/config.json) | L23–30：`nunchaku_lite`、pre-quantized、BF16 compute、SVDQ `nvfp4`/group16/rank32；L32–343 是312个SVDQ target；L346起是50个INT4 AWQ W4A16 target。 | 应沿 NVFP4 SVDQ loader 追踪；AdaLN 的 INT4 **weight** 不表示其 activation 是 A4。 |
| [Diffusers runtime utils](https://github.com/huggingface/diffusers/blob/8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd/src/diffusers/quantizers/nunchaku/utils.py#L170-L213) | L181–195：NVFP4 分支建立 `[K/16,padded_M]` FP8 E4M3 activation scales，调用 `quantize_w4a4_act_fuse_lora(..., nvfp4=True)`；L198–211 也向 GEMM 传 NVFP4 标志。INT4 分支则是 group64、模型 dtype scales。L290–319 从 config 原样传 precision/group/rank。 | 公开 Python 部署路径明确选择 NVFP4 activation 与 GEMM。这里核的是调用合同，未读取不可达 kernel 内部、未实测其实际指令。 |
| [Diffusers quantizer](https://github.com/huggingface/diffusers/blob/8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd/src/diffusers/quantizers/nunchaku/nunchaku_quantizer.py#L24-L74) | L24–52 要求 `kernels`、CUDA；NVFP4 接受 Blackwell 或以后 capability，排除 Hopper；L61–74 在加载权重前按 config 替换模块。 | 存在公开的通用预量化加载实现，SM120 不被该架构检查拒绝；但这不证明相应 ABI/架构 kernel 二进制可取得。 |
| [模型卡](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/README.md#L51-L110) | L62–69 给 `MiniMaxH3Transformer3DModel.from_pretrained(..., subfolder="calibrated-8x20")`；L105–107 指定 `kernels`、remote-kernel trust、SM120/PyTorch≥2.7/CUDA≥12.8。 | 有具体 H3 接入入口；硬件/版本及一次成功加载仍只是发布者要求或声明，非本次复现结果。 |
| [producer README](https://github.com/rootonchair/diffuse-compressor/blob/09658748fdb63b6150a7621566a46fcb2d3b830f/README.md#L53-L55) | L53–55 的旧 `nunchaku_lite` 安装路径要求 release/private package channel，extra 不安装 public PyPI runtime。 | 此处不足以排除当前 Diffusers 的另一条公开 loader；两条分发路径必须分开。 |
| [producer 映射说明](https://github.com/rootonchair/diffuse-compressor/blob/09658748fdb63b6150a7621566a46fcb2d3b830f/docs/deepcompressor_mapping.md#L30-L42) | L30–35 区分 INT4 activation metadata 与尚未单独实现的 FP4 activation packer；L42 说明若干 smoothing policy 未完全同等实现。 | 生产/校准 metadata 和当前部署 loader 不是同一证据层。不能据 metadata 宣布部署 INT4，也不能据当前 loader 宣布历史校准已忠实使用同一 NVFP4 A4。 |

实际 kernel 依赖在上述 runtime utils L20–36：唯一指定 `rootonchair/nunchaku-lite-kernels`，`get_kernel(..., version=2).ops`。本次匿名请求其 HF model API 的 main、试探 v2 revision 以及 main README 均返回 HTTP 401；web tree 也不可达。`version=2` 如何映射具体 git ref 未核实，不能把试探的 `v2` 当作已知正确 ref；main API/README 同样未取得内容。本轮没有下载 kernel 二进制，也没有调用 `get_kernel`。公开 loader 文件本身已保存为[utils.py](/home/wjq/workspace/svdquant-exp/results/research/h3_community_runtime_contract_20261004/utils.py)（SHA `fd78f09f28181e2a07250b0d09c3806064b4cf16a8e44cfa1efae56971d7691b`）与[quantizer](/home/wjq/workspace/svdquant-exp/results/research/h3_community_runtime_contract_20261004/nunchaku_quantizer.py)（SHA `e0cc39f2717dfb9a928eeda54a24a886c1b70d68ca4177af4dd578ff8b1ec115`）。

root 的单独本地只读核查补充：已检查 native/recovered 两个 site-packages，未找到 diffusers/nunchaku_lite/kernels；legacy PTQ 环境是 diffusers 0.33.1；标准 HF token 文件和两个常见 token 环境变量均不存在，标准 cache 及 research/cache 所查位置未发现 nunchaku 目录。这只说明已查环境没有可立即复用的依赖/缓存，不声称全盘无包；本轮没有读取任何 token 内容或尝试私有访问。

不能用 target 数量或模型卡参数量偷换底模身份。50 个 fused QKV 拆成 Q/K/V 会把本地200个主线性层的计数变为300；两个 refiner 各6个 target 正好再加12。该计数本身不是底模不同的证明。真正明确的量化范围差异是 refiner 和额外50个 AdaLN AWQ 被量化。并且 loader 给每个拆分 Q/K/V 建 rank32 模块；这些分支因子是否相同或共享尚未读权重，不能直接等同本地 fused-QKV 的共享 rank32 预算。模型卡所写参数量/文件体积也没有经过本轮 tensor 映射核实。[config](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/calibrated-8x20/config.json)、[模块构造](https://github.com/huggingface/diffusers/blob/8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd/src/diffusers/quantizers/nunchaku/utils.py#L111-L168)。

NVFP4 也不等于本地量化配方逐项相同：公开接口带 FP8 micro-scales 及 `wcscales/wtscale`，fused activation/LR 内核处理 smoothing；本次未核其 rounding、global-scale、低秩输入算术与本地 native 合同的逐项一致性。不能从格式名推出同配方，也不能因 producer 未声称完整官方同等实现就判它质量弱。

当前可以改变的研究决定是：撤回“没有公开 H3 loader”的笼统排除，并停止把 sidecar 的 INT4 字段当部署 INT4 的证据；保留这套 checkpoint 为需要比较的外部产品候选。现在尚不足以取消同配置强基线建设。若后续要采用该成品，最先需要解决的单一门槛是取得**公开可访问、可固定版本的实际 SM120 kernel 分发入口**，再做有限加载一致性检查；在这一步之前不必下载约19GB权重。即使之后运行成功，也应先标为完整产品对照，待同源权重、量化覆盖/低秩预算及参考 BF16 的映射明确后，才讨论能否承担受控方法比较的强基线角色。本次不产生质量优劣、方法创新或历史校准一致性结论。
