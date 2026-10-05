# H3 → SpargeAttn mean-sim router 可行性核查

2026-10-02。CPU-only；未启动 GPU、安装包、编译扩展或修改冻结源码。部署后路由机制的门槛实验建议，不是研究结论。

**最终状态：NOT_READY / 未执行，停止 E011 实现扩展。** 官方 utils 可以按文件导入，但这不足以证明真实 H3 全长度 guard 语义合法。尾块归一化的 0/0 会被 bool 比较吞掉，本地也没有已验证可用的 SM120 sparse consumer。父 agent 决定不新建 12-case 重捕获、不跑 GPU smoke、不写/修 kernel。以下保留可复用接口与未执行设计；不是机制阴性或 GPU 实验结果。

## 本地环境与官方代码

检查所有仍有可执行 Python 的六个环境：H3 recovered、convrot-wan、svdquant-ptq、mjvideo、nvfp4-native-20261002、nvfp4-fused-20261002。均未找到 SpargeAttn/SLA 导入模块，workspace/共享第三方目录也未找到对应 checkout。部分旧 venv 目录存在但没有可用 Python，不能计为可用环境。

recovered、native、fused 已有 torch 2.11.0+cu128、Triton 3.6.0、einops 0.8.2。recovered 的 SageAttention 1.0.6 不是 SpargeAttn。官方 SpargeAttn 包入口会导入 `_qattn`、`_fused` 扩展，本地未安装；pin 的 setup.py 显式架构集合仅含 8.0/8.6/8.7/8.9/9.0，自动检测路径是否可构建 SM120 本轮未验证。因此不能声称可直接调用完整 sparse consumer，也不据此断言所有版本均不支持 SM120。[官方入口](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py)

与 literature 统一 pin `thu-ml/SpargeAttn@ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`。下载只读核查副本到 `/tmp/E011_sparge_source`。`spas_sage_attn/utils.py` SHA256 为 `eecf9109d12bbf34e853e327c4fa8f1b60e66dca74ec1a6ca355236bbbd06325`。已用 recovered Python 的 `importlib.util.spec_from_file_location` 成功导入，`torch.cuda.is_initialized()` 仍为 false；这是导入证据，不是 GPU kernel 可用证据。

官方可直接调用 `get_block_map_meansim_fuse_quant(q,k,km,BLKQ=128,BLKK=64,simthreshd1=...,cdfthreshd=...,return_lut=False)`，返回 block map 与独立 INT8 张量/scales。只依赖 torch/einops/Triton。按 core 的 non-sm90 路径使用 `km=k.mean(-2,keepdim=True)`，保持 BF16 输入和真实序列顺序；不能用无 `km` 的非 fused helper 代替。dense guard 将低均值相似度的整行/列保留；路由来自当前浮点 Q/K，不能描述成 INT8 反量化路由。consumer 的 CDF 与推荐 top-k 入口默认阈值不同；E011 须预先确定配置，不看结果调参。[固定 utils](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py)

SpargeAttention2 没有在本地找到独立后端；官方仓库公开的 code-release issue 仍可见，不能将 `spas_sage2_*` 的 SageAttention2 后端误称 SpargeAttention2 实现。[官方 issue 列表](https://github.com/thu-ml/SpargeAttn/issues) SLA2 对应官方 `thu-ml/SLA` 仓库，存在 `sparse_linear_attention` 和 SageSLA 源码，但本地没有安装；其训练路由/线性分支不是此处 mean-sim dense guard 的直接替代。[官方 SLA 仓库](https://github.com/thu-ml/SLA)

## E006 实际保存范围与 E011 最短捕获

`scripts/research/probe_h3_attention_interface.py` 中 `captured`、`qkv_pairs` 仅保留在 CPU 内存，case 末尾删除；没有 `torch.save` QKV/hidden。`results/research/E006_h3_attention_interface.json` 仅保存各 T0/T1 的 shape、SHA、误差摘要。不能宣称已复用实际 QKV artifact。

现成原始输入为 `results/calib/minimax_h3_svdquant_standard_8p64s/p{1,20}/sample_p{pid}_s{00,19}.pt`。12 个固定 block-case 为两个 prompt、两个 step、blocks 0/24/48，均属旧 PTQ 校准样本，不是 heldout。

父 agent 已确定实际部署配方：使用 E009 `legacy_export/blocks.*.attn.qkv_proj.pt` 对应 payload 和 NativeH3Linear，以及独立 zero-SF compatibility fast adapter。E006 原投影为 RNE，E009 为 legacy；仅复用捕获脚手架，不能混用 E006 T1 SHA 作为新 native 参考。

建议新脚本 `scripts/research/probe_h3_sparse_router.py`：

1. 复用已验证 `bench_h3_native_nvfp4.make_h3_resident`，一个 BF16 常驻模型，四次 teacher 前向捕获三个 block 输入/kwargs/endpoint；其余模型维持 BF16。
2. 每 case BF16 block 重放必须与 teacher endpoint 精确一致；捕获原模型 norm/RoPE 后实际 Q/K/V。再仅临时替换 qkv linear 为对应导出 NativeH3Linear，第二次重放捕获 native-projection Q/K/V，之后恢复原模块。
3. 按真实 `cu_seqlens` 分段调用固定官方 router；保存 Q/K/V SHA、guard 位图、row/column 计数、active-block 计数及真实 token-pair 加权工作量。若保存全部双臂 QKV，约每 case 1.8 GiB、12 case 约 21.5 GiB；首轮不需要保存全部 V，可保存少量失败输入与小统计。
4. 不运行 sparse consumer、rollout、完整性能或新校准；E011 预注册决定是否扩展。

## 必须先过的边界与成本

H3 post-RoPE shape 是 `[T,56,128]`。p1 `cu_seqlens=[0,22384,22400]`，p20 为 `[0,22432,22464]`。main 长度不整除 128/64；独立 padding 段长 16/32，小于官方 core 的 sequence>=128 要求，直接保留 dense，不混入主统计。

固定官方 Triton 存在非整块边界风险：Q masked load 未显式 `other`，K padding 归零后 normalize 出现零范数。不能通过偷偷截掉真实尾 token 或将 padding 当有效 key 来绕过。先用相同 length residue 的短随机输入验证 repeat determinism、有效 pooled/scale finite 与完整 map；全零/低范数组单独记录。若官方合法域不能稳定表达真实序列，则停止当前路由实验并报告接口限制，不自行修 kernel 冒称官方结果。

成本估计：E009 已测同形状 BF16 完整 DiT 约 8.26 秒，因此四次 teacher 算子时间约 33 秒；12 个 case 每个两次局部 block replay，native 投影仅一次，远低于 E006 五臂诊断的总 312 秒。模型加载、hash、CPU captures、首次 Triton JIT 与实际 router sorting 仍需额外时间，建议单 GPU 10 分钟硬预算，工程预计半天以内。常驻 BF16 模型约 37.46 GiB，已有真实 teacher 峰值约 40.86 GiB，router 保留双臂输入和临时张量时预留至 50–55 GiB；这是预算估计，不是实测峰值。此轮未启动该实验。
