# E022：本地 rCM 参考实现只读核查

2026-10-03。范围仅为已读本地源码、checkpoint 元数据和旧转换验证记录；没有运行 GPU、修改生成源或增加实验。

**结论：未发现 E022 在 rCM 采样、文本 mask/zero-padding 或已读转换关键映射上的明确偏差；本次未定位 BF16 视频青绿条纹原因。这不证明完整官方 rollout 与当前 Diffusers 路径等价。**

本地官方仓库 `/home/wjq/workspace/rcm` 的 remote 为 `https://github.com/NVlabs/rcm.git`，HEAD 为 `ed3cb14dd936f92cdc9f9381af7369991509b41f`（2026-06-26，`update paper link`），本次读取时工作树干净。公开非因果 T2V 采样入口最后修改 commit 为 `26af800530a06621c054d7cd5b5650131e4f8598`（2025-12-12）。不将同仓库后来的 causal/VBench 入口默认值混入该 checkpoint 的公开采样合同。

## 已核对的合同

| 项目 | 本地可复核证据与结论 |
| --- | --- |
| 模型与权重来源 | 官方 [入口配置和加载](/home/wjq/workspace/rcm/rcm/inference/wan2pt1_t2v_rcm_infer.py:35) 为 Wan T2V 1.3B（30 层、12 heads、hidden 1536），96–105 行读取指定 checkpoint 并去掉 `net.` 前缀。现有原始文件为 `/data1/models/svdquant-wjq/models/rcm-Wan/rCM_Wan2.1_T2V_1.3B_480p.pt`；其本地 HF 下载 metadata 记录 revision `142d77a331f1516fea772607c1bf25a2785a2505`。E022 [显式加载](/home/wjq/workspace/svdquant-exp/scripts/research/generate_wan_qad_expanded_comparison.py:211) `/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer`，文本/tokenizer/VAE 从 BASE Wan Diffusers 资产加载，绕开已知失效的 rCM 根目录软链。本轮未重新逐张量验证源 checkpoint 与导出文件。 |
| 步数、schedule、velocity | 官方 [默认参数](/home/wjq/workspace/rcm/rcm/inference/wan2pt1_t2v_rcm_infer.py:68) 为 4 步、sigma_max=80。官方 [143–171 行](/home/wjq/workspace/rcm/rcm/inference/wan2pt1_t2v_rcm_infer.py:143) 使用角度 `[atan(80),1.5,1.4,1,0]`，以 `sin(a)/(cos(a)+sin(a))` 转 RF，终点为 0，无独立 sigma_min；更新为 `x_next=(1-t_next)*(x-t_cur*v)+t_next*noise`。与 E022 [166–179 行](/home/wjq/workspace/svdquant-exp/scripts/research/generate_wan_qad_expanded_comparison.py:166)、[243–252 行](/home/wjq/workspace/svdquant-exp/scripts/research/generate_wan_qad_expanded_comparison.py:243) 一致。 |
| dtype 与 guidance | 官方 [135–171 行](/home/wjq/workspace/rcm/rcm/inference/wan2pt1_t2v_rcm_infer.py:135) 使用 FP32 初始/更新噪声、FP64 采样状态、BF16 DiT 输入及 timestep，预测转 FP64 后更新；timestep 同样先 `t_cur.float()` 再乘 FP64 ones 与 1000、转 BF16。E022 保留该顺序。官方入口只有条件预测、没有 CFG 混合；E022 guidance=0 关闭 CFG，语义一致。此项不表示两个 DiT 内部实现逐算子或逐位相同。 |
| 时空尺寸 | 官方 [79–82 行](/home/wjq/workspace/rcm/rcm/inference/wan2pt1_t2v_rcm_infer.py:79) 默认就是 77 帧、480p、16:9；[尺寸表](/home/wjq/workspace/rcm/rcm/datasets/utils.py:29) 对应 832×480。[VAE 帧数映射](/home/wjq/workspace/rcm/rcm/tokenizers/wan2pt1.py:708) 为 `1+(frames-1)//4`、空间压缩 8，因此 latent 为 `[1,16,20,60,104]`，与 E022 一致。81 帧是其他入口/示例的默认值，不构成本入口的偏差。 |
| 文本清理、mask 与零 padding | 官方 [tokenizer](/home/wjq/workspace/rcm/rcm/utils/umt5.py:69) 使用 whitespace/basic clean、max_length=512、截断及 special tokens；[503–520 行](/home/wjq/workspace/rcm/rcm/utils/umt5.py:503) 将 attention_mask 传入 UMT5，按 mask 得到有效长度，截取有效 embedding 后零补到 512。安装的 [Diffusers 文本路径](/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/lib/python3.12/site-packages/diffusers/pipelines/wan/pipeline_wan.py:153) 执行相同合同；E022 [172 行](/home/wjq/workspace/svdquant-exp/scripts/research/generate_wan_qad_expanded_comparison.py:172) 显式指定 max_sequence_length=512、关闭 CFG。 |
| DiT 的文本 padding attention | 官方 [context_lens=None](/home/wjq/workspace/rcm/rcm/networks/wan2pt1.py:791)，[cross-attention](/home/wjq/workspace/rcm/rcm/networks/wan2pt1.py:292) 直接对完整 context 调 attention，不屏蔽零 padding。Diffusers [cross-attention 调用](/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/lib/python3.12/site-packages/diffusers/models/transformers/transformer_wan.py:277) 同样未传 padding mask。因此本轮没有证据支持“E022 漏传 mask 导致短 prompt 失败”。 |

## 转换验证覆盖边界

[转换器](/home/wjq/workspace/svdquant-exp/scripts/convert_rcm_wan_to_diffusers.py:86) 去掉 `net.`、排除 `accum_` 训练计数，映射模块名称，检查目标 key/shape/覆盖；97–98 行将 patch Linear 权重重排为 `[1536,16,1,2,2]` Conv3d 权重。没有从这些已读映射中找到明确漏项或错配，但代码中的检查不等于本轮重新验证所有已有导出权重。

[旧数值验证脚本](/home/wjq/workspace/svdquant-exp/scripts/verify_rcm_wan_transformer.py:41) 仅使用随机 BF16 latent `[1,16,5,32,48]`、随机全长非零 text `[1,512,4096]`、timestep=500，比一次原版/转换版 forward。[已有报告](/home/wjq/workspace/svdquant-exp/results/reports/rcm-wan-transformer-equivalence.json) 为 max_abs_error `0.0707130`、RMSE `0.0138622`、cosine `0.9997483`。

该检验**没有覆盖真实短/长 prompt 的 UMT5 编码、零 padding 实例、目标 20×60×104 latent、四个实际 timestep、四步自由 rollout 或端到端视频**。不能用它证明完整官方生成等价，也不能据此断言转换导致当前条纹。

本轮前置的固定两 latent BF16/FP32 VAE 对照已由 root 执行：BF16 复现原视频 SHA，坏例 FP32 仍保留同样条纹；该结果不支持 VAE dtype 是此次现象的解释，未进行批量 FP32 重解码。短 prompt 多个失败、较长阅读 prompt 正常仍只是样例描述。若以后需要原始 rCM 真实文本端到端对照，应另定范围；本轮不实现、不启动。
