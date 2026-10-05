# E046 后：原版 Wan SVD 基线配方与资产审查

2026-10-03；限定读取已有 checkpoint 元数据、成功日志、三个 cache 的 CPU mmap 元数据及官方小配置。无 GPU、下载、重新校准或大权重哈希。**最值得先补的是匹配当前生成分布的原版 Wan PTQ；不是先把 rank32 升为64/128。旧基线的数值执行已验证，不等于其校准分布匹配81帧/shift8。**

来源：当前 [五文件身份记录](../../results/research/E044/checkpoint_identity.json)；[成功 config](</data1/models/svdquant-wjq/runs/wan_s16_real_nvfp4/diffusion/wan2.1/wan2.1-1.3b/w.4-x.4-y.16-w.4/w.sfp4_e2m1_all-x.sfp4_e2m1_all-y.bf16-w.sint4/w.v16.sfp8_e4m3_nan.tsnr.fp32-x.v16.sfp8_e4m3_nan.tsnr.fp32-y.tnsr.bf16-w.v64.bf16/smooth.proj-w.static.lowrank/skip.x.[[w]+tan+tn].w.[e+rs+rtp+s+tpi+tpo]-extra.[tan+tn]-low.r32.i100.e.skip.[rc+tan+tn]-smth.proj.GridSearch.bn2.[AbsMax].lr.skip.[rc+tan+tn]-qdiff.64-h480.w832-t50.g6.0-s5000/run-260826.234259/config-260826.234259.yaml>) 与 [成功 log](</data1/models/svdquant-wjq/runs/wan_s16_real_nvfp4/diffusion/wan2.1/wan2.1-1.3b/w.4-x.4-y.16-w.4/w.sfp4_e2m1_all-x.sfp4_e2m1_all-y.bf16-w.sint4/w.v16.sfp8_e4m3_nan.tsnr.fp32-x.v16.sfp8_e4m3_nan.tsnr.fp32-y.tnsr.bf16-w.v64.bf16/smooth.proj-w.static.lowrank/skip.x.[[w]+tan+tn].w.[e+rs+rtp+s+tpi+tpo]-extra.[tan+tn]-low.r32.i100.e.skip.[rc+tan+tn]-smth.proj.GridSearch.bn2.[AbsMax].lr.skip.[rc+tan+tn]-qdiff.64-h480.w832-t50.g6.0-s5000/run-260826.234259/run-260826.234259.log>)（两文件的完整绝对路径均已核实存在），明确保存到当前非 rCM checkpoint。该 config 的 model path、成功保存记录和既有原权重小片核对支持非 rCM 来源；没有当年原模型全量加密谱系，不能补称已有。

| 项目 | 当前旧 checkpoint / 校准事实 | 与官方配方或本轮生成的关系 |
|---|---|---|
| rank / 拟合 | rank32，OutputsError，最多100迭代、early-stop | 与 [官方 SVDQuant 通用默认](https://raw.githubusercontent.com/nunchaku-tech/deepcompressor/main/examples/diffusion/configs/svdquant/__default__.yaml) 一致；没有证据证明 Wan 必须 rank64/128 才算 canonical。 |
| 数据量 / smoothing | 16提示采集池1600缓存；PTQ取64个 cache records，g10 | 与 [官方 fast.yaml](https://raw.githubusercontent.com/nunchaku-tech/deepcompressor/main/examples/diffusion/configs/svdquant/fast.yaml) 的64记录/g10吻合，是明确快速配置；不等于64提示。默认g20也存在，但不能将fast本身写成实现失败。 |
| group / 数值 | W/A E2M1，group16，E4M3局部scale；A动态 | 与 [官方 nvfp4.yaml](https://raw.githubusercontent.com/nunchaku-tech/deepcompressor/main/examples/diffusion/configs/svdquant/nvfp4.yaml) 的核心位宽/局部分组一致。 |
| tensor scale | 本地 `real_nvfp4.yaml` 在 W、A 两边显式加入 FP32 tensor-global | **确定不同于所读上游 YAML**：其 W outer dtype为null，A仅列group16/E4M3。这里是本地显式两级原生合同，不可据此判错，也不能把上游 YAML 与本地 native pack 全部称逐位相同。上游null的运行时dtype未在本轮推演。 |
| 帧数 / CFG / 步数 | archived config：33帧、480×832、CFG6、UniPC50；`shift_activations=false` | CFG6/50步匹配当前；**33→81帧确定不匹配**。`shift_activations` 是量化激活选项，不是 scheduler flow shift。 |
| 实际 timestep | 旧缓存 step0/25/49 为999/749/57；当前 E043 对应999/889/146 | **中点及末点实际时间表确定不同**；与本地 collector 默认flow_shift3一致，但旧完整 flow_shift 配置没有直接保存，不能将推断写成归档参数。 |

**三个实际 cache 证据。** 均来自 [旧校准目录](/data1/models/svdquant-wjq/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches)：

- [0019-0-00000-0.pt](/data1/models/svdquant-wjq/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches/0019-0-00000-0.pt)：`step=0, input_kwargs.timestep=int64[999]`。
- [0019-0-00025-0.pt](/data1/models/svdquant-wjq/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches/0019-0-00025-0.pt)：`step=25, timestep=int64[749]`。
- [0019-0-00049-0.pt](/data1/models/svdquant-wjq/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches/0019-0-00049-0.pt)：`step=49, timestep=int64[57]`。

三者 `input_args[0]`/`outputs[0]` 均 BF16 `[1,16,9,60,104]`，embedding BF16 `[1,512,4096]`，`filename='0019-0',guidance=0`；这里 guidance 是缓存分支索引，不是 CFG scale=0。仅加载元数据与 timestep 小张量，CUDA未初始化。对照 [E043实际步骤](../../results/research/E043/worker_161.json)，不从目录名推定时间表。

**没有可直接替换的匹配 rank64/128 NVFP4 资产。** 限定检查 DATA1 `ckpts` 一级清单、相关 Wan runs、repo `results/checkpoints` 与已知旧 ablation：

- [完整 r64 manifest](/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16-g10-r64/manifest.json) 明确 transformer=`rcm-Wan2.1-T2V-1.3B-Diffusers/transformer`，4步/77帧/sigma80/g0；根 `model` 字段虽写原版 BASE，也不能忽略明确覆盖的 rCM transformer。不是原版50步替代品。
- [旧非rCM r64 metadata](../../outputs/wan_ablation_w4a4_r64/quant_metadata.json) 明确 `sint4_fake`、动态INT4、group64、shift3；虽81帧/rank64，**格式不同**。其 [校准 manifest](../../outputs/wan_svdquant_calib_large/manifest.json) 也为shift3；不能拿目录名当 NVFP4 资产证明。
- 限定资产根未见非rCM NVFP4 rank128或另一个更匹配rank64；这不是全磁盘不存在证明。H3 r64和smoke/aborted目录不属于候选。

**唯一建议：补一个同原版权重、81帧/480×832、UniPC50/CFG6/shift8 的匹配 SVDQuant 校准基线。** 保持已有 rank32/group16/native 两级量化与部署路径，校准提示与当前已观察八例分离，明确采集两支及覆盖时间位置，再比较完整生成。先不同时提高rank、换scale规则或训练。它直接回答目前缺陷是否主要来自旧校准失配；改善不是新机制，无改善也不自动证明 NVFP4/VAE 的普遍失败。本次所读官方资料给出了通用 SVDQuant 配方，但未确立独立的官方 Wan 高rank标准，不应称本地扩展已经完成 canonical Wan 复现。
