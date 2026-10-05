# rCM-Wan 主权重 QAD：本地实现就绪度

2026-10-03；只读代码、配置及 safetensors 头部。未训练、未初始化 GPU、未重新哈希大权重。

**结论：可在现有环境上实现，但仓库没有可直接运行的主权重 QAD 训练器。最小新增边界是“300 个主矩阵的可学习 FP32 master + W/A QDQ-STE”以及“无在线 LR 的 packed 导出/安装入口”；缓存读取、teacher/student 同输入循环、300 个真实 NVFP4 GEMM 基础设施均可复用。** 本轮不增加方法、loss 或质量门槛。

## 模型与可复用入口

- 原模型：`/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer/`。仅读 `diffusion_pytorch_model.safetensors` 的 **84,120 字节 JSON header**，825 tensors 共 **1,418,996,800 参数**；“1.3B”仅是模型名称。30 层，hidden=1536、12×128 heads、FFN=8960、patch=[1,2,2]。
- 300 个 block Linear 主矩阵共 **1,391,984,640 参数**：每层 self/cross attention 各 q/k/v/out，加 FFN 两矩阵。新主权重训练应明确这一范围；bias、norm、condition embedding、patch/output projection可先保留原 BF16。
- 原始输入缓存：`/data1/models/svdquant-wjq/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches/` 有64文件，`/data1/models/svdquant-wjq/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/blockwise_extra4_s16/caches/`有16文件。包含完整输入、512-token text embedding、4步 timestep；E007验证的 latent=[1,16,20,60,104]，实际31,200 patch tokens，可免重跑TE/VAE。它们已用于PTQ/旧实验，不能当新泛化数据。
- 训练循环可读 [exp_rcm_fullmodel_lowrank_denoiser.py](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_fullmodel_lowrank_denoiser.py:57)：teacher缓存与同输入 denoiser loss、AdamW、block checkpointing已有。**旧默认路径继承 `/data/models/...`，当前不存在**；需新入口显式指定 `/data1/...` 并直接加载 rCM transformer，避免误载基础 Wan 或先把完整TE/VAE搬上GPU。
- 已验证推理环境：`/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python`，E007记录 torch2.11.0+cu128、Diffusers0.33.1；复用同一 BF16 torch SDPA。E007/E008可提供部署和4步采样参考，不修改其冻结源码。

## 现有训练器实际训练了什么

[fullmodel trainer:111](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_fullmodel_lowrank_denoiser.py:111)先冻结全模型，仅解冻510个去重的 LowRank 参数张量，导出600个LR状态张量；主权重和smooth固定。[joint trainer:195](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_joint_smooth_lowrank.py:195)中的 `raw_weight` 是 detached 闭包，optimizer仅持有LR与210组smooth delta；其 `base + Q(candidate) − Q(reference)` 锚定表达式也不是一个可直接打包的主矩阵QDQ。现有 `trained_lowrank_branches.pt` / `final_joint_state.pt` 和 joint 推理脚本都保留在线LR，不是所需产物。

## 最小新增边界与合同风险

1. **新薄 Linear 训练模块。** 从原 rCM W 建 FP32 master，前向每次重做已冻结配方的 W/A QDQ、BF16 `F.linear`，反向使用显式STE；不是解冻已量化残差W后直接更新。只读复用缓存/teacher循环，启用现成 block gradient checkpointing。教师计算用 no_grad；student不能包 inference_mode。已有 [make_autograd_safe](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_signed_alignment_audit.py:63)能处理PTQ创建的 inference tensors，但从原 BF16模型构建干净模块更简单。STE须验证前向真实QDQ值和主W梯度/更新，不能只检查loss标量；`x+(Q(x)-x).detach()`存在有限精度抵消边界。
2. **冻结一个量化合同。** NVFP4采用group16、FP32 tensor scale、E4M3 block scale、E2M1 codes；W与A都要量化。E007 fastpack匹配的是历史Wan配方（含codebook中点、zero-scale处理），不能无验证切换为RNE/其他NVFP4配方。训练W若从FP32 master量化，导出不可先把master转BF16再量化。旧recipe还开启extra INT4 norm/add_norm，不能无意继承到“仅300主矩阵”的新基线。
3. **新无LR导出/安装入口。** 直接保存训练结束的codes、SF/global、bias及非目标状态；新安装器替换300矩阵并明确没有 LR hook/branch。可复用 [PackedNVFP4 / NativeWanLinear](/home/wjq/workspace/svdquant-exp/scripts/research/wan_native_nvfp4.py:167) 的存储、decode与GEMM，以及 [fastpack](/home/wjq/workspace/svdquant-exp/scripts/research/wan_nvfp4_fastpack.py:150)。**旧converter硬要求每层一个LR hook且校验旧model.pt，不能原样用于无LR。** `_recover_saved_weight`仅恢复已量化W的码，不是任意新W量化器；DeepCompressor同时请求qdata/dequant有已知alias，应分开获取。低秩合并后重新量化不是原SVD结果的精确等价替换。
4. **训练/部署差别须如实保留。** 现有 NativeWanLinear.forward/main 与packer使用 inference_mode、权重是buffers，没有可用autograd；首版STE训练可以用QDQ，不需要先开发FP4 backward。部署必须验证相同codes/scales进入真实NVFP4 kernel；同码BF16-QDQ GEMM与原生FP4 GEMM已有数值差异，不能要求或宣称整模bit-exact。若保留固定smooth，其输入除scale与W坐标变换须成对保存；最简首版不继承动态smooth/LR闭包。

FP32 master、梯度及两份FP32 Adam状态按300矩阵估计约 **20.74 GiB**，另加约2.59 GiB BF16主权重前向副本、checkpoint激活与临时QDQ；72GB卡具备尝试空间，但旧LR训练的显存/吞吐不能替代主权重实测。无需先证明teacher在人工行为任务上通过；下一步是上述最小训练/导出链自身的数值与梯度正确性，而非新的研究gate。
