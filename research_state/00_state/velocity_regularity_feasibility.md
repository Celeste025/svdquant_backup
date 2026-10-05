# 速度场局部规则性/第二次评估：CPU可行性审计

2026-10-02；只读源码与现有CPU缓存，未写runner、未启动GPU、未修改E009/E010冻结源。**技术上可做，但不足以支持新方向；root已找到Sampling-Aware Quantization/QuAKE直接相关工作，不建议据本审计追加实验。**

**H3可原样复用函数；“BF16连续场”不成立。** 当前H3是flow-matching形式：双scheduler给video/audio不同sigma，原更新为 `x + (sigma_next-sigma)*F`，只有Euler、没有误差控制器或高阶求解器。原 `model_fn_minimax_h3` 接收两个FP32 timestep，内部各变为 `1-timestep/1000`，原DiT输出经unpack并取负才是scheduler消费的F；不能把raw DiT输出直接当方向。见 [双时间与调用](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:180)、[model_fn入口](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:944)、[输出符号](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:1063)、[Euler更新](/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py:377)。

**同一(x,t)可对照native/BF16，无需改model_fn。** E010每步已经保存 `latents_before/noise_pred/latents_after/timestep/sigma`，video `[1,24,37,36,64]`、audio `[2,32,207]`均BF16，时间和sigma为FP32。例如 `/data1/models/svdquant-wjq/research/20261002/E010/denoise_bf16/p030/s05_{video,audio}.pt`；同目录s14可作另一位置。`prepare/p030.pt`含完整embedding/tags；用原 [PackedSequenceBuilder.process](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:827) 重建packed，再直接调用原model_fn即可，CFG=1、无条件锚点/ControlNet。首先要求BF16中心点输出与缓存SHA一致。旧 `results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt` 则是更低层的packed DiT输入/输出，已经绑定视频/音频位置；仅改x而不顾audio_x/符号/输出选行容易错。E010保存契约见 [step记录](/home/wjq/workspace/svdquant-exp/scripts/research/run_h3_native_paired_video.py:341)。

**teacher有两个数值台阶源，不能把所有跳变归因FP4。** 实际checkpoint是ComfyPruned：1025×8 FP32时间表，线性插值后cast BF16；表节点本身是导数折点，不是数值跳跃，但BF16 cast产生台阶。输入投影也强制转为embedding的BF16 dtype。见 [时间插值](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:39)、[输入cast](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:301)。step0两时间均0，`unique`分组发生合并；端点有clamp。因此第一项应固定两个t，只扰动x，实际输入逐元素变化比例/有效delta要记录，不能用名义epsilon外推无穷小Lipschitz常数。

| CPU缓存检查：沿下一步BF16方向的1/16位移，再round BF16 | video变化坐标占比 / 相对L2 | audio变化坐标占比 / 相对L2 |
|---|---|---|
| p30 step5 | 13.92% / 0.000478 | 28.30% / 0.001477 |
| p30 step14 | 62.82% / 0.004386 | 80.12% / 0.006800 |

这里方向为每模态各自 `d_m=(sigma_next_m-sigma_m)*F_B,m`，不能共用video timestep或把video/audio不同幅度当作内在敏感性。末步sigma为video **0.3870968→0**、audio **0.1363636→0**；p30对应未round更新相对L2 **0.6937/0.2852**。这足以使“晚步更差”与粗Euler teacher混杂，绝不是量化专有证据。

**如果以后仅作排错，最小有控制的诊断：** 固定p30 steps5/14，以缓存BF16方向构造中心及±1/16、±1/4步的BF16输入，两个arm消费完全相同的实际输入，固定t；测量 `D(delta)=[F_Q(x+delta)-F_Q(x)]-[F_B(x+delta)-F_B(x)]`，单独报告两模态及BF16自身增量/曲率，不用两个NMSE相减冒充交互。2点×5输入×2arm=20次完整DiT，加中心重复2–4次；按E009测量约 **3 GPU分钟**，加加载/诊断预留 **5–10分钟墙钟**，无TE/VAE/自由生成。只有超出BF16舍入背景且可重复的残差变化才值得继续；阴性直接停止，不扫seed、epsilon或步数。

动态FP4的全张量g、组E4M3 scale与E2M1 code都会影响结果（[定义](/home/wjq/workspace/svdquant-exp/scripts/research/probe_h3_native_contract.py:25)）。flip率与D相关仍非因果。若确需区分，最便宜的后续是**同一真实block0 qkv输入的纯主支**，依次动态量化/固定中心g/固定中心g和组scale但重算codes，中心packet必须逐byte一致；这样才可区分尺度与code效应。全模型冻结所有层scale会同时改变后续activation分布，不能直接归因单一来源；当前没有已验证的H3 linear固定grid adapter，不可把attention固定scale helper未经验证照搬。

**第二次求解器评估要另定问题，当前不实施。** 可沿同一BF16 predictor点同时评估两个arm，显式用两个sigma的新时间；这是受控场差异，不是各arm真正Heun改进。若两arm用自己的predictor，则第二点不同，混入第一步误差传播。原scheduler.step按最近离散时间选索引，不能拿它做任意half-step而声称连续时间积分；改变步数、末步尺寸或采样算法也不能再与原20步协议直接归因比较。

**Wan更不适合作为现成高阶solver问题。** 当前是rCM checkpoint与4步重噪更新 `(1-t_next)*(x-t_cur*v)+t_next*z`，不是这份H3 Euler。latent递推FP64，但模型输入与标量timestep都cast BF16；接近timestep1000时BF16 spacing=4，即归一化时间约0.004，细time扰动可被完全吞掉。见 [原rCM采样](/home/wjq/workspace/svdquant-exp/scripts/infer_rcm_wan_4step.py:209)。E008 `p0001/bf16_step*.pt`保存FP64 latent和BF16 velocity，旧Wan raw cache保存BF16 hidden_states/timestep/embedding；可做固定t的场排错，但拿rCM随机refresh或不同噪声下差异解释高阶不一致无效。不能把“增加噪声会变坏”、两次调用成本翻倍、或常规Heun比Euler好重新包装成量化机制。
