# E022：固定两条原始 rCM 架构完整轨迹对照

2026-10-03。入口：[compare_rcm_rollout_reference.py](/home/wjq/workspace/svdquant-exp/scripts/research/compare_rcm_rollout_reference.py)。这是独立诊断，不修改 E022 既有生成源、manifest、训练或媒体。GPU 由 root 审阅后启动。

**问题与范围**：在同一真实文本条件、初始状态和每步噪声下，本地原始 rCM architecture/.pt 的完整四步输出，是否也出现当前转换实现可见的条纹？比较对象是两条执行路径整体；不把差异单独归因于权重转换、某个 dtype 或短 prompt。没有任意 exact/NMSE 质量通过门槛，也不据静态截图评价运动质量。

- 固定案例：BF16 `vbench_197_r0`（football field，seed 20261124）和 `vbench_133_r0`（Gwen reading，seed 20261118）。无样例替换。
- 从 `results/research/E022/video_baselines.json` 绑定每例既有 shared-input 文件及 BF16 final latent，复用实际 FP64 initial latent、四份 FP32 update noise、FP64 t_steps、BF16 embedding；不重新抽噪声，也不只依靠相同 seed。
- 原模型：`/home/wjq/workspace/rcm/rcm/networks/wan2pt1.py` 的 `WanModel`，原权重 `/data1/models/svdquant-wjq/models/rcm-Wan/rCM_Wan2.1_T2V_1.3B_480p.pt`。去掉 `net.` 前缀、忽略 `accum_` 训练计数，严格加载原 key/shape。保持非因果 T2V、480×832、77 帧、sigma80、四步官方 schedule/velocity 公式、无 CFG。
- 原 UMT5 完整 `.pth` 不在本地，只有 `.pth.incomplete`。按 root 指示，两实现使用 E022 已保存的真实 embedding，不加载残缺文件、不下载、不重建文本编码器。保存实际 embedding 和与缓存的精确身份记录；**不验证原 UMT5 端到端等价**。
- 原始 Q/K/V、归一化与 RoPE 计算来自原架构；仅将 dense attention 的 local consumer 固定为 PyTorch BF16 `FLASH_ATTENTION` SDPA，并计数。输入 layout `[B,S,H,D]` 转 `[B,H,S,D]`，无 mask、dropout=0、noncausal、默认 `1/sqrt(head_dim)` 缩放。RoPE 使用原模块现成 PyTorch FP32 fallback。不是原 flash-attn 二进制复现，也不宣称旧 verify 脚本已有经过验证的独立 adapter。
- **显式运行兼容精度政策**：公开入口整体 `.to(BF16)` 与本地原 forward 的 FP32 time/head 输入、断言存在冲突。保留 `time_embedding`、`time_projection`、`head` 和 affine `WanLayerNorm` 为 FP32，其余主干参数及 attention BF16；保持 `WanRMSNorm.weight` BF16。记录全部实际参数 dtype 及每步 velocity dtype。原输出保持实际 dtype，直接转 FP64 更新，不强行降为 converted 的 BF16 输出。root 已同意此政策；它不等于逐字运行旧官方入口，两路径也不具备全算子同精度合同。
- 只新增 2×4=**8 次原架构 DiT、480 次 SDPA**。保存每步 FP64 state before/after、实际 BF16 input/timestep、velocity 及共享 noise 标识；保存 final latent，并读出与已保存 converted final 的连续误差，不据误差设质量门槛。
- 同一 FP32 VAE 对原架构新 final 和转换版旧 final 各解码两例，共 **4 次 decode、4 个完整 77 帧视频**。两边统一 `FP64→FP32; z/(1/std)+mean`，core128/halo0 单空间 tile，每次清缓存。固定导出帧 `[0,19,38,57,76]`、两张两行对照图和像素统计；旧媒体全部保留。

预算：GPU0、15 分钟 supervisor 总墙钟、60 GiB allocator 限额及峰值检查。失败保留部分结果，不自动扩大预算或重试网格。CPU 准备仅校验两例输入、原 checkpoint key/shape 元数据，以及一个小随机未训练模型的 dtype/layout 执行；后者不作为研究结果。

默认结果：`/home/wjq/workspace/svdquant-exp/results/research/E022/rcm_rollout_reference.json`；媒体/逐步张量：`/data1/models/svdquant-wjq/research/20261003/E022/rcm_rollout_reference/`；CPU 准备记录：`results/research/E022/rcm_rollout_reference_cpucheck.json`。

root 审阅及确认 GPU0 空闲后启动：

```bash
cd /home/wjq/workspace/svdquant-exp
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python scripts/research/compare_rcm_rollout_reference.py --supervise --wall-seconds 900
```

supervisor 写 `results/research/E022/rcm_rollout_reference_supervisor.json` 和 `results/logs/E022_rcm_rollout_reference.log`。现有输出存在时停止，不覆盖。
