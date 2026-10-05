# 下一轮 QAD 强基线：扩充真实 teacher 状态，再做一次固定配置续训

2026-10-03，只读可行性建议，未启动 GPU。E020 的4条 prompt/16个固定状态已被64次更新覆盖4遍；因此“多跑同样的更新必然改善”没有依据。数据覆盖不足是合理假设，时间步之间的优化权衡也有直接线索：两条开发 prompt 的 native pooled NMSE，在 step0 为0.264865→0.345530，而 step1–3 为0.099850→0.089722。这不能单独证明过拟合、梯度冲突或学习率原因；下一轮先建立更充分的普通 QAD 基线，不开发新损失、不引入收益门槛。

**数据建议：32条新训练 prompt×2条噪声轨迹×4步＝256个状态，另留4条新开发 prompt×2轨迹×4步＝32个状态。** 从已有251条 VBench manifest 按三个 dimension tuple 轮转、组内按 case_id 排序选择，先训练32条、再开发4条；排除旧20条缓存文本、旧 dev `0001/0002`，以及 E021 的 `vbench_066/091/182/067` 的规范化完整文本。不能按数字编号跨命名空间排除：缓存 `0001` 与 `vbench_001` 不是同一条 prompt。选择不看视频或评分；manifest 中既往生成过的视频可以对应新的训练状态，但须标为“新加入本轮优化器”，不能称全局未见 prompt。旧 dev 两条继续作历史对照；E021 四条保持独立自由生成观察，已有反复开发用途，不升级为最终测试集。

每个 prompt/轨迹使用显式不同 seed，例如 `20261010+2*manifest_index+trajectory_index`，保存实际整数及 RNG/噪声 SHA，避免所有 prompt 共用两个噪声张量。E021 四条目前使用同一 seed，初始与更新噪声 SHA 也完全相同；它适合配对比较，却不构成广泛噪声覆盖。所有新状态均沿 **BF16 teacher 自己的完整四步轨迹**生成，不用独立随机 latent 冒充后期状态，不注入 student rollout，也不增加 VAE 解码。

最省实现是新增薄状态采集入口，复用 [E021 generator](../../scripts/research/generate_wan_qad_comparison.py) 的资产加载与 `encode_prompt/prepare_latents`（173–205行）、BF16 DiT 与 FP64 rCM 更新（281–300行）；把每步实际 BF16 latent、BF16 timestep、teacher velocity 写成 E020 trainer 已接受的 `input_args/input_kwargs/outputs` 结构。embedding 每个 prompt 保存一次并引用，共同初始 latent/四份更新 noise 每条轨迹保存一次。模型保持480×832、77帧、31,200 tokens、sigma80、guidance0与 FLASH attention。先编码36条文本并释放 text encoder，再加载显式 `rcm-Wan2.1-T2V-1.3B-Diffusers/transformer`；不要直接复用旧 collector 的根模型路径，因为库存已记录 rCM 根目录存在失效模型链接。36条文本来自本地 manifest，现有 base text/tokenizer 资产可直接复用。

**单一训练配置建议：续接 E020 第64步 master＋Adam 状态，新增128次 optimizer update，lr固定3e-6，weight decay 0、clip1，microbatch1、梯度累积4。** 每次 update 从四个 timestep 桶各取一个状态，四个 NMSE 各乘1/4，分别 backward 后统一 clip/step；桶内固定种子洗牌，尽量混合 prompt/轨迹。这样512次完整训练前向/反向恰好覆盖256条新状态2遍，每次更新均包含四步，没有新增时间步权重。仍用 block checkpointing 和已有 STE，FLASH context 覆盖 backward 重算。较小学习率和累积只是保守优化选择，不声称已解决观察到的权衡；数据和优化同时变化，收益也不能作单因素因果归因。

| 阶段 | 有据估算（单卡，不含排队） |
|---|---:|
| 288次 BF16 teacher 调用 | 288×1.815秒≈8.7分钟，另加文本编码/加载/落盘 |
| 512次 microbatch 前向＋反向 | 按 E020 7.826秒/完整单步保守估算≈66.8分钟；累积仅每4次执行一次 optimizer，实际可能较低 |
| 续训前/新增64/128步，40条 dev native 读出 | 120×约1.66秒≈3.3分钟，不另外重算已有 teacher |
| 总预算建议 | 100分钟封顶，含文本编码、checkpoint/导出与余量；不是当前45分钟原预算内的任务 |

数据约2.3 GB的288组 BF16 input/target，加独立 embedding 与可重放 FP64 initial/FP32 noises，总计约6 GB；master/optimizer/导出另外预留约25–30 GB。E020 实测训练 peak allocated 25.423 GiB；microbatch仍为1且每次 backward 释放图，累积4不应按4倍激活显存估算，但新的 full-shape 输入峰值仍以实际记录为准，沿用60 GiB资源上限。数据留CPU/磁盘，teacher、text encoder和student不同时常驻GPU。

记录历史 dev、四条新 dev 的 pooled/per-prompt/per-step native 误差与固定末步导出；新增64步检查点只用于学习曲线，本轮主要终点仍是预先指定的新增128步，不按最漂亮的中间点追改终点。先完成 E021 质量评价，再用固定导出的完整视频判断是否兑现收益。无需等待额外机制审计，也不把这套常规数据扩充、累积梯度或 QAD 作为研究贡献。

依据：[E020库存](../../results/research/E020/E020_cache_inventory.json)、[独立数值汇总](../../results/research/E020/summary.json)、[训练耗时](../../results/research/E020/train_run.json)、[E021实际生成与噪声记录](../../results/research/E021/generation_run.json)、[部署计时](../../results/research/E020/deployment_bench.json)。本建议尚未生成新的 prompt manifest、训练状态或模型。
