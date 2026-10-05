# E052：原版Wan14B首block完整校准资源测量

模式：Exploration / baseline resource。E050仍在采集本模型匹配64records，E051 plain完整视频对照在运行；本任务不改两者，不主张新算法。

**问题与假设：** 同64records、batch4、g10、rank32、最多100次LR/原early-stop的SVDQuant校准，真实14B首block在72GiB单卡能否完成，主要时间与内存花在哪里？需要实测来决定整模校准或资源调整，不能从1.3B耗时/14B生成耗时线性外推。

**最小有效设置：** 运行须E050全部14轨迹/64cache及独立CPUsummary complete。载入固定官方14B为BF16，全40个block/400main Linear权重始终在同GPU；只选择真实block0全部10个main Linear。调用原smooth_diffusion_layer完成g10完整OutputsError，再从同64记录重新采集已平滑block0激活，调用原calibrate_diffusion_block_low_rank_branch覆盖全部self/cross attention和FFN的共享LR组。64样本/sample_size=-1/b4/r32/LR100earlystop/两级scale/E2M1、禁旧gated、Diffusers0.40 RoPE tuple和FlashSDPA均保持。随机seed0仅固定本pilot的随机SVD，不能承诺与原整模先全smooth再全LR的随机顺序逐位相同。

**截断边界：** 限制loader返回层列表为真实block0，用已有early_stop_module=blocks[0]截断下游执行。原模型的patch/text/time等输入变换保留；禁止以toy/孤立随机矩阵、1.3B中间latent或少量样本替代。用实际hooks记录block0/后续block调用次数，后39层实际0forward；权重仍驻留，不能借释放后层制造虚假可行性。LR需要平滑后重新采集，不复用原smooth前activation。

**必要证据：** 每阶段model load→原始activation采集→完整smooth→重新采集→完整LR的同步wall、GPU allocated/reserved峰值、CPU RSS，完整样本与模块覆盖、实际候选/早停记录。保存partial_smooth/partial_branch及来源/配置，明确不构成完整checkpoint；既有ptq把非空cache当全量，不能直接导入部分产物。

**预期与决策：** 完整完成则依据实测成本与阶段瓶颈决定是否承诺整模PTQ，另冻结执行协议；首层成功不证明后层无更高峰值、质量恢复或总时长。OOM/超时则保存真实失败和已完成阶段，定位内存/时间瓶颈，再决定保持数学样本和搜索配方的执行调整；不得静默缩预算或把实现失败当算法质量失败。仅总成本问题可推进工程调整，不生成新研究claim。

**预算/停止：** GPU4在原E050完整结束且真实空闲后才启动；前置等待≤7500秒，worker7200秒、launcher7500秒。named tmux，日志results/logs，数据DATA1；核两次空闲及启动前瞬时状态，不抢E050/E051、GPU6/7不动。失败保留、不自动重试、不自动继续全模型PTQ。CPUcheck只config/meta40/400/目标10，不要求E050已完成，不载权重或CUDA。
