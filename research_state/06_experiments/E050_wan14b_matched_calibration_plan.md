# E050：原版Wan14B匹配校准缓存

模式：Exploration / baseline preparation。当前没有已成立论文claim。E049是原版14B BF16实际参照；为判断1.3B量化损伤能否延伸到更大模型，必须建立本模型SVDQuant基线，不能迁移1.3B权重/低秩/平滑/scale，亦不能只用plain代表强基线。

**待回答问题：** 原版14B在当前真实50步视频采样合同下，能否产出完整且可复用的64条真实DiT I/O校准缓存，实际资源成本多少？本实验仅解决采集入口，不检验新的量化机制或质量收益。

**预期与决策：** 若14条完整轨迹/64cache均有效，进入保留64records、batch4、g10、rank32及LR100早停的真实首block校准资源测量，再决定完整PTQ预算；若执行OOM/超时/不一致，保留失败，定位具体资源或实现问题后新版本修复，不把运行失败视为量化质量失败。后续资源pilot和完整PTQ不由本次采集自动启动。

**最小有效设置：** 继承E047固定Random0选择的64条/14提示/原seed及同prompt SHA；官方原版14B固定HF revision38ec498cb3208fb688890f8cc7e94ede2cbd7f68，E049的CFG5/shift3/81帧480×832/50步UniPC，FP32 sampler与BF16 DiT FlashSDPA。复用已确认与14B相同TE/tokenizer身份的E047真实正负embedding；每条从CPU种子初噪独立完整运行14B轨迹，实际输入在forward前CPU保存、实际output在forward后保存。共享FP32 VAE仅供pipeline结构配置，禁止decode；TE/tokenizer不加载。记录每条实际初噪、scheduler、每选中分支/时间/输入/输出和运行计数。

**避免失真替代：** 不用1.3B中间latent/output；不缩帧、不减步、不只跑一两个时间点，不用单层synthetic输入代表完整采集。所选35/29分支不平衡、35/50时间位置且缺0/48/49是继承fast政策的边界，保留而不新开平衡网格。这里生成潜变量，不需要为每条额外解码视频。

**必要证据：** 四worker真实complete，14完整轨迹、64唯一cache及actual输入/输出元信息与manifest一致，1400DiT/112000SDPA/700scheduler/64cache/0TE/0decode；CPU汇总确认selection、timestep、branch、actualembedding及初噪重放。这些是执行/输入保证，不是质量分数或byte一致性科学门槛。

**预算与停止：** groups0..3分别GPU2/3/4/5，每worker5400秒，总launcher7200秒，启动前空闲等待最多900秒，两次空闲观测并立即复查；若E049尚占用则等待，绝不干扰现有任务。不使用GPU6/7，不kill他人进程，任何失败停止所有本次owned worker并保存，不自动重试。named tmux，日志results/logs。数据/cache放DATA1；不改已执行实验源。

本次结束后仍缺完整14B SVDQuant checkpoint与质量对照，不能称基线准备完成或新的科研贡献。Plain逐层pack代码可并行准备，但不替代本强基线。
