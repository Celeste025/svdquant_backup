# E022：扩充现场 teacher 数据的主权重 QAD 强基线

2026-10-03。E020只有4条训练文本/16个旧轨迹状态，64次单样本更新；训练NMSE下降但开发native上升7.77%。E021四个prompt共用一个实际噪声序列，QAD真实视频呈现明显外观变化，MJ与运动指标各有取舍。尚不能定位原因，不把普通QAD或这轮数据/优化改进称新方法；继续建立可用强基线。

数据固定：从已存在VBench提示库按固定随机序抽取32个训练prompt、4个新开发prompt，排除E020已知20个旧校准文本、E021四文本及旧toy文本；不按生成好坏筛选。manifest在GPU采集前保存完整文字、来源、split、选择算法和seed。每prompt两条独立teacher轨迹，各4步，共256训练/32开发状态。为每个prompt/replica分配不同的确定性seed（20261030+2*index+replica），不让不同prompt共用实际noise。现场记录teacher实际输入、输出、timestep、embedding及shared初始/逐步noise，消除旧cache文本/seed来源缺失。仅生成BF16 teacher状态，无VAE；原rCM四步/31,200 tokens/BF16 FLASH不变，GPU0，20分钟/60GiB。

训练从原BF16权重重新创建300个FP32 master，不续E020第64步。固定AdamW、lr3e-6、weight_decay0、clip1、128次optimizer更新。每次累积4个microbatch，依次包含denoise step0/1/2/3各一个，单样本NMSE除4后反传，四组独立shuffle，shuffle seed20261005。每个step有64状态，128次更新使每个训练状态正好出现两次，共512次microforward/backward。沿用已验证legacy-Wan NVFP4 W/A QDQ+identity STE、checkpointing与FLASH backward、无在线LR，不改变量化合同。

初始、32、64、128次更新均在全部32个新开发状态评价QDQ和实际native，保存输出并分别导出packed权重；保存32/64/128 master与最终Adam。预先按这四个候选的native pooled NMSE选最小者（包含未训练0；完全相等取早者），同时保留固定128终点。该选择只优化这个开发读出，不自动代表视频质量最优。之后对候选与plain/BF16/SVD完成独立新prompt/seed视频验证；本计划不预先声明收益。E021四例保留诊断，不再作为最终独立测试。

训练GPU5，启动前确认空闲，训练90分钟/60GiB，日志与tmux监督；512 micro按E020约7.8秒估计66.8分钟，加完整开发评估和导出，预算约80–90分钟。数据另计20分钟。大文件/data1/models/svdquant-wjq/research/20261003/E022，源与报告进仓库，不推送。

本轮同时改变数据覆盖、学习率及梯度累积，因此只能评价更充分的训练配方，不能据其结果单独归因于数据量或timestep梯度冲突。没有额外质量准入门槛；失败/中间结果均保留。模型错误不按事后提示筛选排除。
