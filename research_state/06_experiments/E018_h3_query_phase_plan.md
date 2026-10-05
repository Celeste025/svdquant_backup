# E018：固定真实 QKV 的 query 分组相位诊断

2026-10-02，GPU 前登记。目的仅为检验一个机制线索，不是新方法或生成质量试验。E017 的单例评分差不作为机制成立证据。

## 问题与碰撞

H3 按 `[text | audio | video | pad]` 打包无条件视频，video 按 T/H/W、W 最快展开。p36 的 video 起点 1227，37 个 latent 帧，每帧 18×32=576 query；官方连续 128-query Qmean 不在模态/帧边界重置，帧起点 mod128 为75/11交替。分组量化依赖排列本身是预期现象；PAROAttention 已研究视频布局与低位量化，按帧重分组不自动构成贡献。

本次问：在同一真实输入、K/V 包和 KV 遍历不变时，改变 Qmean 的分组相位，是否产生随人为相位改变的帧内/帧间误差结构？不把一个非零排列差异判作阳性发现，不推断输出视频有两帧闪烁。

## 最小输入与执行

- 固定 E017 block_mean 的 p36/seed59526 自由轨迹第14步自己的 before video/audio，复用原 p36 embedding、timestep 和 packing。只重放一次完整原 DiT；actual kwargs、双 raw output 和 unpacked velocity 与 E017 对应记录 byte exact。
- 在 blocks 0/24/48 的原 post-RoPE attention helper 保存有效段 Q/K/V 与原 attention 输出。三块预先固定为早/中/晚位置，不能看结果换层。模型、200 个原生 SVD linear、50 个 FP4 attention 与52个 BF16 segment保持原调用。
- 捕获结束退出模型进程，再单独加载三个固定 QKV。对每块，将 video Q 在自己的21312-token区间循环移位0/64/128，text/audio Q、全部 K/V 顺序与值不变。输出逆置换到原 query 身份。官方 block-mean 每次重算 Qmean/Q pack/correction；K/V 使用第一次的同一份 packet，验证官方重新生成的 K/V 数值 byte 相同。
- 两类控制同样做0/64/128：原 torch BF16 SDPA；官方 global-mean 先在基准顺序计算一次 Qmean/centered Q，再只置换 centered Q 并复用同一 correction，以排除 reduction 次序漂移。global 的0相位必须与官方 global API逐包、逐输出对齐。
- 128位移是内部完整 Q-group 保持控制；64位移改变组成员。循环接缝和模态边界不用于主统计：固定只分析 latent 帧2..34，共33帧；三种相位都用同一 original-query mask。仍保存全部输出与全区间统计，不隐去边界。

## 输出与解释

保存三块全部 QKV/输出、每个模式/相位的输出、packet/源绑定、逐query误差能量及控制差异。统一以同一个 BF16 baseline 输出为参考；报告相对误差和原 query 对齐的 delta，不只比较两个模式各自 NMSE。

主诊断：逐帧错误能量、相位64相对0的配对能量变化、33帧的交替分量、按帧内位置/分组相位展开的误差图；同时报告128控制与global/BF16控制。内容本身可有时间结构，单个 odd/even 对比没有因果解释；需要配对结构随人工相位移动且控制不产生同样结构。三层也不是独立视频样本，不做虚假显著性检验。统计只是决定是否值得下一步验证，不用拍脑袋的固定百分比阈值判机制普遍成立。

交替分量使用 `0.5×(even帧均值−odd帧均值)`，避免33帧的不等样本数把常数漏为交替项。误差差值同时分解为 `2<e0,delta>+||delta||²`。以真实帧边界±32 token和帧内+256 token的同相位伪边界±32作比较，后者保留128组相位，不能把同样出现于帧内的条带称为时间边界特效。block phase0原输出必须精确重现；global phase0使用官方pack，其他相位冻结同一全局中心化。

固定K/V不等于固定P：Qcenter/code/correction变化会通过logits改变P scale/code。这里测完整operator的干预效应，不将其单独归因于Q4重构误差。实际Q packets保留供核查，本轮不增加P消融或Q-only重构实验。

若只看到普通的组内量化敏感性，没有清晰可复现的相位随动结构，就停止当前“帧相位”解释。即使有结构，也不立论文主张：仍需验证真实部署影响、跨输入证据，以及 global mean/普通重排等已知修复无法解决的约束。

## 资源与不可变性

物理 GPU5，启动前检查连续空闲；tmux 与独立日志。最多1次完整 DiT、每层9次独立 attention 调用，共27次，不生成/解码视频，不扩提示/seed。共享墙钟预算900秒，模型加载峰值上限60GiB；捕获与probe两个进程依次运行。CPU预检不初始化CUDA；失败保留日志，不悄悄重置预算。源、协议和依赖在GPU前冻结，E005–E017完全不改。数据在 `/data1/models/svdquant-wjq/research/20261002/E018`，小报告在 `results/research/E018`。
