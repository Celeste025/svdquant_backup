# E038：coarse16是否带来实际视频质量需求

2026-10-03，生成前固定。状态prepared；模型输出尚未生成。

E036同coarse16合同降低了两卡完整attention成本；E037 global更便宜而局部误差较高。E017已表明更低tensor误差未必改善视频。下一投入直接检验生成，不再扩层/中心数量/拓扑网格。

## 固定样本与干预

公开VBench_full_info.json的零基索引161、192、269、316，分别为拍手、叠衣服、汽车转弯、大象喷水，原文不改。四种动作类型为生成前有目的选择，不是随机代表全基准；未用本批输出筛选，亦不是旧H3 prompt表的ID。每题两个独立固定seed，完整记录见 E038_center_video_manifest.json。只排查与已用中心开发文本重叠，不声称训练数据无污染。

三臂同E009 native SVD线性权重：bf16 attention、现成global_mean、coarse16。所有50主attention采用该臂选择；2个token refiner及独立模型padding段保留原BF16。coarse16沿用E034数值规则：G=ceil(N/128)，第j段边界floor(jG/16)个128-token块，padded Q直接BF16均值，K均值只含有效token，原K计算FP32 correction；使用已完成的共享行consumer。真实N随prompt改变，不写死开发样本长度。

原H3采样：576×1024、124帧、20步、cfg=1、flow_shift=12、audio_flow_shift=3；同一VAE与分块参数、24fps。每文本TE一次，8case真实CPU初始video/audio noise保存后由三臂读取，不能只用同seed声明一致。每臂自由递推20步，保存最终latent、逐步小量诊断、真实attention/native线性调用计数。总480DiT、24视频，不加网格/重采好样本。

## 评价与解释

主结果为MJ-VIDEO total/alignment/fineness/coherence，安全和bias只留原始输出；辅助复用AMT、RAFT动态判定/原始量、DINO时序一致性。按同prompt/seed配对，报告逐题和两seed符号，不把帧/问题数当独立样本。生成成本是实际完整过程诊断；不把E036两卡单层测量乘层数当整模加速。

保留全部24视频、失败记录和原始评分。可制作统一时刻抽帧供观察，但不得将抽帧观察称为完整视频人类盲评；任何可用视频观看工具观察要写明范围。动作幅度、平滑或奖励均值不是单独质量真值。若评价冲突，结论为不确定，不自动追加参数扫描。

继续条件：出现可复核、跨seed保留的具体global语义/时序损伤，而coarse避免，才有理由投入完整系统集成；仍须更大独立验证和最近工作核查。若无清晰分离或coarse退化，停止扩大中心表示路线，保留工程结果。不存在任意百分比发表门槛。

## 执行边界

大文件在 /data1/models/svdquant-wjq/research/20261003/E038；代码与小JSON留仓库。GPU0/1/5分别三臂，启动前检查空闲；TE单进程准备，生成后独立decode，再评价。具名tmux和日志、进程组deadline：prepare600秒，每臂生成1800秒，每臂decode600秒，评价1200秒。峰值allocated守卫60GiB。仅清理本实验子进程，不触碰他人负载。旧E010/E017/E034等源码、输出不改。
