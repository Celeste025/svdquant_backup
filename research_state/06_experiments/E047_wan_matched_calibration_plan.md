# E047 — 原版Wan匹配校准基线

2026-10-03。D072后，先补匹配当前采样分布的SVDQuant基线，不把旧33帧/不同schedule的缺陷当新机制。本轮不提出新方法，也不承诺质量改善。

## 问题与固定设置

原版官方Wan2.1-T2V-1.3B（沿用E043模型身份），81帧480×832、UniPC50/CFG6/flow_shift8，BF16 DiT/TE、FP32 sampler，FlashSDPA。校准不解码、不生成评分媒体。16个原校准提示由原YAML与Random(0)确定；从其虚拟1600文件名按原loader Random(0)确定64条，仅需执行其中14提示的完整50步轨迹。校准文本均不与E043的四诊断文本重复。相同hash seed规则、CPU FP32初始噪声沿用本轮参考流程；不同形状/runtime不称旧轨迹相同。

**采样局限明确保留：** 64条中cond/uncond为35/29，实际35个时间位置、14/16提示，step0/48/49未入选；仅两个(prompt,step)双支成对。这是原fast抽样政策，不据此宣称错误，也不加入末步定向重采样。采集之前固定完整名单，评分不反向改名单。

## 实施与预算

1. 四worker GPU0/1/2/3各3–4提示，每个提示100次实际DiT、50次scheduler及2次TE。总1400 DiT/700 scheduler/28 TE/0 VAEdecode。仅指定64次保存真实模型I/O，格式兼容DeepCompressor CollectHook：input_args=[BF16 hidden_states]、input_kwargs（含实际timestep及embedding）、outputs及filename/step/guidance。保存初态、终态、embedding与50步标量记录，不存每步大latent。DATA1目录独立，禁止覆盖旧缓存。
2. worker预算1800秒、launcher2100秒；named tmux，启动前复核GPU，超时只终止自身进程组。失败收据保留，不自动重试。一次CPU入口检查后冻结源。
3. PTQ使用现有DeepCompressor real_nvfp4+wan_s16：rank32、group16、显式W/A FP32 tensor-global/E4M3 group scales、g10、64 records、OutputsError、low-rank最多100迭代early-stop、shift_activations=false。保持batch4/sample_size=-1；如运行资源证明必须更改，明确另版本记录，不静默缩小样本。只用原BASE transformer，不载rCM覆盖。旧成功日志耗时1小时59分47秒；81帧校准尚无实测时长或峰值。预定GPU4单worker上限21600秒、launcher21900秒，不能从旧日志推定新峰值。当前Diffusers0.40的RoPE为tuple，入口仅对关闭的历史gated扩展作局部兼容，不改变普通OutputsError配方；执行环境与新采集相同。
4. 全部64cache完成后才能PTQ。完整模型五产物、rank/quant配置与实际数据身份记录后，再使用E044现有native消费路径，在八个已有诊断case上完整生成与相同MJ/AMT/RAFT/DINO评价。生成协议单独冻结，禁止按结果丢例或改seed。

## 结论边界与下一决策

比较新匹配基线与旧SVD、原BF16、E046末步强控制；主看每例视觉和MJ四项，时序指标辅助。匹配基线改善是基线补全，不是论文贡献；无改善不证明所有NVFP4或VAE失败。帧数/schedule同时改变，不作单一因素因果归因，也不称本地路径是完整官方Wan canonical复现。完整结果出现前不扫rank/网格/保护步数，不做新loss或patch相位GPU实验。

Manifest：[固定数据与配置](E047_wan_matched_calibration_manifest.json)；起点：[基线审查](../00_state/baseline_readiness_after_E046.md)。
