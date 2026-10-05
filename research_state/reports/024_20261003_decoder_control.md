# 024：完整解码器没有修复当前主要失败

2026-10-03。**E025完成了全部16个固定latent的完整Wan VAE FP32解码及配对评价。动态判定16/16不变，主要语义偏差仍在；当前证据不支持快速TAEHV是主要原因。结束这条VAE排错线。**

复用E024全部8文本×2seed的final latent，没有重跑DiT或选择更好seed。新臂使用同官方模型包的完整Wan VAE资产、Diffusers FP32与一次合法反归一化；原臂为官方TAEHV FP16。因此是decoder实现、权重和dtype的联合介入，不能拆成单独精度因果。全部81帧480×83216fps；16次解码320.16秒、实际峰值13.22GiB，含诊断，不是部署速度测试。

| 描述性均值 | TAEHV FP16 | 完整Wan VAE FP32 |
|---|---:|---:|
| MJ总分 | .322401 | .332559 |
| 文本对齐 | .307388 | .318924 |
| 细节 | .097620 | .100750 |
| 连贯性 | .570618 | .576050 |
| AMT | .994420 | .994401 |
| RAFT动态通过 | 2/16 | 2/16 |
| DINO主体一致性 | .979017 | .976470 |

MJ按两seed先平均、再八文本等权。逐文本差值只有3正5负、逐seed8正8负；航天器题差值+.161784，足以解释总体+.010158，其他七题均差为−.011503（只是贡献分解，未从统计中剔除）。RAFT仍仅航天器两seed通过；DINO16条全部下降。高AMT/DINO不能替代正确动作，且八题中四题仅场景名，本批次不能作为专门运动基准。

已查看全部8个replica0的固定五帧配对图：科学馆仍为彩色瓶子、港口仍为画笔水杯、Iron Man仍为披风人物；完整VAE有细节变化，没有修复主要内容偏差。这是抽帧观察，不冒充完整运动人工盲评。[前四题](/data1/models/svdquant-wjq/research/20261003/E025/contact_sheets/first4_r0.png)、[后四题](/data1/models/svdquant-wjq/research/20261003/E025/contact_sheets/last4_r0.png)。

同一latent、全部媒体及AMT40项/RAFT40项/DINO80项归一化与MJ汇总均由CPU独立核实；[配对汇总](/home/wjq/workspace/svdquant-exp/results/research/E025/evaluation_summary.json)、[解码原始记录](/home/wjq/workspace/svdquant-exp/results/research/E025/decode_run.json)。未证明两解码器质量等价，也未定位上游蒸馏或量化的原因。

官方采样合同也已收尾：[发布历史核查](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E024_official_sampling_contract.md)支持现有三步UniPC/shift3，训练DMD配置未绑定该公开权重，不换采样器追结果。

下一项E026转向有明确代数问题的算子诊断：复用已存真实H3层FP4数据，固定前六个packet，只用μ×K4替换块均值校正。两次attention、零DiT，先测复用低位K的误差下限，再决定是否值得开发省去二次中间矩阵的内核。已有近邻与额外MMA成本均已记录；当前仍无新方法、等质量加速或可投稿核心贡献。
