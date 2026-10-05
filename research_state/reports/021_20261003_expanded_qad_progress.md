# 021：扩大 QAD 基线与全精度生成异常

2026-10-03；阶段进展，E022 的128次更新仍在运行，尚未得到所选 QAD 的测试视频。**普通主权重 QAD 的第64次更新使native开发误差下降32.22%；但多个 BF16 基线也有明显竖条，改用FP32 VAE未解决，完整视频质量仍待验证。仍无成立的新方法或可投稿核心贡献。**

## 本轮做了什么

E022 从原 BF16 权重重新开始，训练300个主矩阵的约1.392B参数，不带在线低秩分支。新采集32训练文本×2独立噪声×4步，共256个 teacher 状态；另4开发文本提供32状态。288次 BF16 前向已经完成。固定128次 AdamW 更新、lr=3e-6，每次四个去噪步各一个 microbatch，所有训练状态访问两次。数据、顺序与候选选择规则均在结果前固定。

保存的开发输出已由[独立 CPU 汇总](../../results/research/E022/snapshot_step0068.json)重算：

| 验证路径 | 初始 | 第32次更新 | 第64次更新 |
|---|---:|---:|---:|---:|
| 训练用 QDQ | 0.153120 | 0.123031 | 0.104819 |
| 实际 NVFP4 部署 | 0.154799 | 0.123383 | 0.104920 |

四个开发文本和四个去噪步的汇总误差均逐检查点下降；32→64的32条记录也全部改善。首步误差仍明显高于其余三步；这只是当前数据上的描述，不构成新机制。原生与QDQ输出差异能量占QDQ输出能量从初始0.05488降至32步0.01273，64步略回升，不能称该差异单调下降。指标更接近不能替代真实生成质量，32个状态也不是32个独立视频。

[学习曲线](/data1/models/svdquant-wjq/research/20261003/E022/learning_curves/snapshot_step0068.png)。图中在线训练loss只有日志标量，开发曲线来自保存输出；有0/32/64三个完整验证点，不声称已收敛。最后仍按预定规则在0/32/64/128中选择native开发NMSE最低者。

## 视频检验与新发现

8个固定新文本×2独立噪声的BF16、plain NVFP4、旧SVD配方，共48视频已经生成：192次DiT、38,400次真实FP4 GEMM、11,520次BF16 SDPA。全部完整解码为77帧、480×832、16fps，输入与输出核验通过。16份实际初态各不相同。训练与生成并行，672.86秒生成用时不作为性能基准。

root看过全部8张固定replica0、五帧、三臂[对照图](/data1/models/svdquant-wjq/research/20261003/E022/videos/contact_sheets/baselines)。Iron Man、科学馆、庭院、港口、足球场的BF16也可见青绿色竖条；阅读人物和航天器例没有同样的大面积条纹。这是抽帧观察，未据此宣称完整运动质量，也不事后排除坏例。

本地[官方模型README](/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers/README.md:175)明确以FP32加载VAE；旧rCM入口以及E021/E022却将VAE整体置于BF16。已用两个相同的已保存latent（球场坏例、阅读人物对照）完成BF16/FP32共4次解码、0次DiT，52.82秒、峰值13.20GiB。两个BF16视频SHA均精确复现原输出；root查看两张五帧对照，**FP32球场仍有同样明显竖条，因此“VAE低精度是本例主要原因”未得到支持。** 全帧像素MAE分别0.00484/0.00210（解码范围−1至1），不把微小改变说成质量修复。[诊断记录](../../results/research/E022/vae_precision_diagnostic.json)、[球场对照](/data1/models/svdquant-wjq/research/20261003/E022/vae_precision_diagnostic/vbench_197_r0_vae_precision.png)。core128只覆盖一个完整60×104 latent块，无空间拼接；未发现跨视频cache污染证据。仅此对照不能排除其他共同实现或checkpoint配方问题。

48段视频的MJ和AMT/RAFT/DINO均完成并独立复算：[全部原始与分层均值](../../results/research/E022/video_baselines_summary.json)。按两seed先平均、再八prompt等权，BF16/plain/SVD的MJ总分为0.18077/0.07106/0.10172，AMT为0.98365/0.98394/0.98232，RAFT动态阈值通过9/16、6/16、10/16，DINO为0.91226/0.92650/0.90684。这些数值包含全部失败样例；全精度自身的异常尚未定位，不能据此提出量化质量收益或新机制。汇总JSON在解码诊断前生成，其“VAE待查”限制由本报告后续结果补充，不覆盖旧记录。

后续[只读核查](../06_experiments/E022_rcm_reference_check.md)未发现可指认的采样或文本mask偏差：本地官方rCM入口本就使用77帧、sigma80、同四步角度与更新公式；原UMT5和Diffusers都先传attention_mask、再零补至512，原DiT也不屏蔽这些零padding。转换器关键名称/shape/patch映射未见明确错误；但旧等价报告只有随机全长非零文本、小latent与t=500的单forward，**不证明真实短文本和完整自由生成的两实现等价**。原因仍未定位，不能转而宣称模型固有缺陷或短文本机制。

## 接下来

1. 共同生成异常仍保留为未决项。只读核查和VAE对照没有定位原因；若继续核查，应直接比较同实际条件的官方原始rCM完整输出，避免再次用小shape或随机文本代替。暂不批量FP32重解码，也不删失败样例或扩大盲目参数网格。
2. 完成E022原定128次训练、最终开发验证与预定候选选择，再生成同16输入的QAD视频。以完整视频结果检验开发误差改善是否有实际意义。
3. 成熟QAD配方核查已固定官方ModelOpt源码版本：其默认保留校准outer global，而当前每次重算。[单因素固定W-global计划](../06_experiments/E023_fixed_weight_global_plan.md)仅准备、未运行；这是普通强基线补充，不称新方法，也不是完整ModelOpt复现。

原始记录：[训练](../../results/research/E022/train_run.json)、[基线生成](../../results/research/E022/video_baselines.json)、[CPU核验](../../results/research/E022/video_baselines_validation.json)、[成熟配方与先前工作](../01_literature/qad_strong_baseline_recipe_check.md)。
