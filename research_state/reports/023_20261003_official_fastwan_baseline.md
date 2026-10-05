# 023：官方FastWan-QAD产品基线已实际跑通

2026-10-03。**E024完成了官方NVFP4线性＋FP4自注意力的完整模型运行、固定16段视频和全部评价。其MJ均分高于当前rCM各臂，但内容对齐仍不稳定，运动指标较低；这是公开产品参照，不是本项目新方法或等质量加速成果。**

## 实际运行的是什么

固定FastVideo `8444c089`、HF FastWan-QAD-1.3B `621c6aeb`、TAEHV `011dfc21`。新建独立torch2.12/cu130环境，使用官方SM120 kernel；新transformer实际5.676GB、TAEHV约22.7MB，均核对公开哈希。本地23.25GB的文本编码器/VAE/tokenizer经完整SHA核对后复用，旧环境和模型未改。[环境记录](/home/wjq/workspace/svdquant-exp/results/research/E024/environment_summary.json)、[完整资产记录](/home/wjq/workspace/svdquant-exp/results/research/E024/assets_manifest_attempt3.json)。

复用官方build_generator和FP16 TaehvDecoder，三步、CFG1、81帧、480×832、16fps，关闭compile；self-attention为官方SM120 FP4，cross-attention按官方回退为dense SDPA。实际scheduler是FlowUniPC/shift3，timesteps **[999,857,599]**。单条smoke成功后，完整保留E022原8文本×2seed的16条产品测试；不同帧数、权重、scheduler使实际noise不等同rCM，不能称同噪声量化对照。

正式suite实测 **48 DiT、14,400 NVFP4 GEMM、1,440 FP4 self-attention、1,440 dense cross-attention**，16段全部完整解码为81帧。全部生成及评价进程已退出。smoke含首次CUTLASS编译，209.7秒pipeline用时不作速度基准。正式suite在缓存就绪后用时112.69秒，含15.35秒构建；每条pipeline中位4.916秒，TAEHV解码中位0.230秒。这些仍是带观测hook、stage同步、逐步latent保存的诊断时间，且pipeline包含文本编码；不是官方compile配置的稳态基准，也不能直接除以旧rCM的单DiT时间宣称加速。两进程各自历史allocated峰值之和32.91GiB是保守统计，不是测得的同一时刻总峰值。

## 质量结果与局限

各文本先平均两seed，再八文本等权。MJ实际8帧为0/10/20/30/40/50/60/70；AMT用41偶数帧预测40奇数帧，RAFT读41帧/40个flow并按原round公式判动态，DINO用全81帧/80项。沿用原模型及公式，未把77帧分母硬套到新视频，也未筛除失败样例。

| 描述性均值 | rCM BF16 | rCM plain NVFP4 | rCM旧SVD | rCM QAD64 | 官方FastWan-QAD |
|---|---:|---:|---:|---:|---:|
| MJ总分 | 0.18077 | 0.07106 | 0.10172 | 0.07193 | 0.32240 |
| 文本对齐 | 0.15886 | 0.03751 | 0.06543 | 0.02881 | 0.30739 |
| 细节 | −0.08195 | −0.07734 | −0.07249 | −0.10547 | 0.09762 |
| 连贯性 | 0.44806 | 0.43732 | 0.44904 | 0.43689 | 0.57062 |
| AMT平滑度 | 0.98365 | 0.98394 | 0.98232 | 0.98398 | 0.99442 |
| RAFT动态通过数 | 9/16 | 6/16 | 10/16 | 11/16 | 2/16 |
| DINO主体一致性 | 0.91226 | 0.92650 | 0.90684 | 0.88815 | 0.97902 |

root看过全部8题r0的固定0/20/40/60/80帧：火星日出和阅读人物有可辨认主体，未见旧rCM那样的大面积竖条；但航天器结构奇怪，“科学馆”接近彩色瓶子，“港口”接近画笔水杯，Iron Man接近穿披风的人物。高平滑度/一致性不能替代内容正确或动作完成，也不能从少数抽帧声称完整观看运动。两个seed差异明显，例如球场MJ为+0.61616/−1.00510。原数据与逐文本汇总见[独立评价汇总](/home/wjq/workspace/svdquant-exp/results/research/E024/evaluation_summary.json)。

相对rCM plain，MJ在6/8文本提高；相对rCM BF16仅4/8提高，球场一题的差值+0.87414对总体影响明显。RAFT仅航天器两seed通过动态阈值。独立CPU已复算全部AMT/RAFT/DINO原始读出和MJ分层均值；MJ总分本身是奖励模型输出，不冒称能由28项标准直接相加重构。

[前四题固定帧](/data1/models/svdquant-wjq/research/20261003/E024/contact_sheets/suite_first4_r0.png)、[后四题固定帧](/data1/models/svdquant-wjq/research/20261003/E024/contact_sheets/suite_last4_r0.png)。全部8题是局部测试，且当前有多项配方差异；不由该表推断FP4比BF16更好或量化导致静止。

## 当前决策

外部产品已在本机建立真实运行参照；下一步首先澄清其采样合同，而不是立即围绕以上现象设计新loss。公开模型卡当前命令走上述UniPC，但同版本QAD训练文档列DMD rollout `[1000,757,522]`；正在核查权重发布时的官方入口/历史，确认是否存在明确的接线差异。任何后续采样对照都须独立标明，不覆盖本轮、不换样例，也不把更换scheduler当量化创新。

尚未运行完整Wan VAE控制或关闭观测后的compile性能实验，故当前无法拆分decoder、蒸馏、量化和编译的贡献。E023普通W-global训练仍未启动；运动收缩机制暂未获得同权重因果证据。顶会目标继续，当前没有足以投稿的新核心贡献。

必要的失败记录均保留：下载初期因tmux未继承代理而0字节失败，显式继承后完成；内核smoke首轮PATH缺ninja，修正后实际前向成功，但末尾profiler字段报告出错，未重复GPU前向掩盖它。正式模型smoke、suite和评价均正常完成。来源和复现入口见[E024计划](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E024_official_fastwan_plan.md)、[suite原始记录](/home/wjq/workspace/svdquant-exp/results/research/E024/suite_run.json)。

**同日续跑补充：** [采样发布历史核查](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E024_official_sampling_contract.md)已结束。首发QAD入口与registry同样走三步FlowUniPC/shift3；训练DMD scenario未找到与这份公开权重的绑定证据。保留当前配置，不盲试DMD。下一项仅为E025：固定全部16 final latents的完整Wan VAE FP32控制，零DiT，判断decoder对已有失败解释的影响。
