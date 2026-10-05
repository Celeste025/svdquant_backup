# E024：官方FastWan-QAD完整产品基线

2026-10-03；环境/固定资产准备已启动，尚无本机模型前向或视频质量结果。目标是建立可实际使用的外部强参照，再决定研究切口；官方已有方法不算本项目贡献。

固定FastVideo `8444c0897a8b96848eb85b6e5750ef486f79fc92`、HF FastWan-QAD-1.3B `621c6aeb900f9f9a2ebb9ea9ed74c0daf31d5e6a`、TAEHV `011dfc2112197741c540e0bdd5b7b67bcc930771`。使用独立torch2.12/cu130环境和官方SM120推理kernel；主矩阵NVFP4、self-attention为ATTN_QAT_INFER，cross-attention按官方实现保持dense。复用TE/VAE/tokenizer须核对公开文件SHA，原模型及旧环境不改。

第一阶段直接复用官方 `FastWan_QAD_TAEHV.py` 的build_generator和TaehvDecoder，`model`指固定完整本地资产、`distilled_model=''`，TAEHV FP16、compile关闭、CFG1、三步、81帧、480×832、16fps。官方当前入口实际为FlowUniPC/shift3；同commit训练说明却列DMD rollout `[1000,757,522]`。保留这一接线差异，记录实际scheduler/timesteps，不暗中切成另一个FastWan2.1 preset。如果后来需要显式DMD对照，另列配置与结果，不能覆盖本轮或称唯一官方设置。

功能smoke固定E022测试表第一条 `vbench_128_r0`（原文航天飞机，seed20261110），不依据画面挑例。单条成功后正式suite使用原8prompt×2seed全部16项；重复smoke样例也重新生成，以统一模型驻留过程。相同seed仅是文本/种子层面的配对，81帧、不同模型与scheduler使实际噪声/轨迹不等同rCM77帧；不能算同权重量化因果对照。

薄runner仅记录观测，不改官方计算：保存final与官方返回的逐步latent/实际timesteps、模型与scheduler类型、实际NVFP4/self-attention/cross-attention执行数、每组件耗时、峰值allocated/reserved及完整MP4。启用stage logging会同步，首次JIT/冷加载也单列；此阶段是功能与质量基线，不能把一次冷启动时间当稳定性能。预期每视频3 DiT、900主矩阵FP4 GEMM、90 FP4 self-attention和90 dense cross-attention，实际不符先检查路径，不能用预期数代替测量。无需跨实现byte等价。

模型执行前现查GPU0空闲，tmux+log、单阶段20分钟/60GiB上限；环境kernel smoke由systems仅用GPU1另行记录。下载/安装/GPU失败均保留来源与错误，限定修复，不更换算法掩盖失败。

后续仍按每prompt两seed→八prompt等权报告MJ总分/对齐/细节/连贯性及AMT/RAFT/DINO，各指标分开。TAEHV输出比旧rCM多4帧，完整Wan VAE解码控制与独立驻留性能测量在功能验证后另定；不将Tiny decoder或少一步带来的全部加速算成量化收益。不按BF16质量或本模型成绩排除原8个文本。

依据：[版本与资产清单](/home/wjq/workspace/svdquant-exp/research_state/00_state/fastwan_qad_external_baseline_readiness.md)、[入口实施备注](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E024_runner_notes.md)、[E022固定文本表](/home/wjq/workspace/svdquant-exp/results/research/E022/video_test_manifest.json)。
