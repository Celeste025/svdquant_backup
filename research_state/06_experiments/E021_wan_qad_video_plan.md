# E021：主权重QAD的首组完整视频对照

2026-10-03。候选prompt在64步训练结果出现前已由E020库存固定：vbench_066/091/182/067；不属于E020训练/开发24缓存文本。固定seed20261004（历史其他prompt用过同整数，不声称全局未用），每prompt一次实际initial latent/逐步noise/embedding保存，四臂加载相同tensor，独立自由递推。manifest草案一旦执行即作为固定协议使用，不因后缀draft允许事后修改。

四臂：原BF16、plain原W NVFP4、固定64更新QAD、现有SVD nativefast。最后一步开发NMSE回升已经知道，仍执行原定64更新模型；本轮不增挑选更好的32步模型。SVD比较是整套部署配方，不把其他INT4 norm/smooth差异都归LR。普通QAD不是新方法；本轮用于检查训练改善是否能兑现为真实输出，四prompt一seed不是最终泛化基准。

原rCM四步、sigma80、guidance0、480×832/77帧/16fps、BF16 FLASH attention和相同VAE tiled(core128/halo0)。共16视频、64DiT、14,400个真实FP4 GEMM。完整保存final latents、媒体、实际step/算子计数。生成GPU0（启动前现查空闲），30分钟/60GiB；数据/data1/models/svdquant-wjq/research/20261003/E021，日志results/logs/E021_generation.log。GPU5可同时做独立部署计时，其他用户卡不干预。

随后评价时固定比较所有16视频，使用本地已可用视频评分并审视时间维度的限制；原生Tensor NMSE和稀疏抽帧奖励仅为辅助，不能自动等同视频质量。普通相邻帧MAE只适用于静态flicker，不将本轮含运动的候选套该指标当运动质量。暂未确定/运行新评价器，不先声称质量提升。
