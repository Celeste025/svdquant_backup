# E083：BF16主干隔离attention方法的完整视频比较

2026-10-05，用户建议并授权的必要对照；exploration/已有方法效果判别。问题：E082在SVD主干上未见明显细节收益，是否受上游量化/不同轨迹影响？在BF16主干上，V-center128相对原Sage3是否更接近完整BF16视频、是否修复可见手部重影？不预设SVD掩盖收益，不把BF16阴性外推全部attention方案。

新生成四条：两clap seed×{bf16_sage3,bf16_sage3_center128}。完整BF16主干+原SDPA参照复用E073两条，真实初始noise、文本embedding、prompt、scheduler、20步、CFG1、1024×576/124帧、shift12/3、原video/audio VAE严格相同。全50 main attention采用对应kernel；refiner/padding维持原BF16路线。200主干线性层必须原torch BF16模块，完全不安装SVDQuant；每DiT要求scaled_mm=0、SDPA=52、native attention=50、disk_loads=0。BF16权重/激活在线性层原路径；Sage3自身QK/PV仍NVFP4，不能叫attention BF16。

复用已验收E082原生center128及binary，不修改kernel/group/mask或计作创新。官方Sage3及私有源SHA和E082验收绑定；原pipeline源与E073/E079合同冻结。E083只切换主干精度，不能把原SVD捕获中的40%局部PV改善直接外推到新的BF16主干。

必要核验：CPU prepare绑定与源hash；记录200线性层weight/bias BF16/type/cuda，确认无native replacement；运行每步counter/有限性/最终latent与完整VAE媒体合同。每arm相同真实noise/embedding/setting；不要求不同arm输出相同。视频评价前重hash媒体。

读出：三路并排BF16/ BF16+Sage3/ BF16+center128；每clip全部124帧LPIPS-Alex/MAE/RMSE对完整BF16，另center与Sage直接距离。固定32抽帧/clip观察手指可辨性、重复轮廓、动作存在与新增结构缺陷；未知保留，不据构图变化或距离单项宣布画质优劣。root全8页，独立agent在未读指标时看seed1四页。原E082 SVD组作为跨精度对照，不重生成。

决策：若BF16主干两seed均明确细节改善且无明显新增缺陷，才支持有限范围的精度依赖收益，后续另定更广案例验证；若同样不改善，不能继续用SVD混杂解释这两例结果；若不一致则如实报告，不扫层/步/group或新增kernel。视频C−B是完整生成效果，随轨迹改变QKV，不能作为固定P/V机制贡献；不由此认领SVD有害交互。

资源：GPU0/1各运行该seed两个arm，每arm20DiT，总80DiT、4video+4audio VAE；每GPU共享1800秒、60GiB。数据5GiB。0训练/校准/新kernel/重生成BF16参考。命名tmux，保存失败，不改已执行旧源。此为用户要求对照，不需新授权。

创新与口径：当前连续块去均值+online mass恢复是VC-Attention V-Smooth中的已知组成，不含其聚类重排，不当作完整VC复现。40.24/42.07%来自E081 s14/b24 g32、原SVD轨迹72queries/all56heads，固定量化QK下PV SSE=(O−softmax(Sq)V)^2之和的相对下降；不含QK误差/主干误差/视频误差。E082实际g128，局部PV改善37%–40%，计入QK的原生attention改善14%–15%。新来源复核：https://arxiv.org/html/2609.15810v1 §3.2。
