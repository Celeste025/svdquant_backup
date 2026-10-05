# 视频生成模型 NVFP4 W4A4 量化：中期报告

2026-10-04。主实验为 MiniMax-H3，Wan 仅保留前期验证结果。

**总体判断：已有可复核的数值观察和真实部署收益，但尚未形成论文级方法创新，也没有独立人评验证的等质量加速。** 下面按我判断的创新潜力排序；排名靠前表示更值得保留研究线索，不表示已证明新颖或实验成功。

| 排序／尝试 | 动机／观察 | 实验 | 结果／结论与创新判断 |
|---|---|---|---|
| **1．误差的组织方式与局部校准目标失配** | 每层都更准确，完整模型是否一定更准确？平均误差是否遗漏了重要结构？ | H3 全部 200 层比较两种低秩校准迭代；在四个相同输入上检查全部 token，另做误差方向分析。 | **800 个层×状态的局部视频误差全部下降，各状态层中位下降约 3%；完整 DiT 视频预测误差反而增加 0.47%–12.93%。** 已排除这些输入上的简单采样、矩阵形状假象。这是目前最扎实的研究观察；一般局部／整体目标失配并不新颖，具体致害机制和画质后果仍未成立。[060](/home/wjq/workspace/svdquant-exp/research_state/reports/060_20261004_h3_carry_q_results.md)、[061](/home/wjq/workspace/svdquant-exp/research_state/reports/061_20261004_h3_local_transfer.md) |
| **2．压缩 FP4 attention 的均值校正表示** | 分块均值校正较准确，但构造和消费有成本；能否以少数中心保留其收益？ | 比较 rank-16 中心、跨文本固定基、在线聚类、简单连续粗分组；最终生成三配置共 24 条视频。 | 同样本 rank-16 保留 **91.7%–96.7% 的局部误差改善量**，跨文本降至 62.0%–75.6%；简单粗分组已取得聚类改善量的 93.1%–97.3%。**未获得稳定视频优势，已停止扩展。** 这是较具体的方法尝试，但低秩代数本身不新，现有证据不足以支持复杂中心方案。[027](/home/wjq/workspace/svdquant-exp/research_state/reports/027_20261003_restricted_query_centers.md)、[030](/home/wjq/workspace/svdquant-exp/research_state/reports/030_20261003_coarse_center_control.md)、[034](/home/wjq/workspace/svdquant-exp/research_state/reports/034_20261003_center_video_results.md) |
| **3．利用跨去噪步骤的误差相关性** | 如果相邻步骤误差方向持续一致，能否设计互补量化，减少累积？ | 两个 H3 窗口，分别开关第一步／第二步量化，并分离采样更新舍入影响。 | 去均值及输出比例后，误差余弦仍为 **0.701／0.622**；但输入变化会显著改变下一步误差。去掉第二步更新舍入后，交互净作用为 **+0.17%／−6.17%**，并非稳定致害。**停止直接推进固定误差线性抵消；相关性不是可直接利用的新机制。** [051](/home/wjq/workspace/svdquant-exp/research_state/reports/051_20261004_h3_temporal_diagnostic.md)、[052](/home/wjq/workspace/svdquant-exp/research_state/reports/052_20261004_h3_update_arithmetic.md) |
| **4．按下游计算需要组织跨卡通信** | SVD 投影需要 FP4 主支输入及高精度低秩信息，是否必须传完整 BF16 激活？ | 双卡比较 BF16 回传、FP8 回传、源端投影，以及 FP4 数据包加 32 维低秩部分积。 | 单个通信＋投影边界 **6.348→4.897 ms，减少 22.85%**。有实际系统收益，但缺成熟融合 FP8 的公平比较和整模型收益；信息分解与代数重排也不足以构成新意。**保留工程结果，独立论文路线已关闭。** [035](/home/wjq/workspace/svdquant-exp/research_state/reports/035_20261003_output_projection_boundary.md) |
| **5．量化与模态、稀疏、网络结构的交互** | 量化是否造成可利用的模态失衡、稀疏预算膨胀，或特定非线性放大？ | 模态重加权／两步传播；三层稀疏路由；SwiGLU、QK 归一化及深度输入漂移对照。 | 模态改善不能稳定迁移；稀疏选块虽改变，预算净增仅 **0.37%–0.90%**；SwiGLU 和已测深度交互净作用均为抵消；QK 收益未主要浪费在被归一化消除的方向。**这些具体假设未获支持，停止相应方案；组合本身不算创新。** [005](/home/wjq/workspace/svdquant-exp/research_state/reports/005_20261002_calibration_stop_native_ready.md)、[014](/home/wjq/workspace/svdquant-exp/research_state/reports/014_20261002_crossmodal_propagation.md)、[031](/home/wjq/workspace/svdquant-exp/research_state/reports/031_20261003_projection_sparse_routing.md)、[056–058入口](/home/wjq/workspace/svdquant-exp/research_state/reports/058_20261004_h3_depth_four_corner.md) |
| **6．原生部署与校准基线** | 旧伪量化不能代表实际速度，需要建立真实成本参照。 | 接入 H3 的 200 个原生 NVFP4 线性层，保留 rank-32 分支；接入现成 FP4 attention，核查校准配方。 | H3 单次完整 DiT **8.291→6.804 秒，约 1.22×**；稳态前向 allocated 峰值 **40.86→16.81 GiB**。另一同轮测试中，FP4 attention 相对 BF16 attention 再快 **1.33–1.37×**。**属于必要工程，不计方法创新；不是完整视频生成的等质量加速。** [013](/home/wjq/workspace/svdquant-exp/research_state/reports/013_20261002_plain_native_tradeoff.md)、[015](/home/wjq/workspace/svdquant-exp/research_state/reports/015_20261002_full_fp4_attention.md) |

**两条值得保留、但尚不能称为创新点的线索。**

- **误差能量近似不变，误差结构可以明显改变。** 改变 attention 的 query 分组相位，三层输出变化的能量相当于原量化误差的 29%–62%，而 NMSE 只变约 ±0.17%。这是比“误差大小不代表质量”更具体的干预现象，但尚未证明导致闪烁或真实帧边界损伤。[017](/home/wjq/workspace/svdquant-exp/research_state/reports/017_20261002_query_group_phase.md)
- **动作可辨性下降，未必等于动作变慢。** 固定主干 W4A4、只改变 attention 精度的拍手样例中，低位画面仍有快速开合，却因手部重影难以计数。因此光流下降不能直接解释成运动减少；目前也没有证据确认“接触后身份混淆”机制。这是问题线索，尚无改进方法。[037](/home/wjq/workspace/svdquant-exp/research_state/reports/037_20261003_clapping_readability.md)、[068](/home/wjq/workspace/svdquant-exp/research_state/reports/068_20261004_h3_readability_hypothesis_screen.md)

**前期 Wan 工作，保留但不再作为主线。** 主权重 QAD 在 native 开发集上降低误差 32.22%，64 条视频未建立稳定改善；原版 Wan 单次末步量化已能出现碎片，恢复 BF16 末步、匹配校准均未充分消除。说明继续压低局部误差或机械扩大校准并不保证解决当前缺陷，不能外推为 H3 的机制。[022](/home/wjq/workspace/svdquant-exp/research_state/reports/022_20261003_expanded_qad_final.md)、[041](/home/wjq/workspace/svdquant-exp/research_state/reports/041_20261003_terminal_intervention.md)、[044](/home/wjq/workspace/svdquant-exp/research_state/reports/044_20261003_matched_calibration_results.md)

**当前证据边界与下一步。** 局部／整模 SSE 均是数值代理，MJ-VIDEO-2B 仅作辅助，模型看帧不等于独立人评。当前 H3 使用旧 smooth＋rank-32 配方，尚未完成官方完整 SVDQuant 强校准复现。E068 的两配方完整视频已生成，匿名抽帧没有辨认出稳定功能差异，人评仍待；不能将第 1 项写成“局部改善导致画质下降”。

新增 E073 的 **8 条完整 BF16 参考视频已生成并解码完成**，与既有主干 W4A4 视频共用文本条件、实际初始噪声和采样设置；尚未完成成对观察。下一步先判断最终视频是否存在稳定、可辨的缺陷，再决定是否值得做机制干预。优先保留第 1 项的校准失配问题及动作可辨性线索；若没有任务层面的证据，就不继续围绕代理指标扩实验，也不恢复 SVD 提速、复杂中心或通信工程支线。

E073 视频位于 [replica 0](/data1/models/svdquant-wjq/research/20261004/E073/decode_full_bf16_r0/) 和 [replica 1](/data1/models/svdquant-wjq/research/20261004/E073/decode_full_bf16_r1/)。本报告整理期间仅核查既有日志和文件，未启动新 GPU 实验。
