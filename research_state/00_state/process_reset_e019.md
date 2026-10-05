# E019 后流程复盘：先建立可训练、无在线低秩分支的 NVFP4 强基线

2026-10-03。独立复盘；仅阅读本地材料，未搜索、运行 GPU 或修改旧实验。本记录确定下一阶段主线，不声称已有新方法。

## 判断：已经形成了低信息增益的循环

**是：近期反复走入“寻找现成方法的边界 → 做一次小审计 → 不能立新 claim → park”。** 正确性工作曾经必要：旧 H3 低秩输入确有错误，真实 native 路径也需要贯通。但在合同已经建立后，继续把完整新颖性、干净 teacher 行为、局部效果门槛依次作为实做前提，会让研究长期停留在边界反例，无法积累可训练模型和稳定的质量—成本曲线。我此前因 QAD 已有先前工作而过度保守地推迟实施强基线，也重复了这个错误。[先前流程复盘](/home/wjq/workspace/svdquant-exp/research_state/00_state/phase4_process_critique.md) 已明确指出这一点，本轮需要真正落实。

近五份报告中，最能改变决策的是完整基线和实际生成：E014 的 plain native 比现有 SVD native 快 14.4%，局部误差排序并不统一；E016 直接确定了完整 FP4 attention 的成本；E017 的 p36 中，block 的局部及最终 latent 更接近 BF16，四个运动/稳定/细节题却从 yes 变为 no。最后一项只是一例、评价仅看六帧，不能推出普遍机制，却足以否定“继续优化局部 NMSE 就能可靠选择部署方案”。E015 的局部跨模态恢复和 E018 的分组相位干预则没有产生可以据此持续改进模型的证据。这不证明这些现象不存在，而是说明继续追加同级审计的机会成本已经很高。[报告013](/home/wjq/workspace/svdquant-exp/research_state/reports/013_20261002_plain_native_tradeoff.md)、[014](/home/wjq/workspace/svdquant-exp/research_state/reports/014_20261002_crossmodal_propagation.md)、[015](/home/wjq/workspace/svdquant-exp/research_state/reports/015_20261002_full_fp4_attention.md)、[016](/home/wjq/workspace/svdquant-exp/research_state/reports/016_20261002_fp4_attention_video.md)、[017](/home/wjq/workspace/svdquant-exp/research_state/reports/017_20261002_query_group_phase.md)。

本轮 root 通报 E019 为 12 次 attention 配套冻结 88 文件和多重审计：流程负担已明显失衡。跨 `ReturnLSE` 模板的 byte gate 阻断，**可能**只是不同编译路径的舍入，当前不能写成已查明原因。不同数学等价实现不自动承诺逐字节相等；应核实实际合同、配套数值控制与差异量级，而非把 byte 不同直接升级为研究停止。原实验的失败与冻结记录保留，不追改。

## 唯一下一步：成熟 QAD 的真实部署基线

**研究问题：在固定、无在线低秩分支的原生 NVFP4 部署图下，学习主权重能否持续改善独立完整视频的质量—成本取舍；若训练改善不能兑现，失败发生在训练目标、量化导出还是自由生成泛化的哪一环？**

先在 **rCM-Wan2.1-T2V-1.3B** 上实施普通 quantization-aware distillation（QAD）：被量化线性层的主权重可学习，训练前向采用目标 NVFP4 配方的 QDQ/STE；采用成熟 teacher–student 蒸馏目标，导出后一次性 pack 主权重，每个线性层使用单条 native 主路径，无在线 LR。固定合法 rCM 四步采样、BF16 attention 和现有 VAE。以完整原始 BF16 权重初始化，**不能直接删除现有 SVD checkpoint 的 LR 分支后把残差权重当完整权重**。

普通 QAD、STE、无在线分支或 NVFP4 本身均不是新方法。这一实施不再以“先拥有完整 novel claim”为前提，也不同时改步数、attention、精度路由或发明新损失。后续创新只从这条主线上可复现的训练失败或实际质量—成本约束中产生；做出强基线也不自动等于有论文。

选 Wan 有具体依据：现有 E007 的 BF16/native 中位数为 1.770/1.975 秒，主 GEMM 节省被 LR、packing、smooth 抵消。这不是最优融合实现的下界，但足以支持实际测量“去掉在线分支并训练主权重”这一完整方案。小模型已有合法四步生成和 native 执行入口，六张可空闲的 72GB SM120 可以承担实际训练、独立生成评估及重复验证；具体并行和优化器显存仍须实测。H3 的价值先是给出局部误差代理失灵的警示与后续外部验证入口，当前不扩成 H3 大模型训练工程。[当前状态](/home/wjq/workspace/svdquant-exp/research_state/00_state/current_state.md)。

这比再做一轮 operator 边界审计的信息增益高：它同时回答现有分支成本是否能被训练取代、训练收益能否通过真实导出保留、以及局部拟合是否在独立自由生成中兑现。即便失败，也在同一系统中留下可连续研究的学习曲线和可定位的落差，而非再增一个不相连的边界反例。

## 本地可复用入口及缺口

| 入口 | 已确认用途与限制 |
|---|---|
| [真实 rCM 配置](/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer/config.json)；[native 模型路径](/home/wjq/workspace/svdquant-exp/scripts/research/bench_wan_native_nvfp4.py:25) | 目标确认为本地 rCM-Wan2.1-T2V-1.3B：30 层、12 heads、head dimension 128。BASE 为 `/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers`，transformer 用 rCM 权重；不是普通 Wan 的长步采样。旧脚本 `/data/models/...` 路径不能直接沿用。 |
| [现有全模型训练](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_fullmodel_lowrank_denoiser.py:190)；[on-policy 训练](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_onpolicy_rollout_distill.py:165) | 可借 teacher/student 调用、gradient checkpointing 和学生轨迹取样；两者均先 `freeze_all` 再 `branch_parameters`，**仅训练低秩分支，不是主权重 QAD**。on-policy 的“fullmodel”不能解释成已训练全部主权重；历史 retry 因 CSV 列不匹配失败，不能算完成训练。 |
| [activation STE](/home/wjq/workspace/svdquant-exp/scripts/exp_rcm_blockwise_lowrank_pilot.py:114)；[plain 配置](/home/wjq/workspace/svdquant-exp/third_party/deepcompressor/examples/diffusion/configs/svdquant/rcm_wan_pure_nvfp4.yaml:1) | 前者可复用量化 forward/STE 思路；后者已设 `enable_smooth: false`、`rank: 0`。仍需新建主权重可学习的训练与导出连接，不能把配置存在当成已有完整训练器。 |
| [native adapter](/home/wjq/workspace/svdquant-exp/scripts/research/wan_native_nvfp4.py)；[fast packer](/home/wjq/workspace/svdquant-exp/scripts/research/wan_nvfp4_fastpack.py) | 可借 packet、scale、真实 GEMM 与独立 decode 合同。现有 adapter 为 SVD 残差图服务，且为 inference 路径，不是可直接反传的 QAD 实现。新版本保留旧文件。 |
| [合法四步采样](/home/wjq/workspace/svdquant-exp/scripts/infer_rcm_wan_4step.py)；[native 配对生成](/home/wjq/workspace/svdquant-exp/scripts/research/run_wan_native_paired_video.py) | 已有 rCM 时间表、共享初始 latent/噪声与完整 VAE 解码。复用采样语义和可用组件，不把旧实验全部多级 gate 复制进每次训练评估。 |

旧 LR-only 训练确有训练损失下降和少量视频指标改善，必须作为已有结果记录；它未回答上述无在线 LR 主权重训练问题。[旧实验审计](/home/wjq/workspace/svdquant-exp/research_state/00_state/legacy_audit.md:23)。

## 成功与失败都可连续推进，但简化流程

先建立一次实际训练—导出—native 生成闭环，再按固定计算预算取得学习曲线。只保留 BF16 teacher、冻结 plain、现有冻结 SVD 和训练后 plain 四个必要参照；prompt/seed 划分训练、验证与最终测试，已看过的 p30/p36 不再算独立测试。主读出是导出后的完整视频质量及真实延迟、峰值显存，质量评价应覆盖时间过程，六帧 VisionReward 和局部/最终 NMSE 只作为辅助解释。展示样本间取舍和不确定性，不设任意 10% 或 teacher 差分 SNR 准入线。

每版只保留必要的输入/采样合同、目标格式与 code/scale 语义、真实 native 执行及梯度/导出检查，以及单份代码/数据/checkpoint manifest 和失败日志。已验证且未变化的部件不重复多重独立审计；只有有理由期待同实现精确重放或同 packet 身份时才要求 byte exact。主权重无梯度、导出配方错或 hidden fallback 必须修复；“暂未证明强 novelty”不阻止训练和实际生成。

若训练与 native 独立质量同步改善，形成真实成本—质量曲线，再研究剩余失效条件；若训练损失持续下降而导出或独立自由生成不改善，就用同一批 checkpoint 定位落差，不能立即改名为新机制，也不因一个小审计阴性又切换 operator 题目。本轮仅确认这条持续主线与入口，未实现训练器。


root结果补记：两次隔离诊断现已完成，False完整复现E018，True仅39/161,559,552元素变化，NMSE2.84546e-12；跨模板存在极小数值差异已确认，具体机器指令原因未定位。原分片对照未执行，不作为机制阴性。独立实现检查进一步确认模型精确参数为1,418,996,800，详见[就绪度](implementation_readiness_wan_mainweight_qad.md)。
