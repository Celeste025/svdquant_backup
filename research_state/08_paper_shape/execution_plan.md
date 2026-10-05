# 当前执行计划

2026-10-03。没有已成立的论文主张；按实验结果安排下一步，不预设投稿包装。

| 状态 | 工作 | 决策 |
|---|---|---|
| 已完成 | E001基线修复与backend核对，E002跨步相关 | 修复保留，无质量收益；跨步路线停止 |
| 已完成 | E003模态pulse与E004等容量普通校准强对照 | 单prompt取舍不稳定；简单模态加权路线停止 |
| 已完成 | E005真实H3 NVFP4格式与native GEMM对照 | 原生可用，舍入变体未改被测排序 |
| 已完成 | E006 projection×FP4 attention 12case | 全部净抵消而非放大；固定scale收益不足，停止当前机制 |
| 已完成 | E007完整Wan原生NVFP4、快速packer及三后端profile | 300W与bypass exact、fast/slow SHA相同；native1.975s vs BF161.770s，尚无速度收益；native/QDQ终点NMSE4.28% |
| 已完成 | E008配对自由rollout与实际视频 | BF16四步历史精确复现、native1200GEMM；单样本人物/背景改变，不判质量优劣 |
| 已完成 | E009 H3完整resident native基线 | 200W exact/迁移exact、slow/fast50block SHA exact；BF16/QDQ/native8.260/20.778/6.787s，native峰值allocated16.80GiB；仅固定输入DiT |
| 已完成有限验证 | FlashInfer SM120 fused-up | 四真实case过门槛，长M的QKV组件1.53×；FC2/短M无收益，整模需另验 |
| 已完成 | E010 H3 heldout配对生成及独立审查 | p30/p36共四视频、160步文件和媒体核验；辅助评分p30相同/p36两题较差，两例无泛化，音频未评 |
| not_ready，未执行 | E011稀疏router guard | 近邻核查收窄到guard强制dense；本地无SM120 consumer且H3 tail语义未验证，不投入新后端适配，不当机制阴性 |
| teacher gate停止，独立验证完成 | E012局部条件响应诊断 | 26BF16、两个原点exact，方向差分SNR1.032/.642不足10；无合格control，native未执行；只说明诊断条件不足 |
| 首对teacher门槛停止，独立验证完成 | E013投影行为窗口 | 两例50步BF16/100DiT/10200SDPA完成；间距guard导致unknown，不执行第二seed/native，不扩toy提示 |

| 已完成，独立核验 | E014完整plain NVFP4强基线 | BF16/SVD/plain 8.291/6.804/5.823s；plain较SVD延迟−14.4%，但双模态误差交叉，保留两配方 |
| 已完成并停止扩展，独立核验 | E015实际一步误差的跨模态四角续接 | 18DiT/101文件核验；oracle恢复净作用有正有负、最大局部11.42%，四组排序不变，不立普遍保护规则 |
| 已完成，独立核验 | E016现成FP4 attention整模基线 | 27DiT/116文件核验；BF16/block/global attention6.791/5.124/4.968s，误差与显存有取舍；无生成质量或新贡献主张 |

| 已完成，独立核验 | E017两种FP4 attention完整自由生成 | 80DiT＋4decode＋8视频232题，273文件核验；p36 block四题变差/global同SVD，停止基础网格扩展 |
| 已完成，独立核验 | E018固定真实QKV的query分组相位 | 1DiT＋27attention、142文件核验；误差有相位结构但并非真实帧边界特有，不立视频/方法claim |

可复用基础：8张SM120 GPU中的空闲卡（每次现查）、已有H3/Wan模型和缓存、native torch FP4 API。其他用户任务不干预。大文件/环境/JIT缓存放/data1/models/svdquant-wjq/research，报告及可复用源码进本仓库；不推送远端。

当前分工：root完成因果诊断与报告；systems完成真实状态捕获及probe数值review；audit完成独立CPU时空归约及追加相干向量探索；frontier完成Qmean/layout近邻核查，当前均已收尾。E014–E018已完成报告013–017，全部研究GPU任务退出、无active claim。行为能力门槛只适用于相应能力主张，不作为全部研究前置条件；自然的单卡资源/延迟约束不要求用户再次明确SLO。E018保留可干预的分组相位结构，但伪边界也交替、投影未整体翻号，尚无实际视频后果或现有分组/重排之外的贡献；方向park，不扩GPU网格。下一步用真实QKV/packets低成本核查P4分区合同及现成PoT/tile缩放对照，先验证实际量级，不先开发新kernel/多卡框架。E011仍未执行。E007–E018已完成实验的源码/结果冻结，后续变化新建文件或显式版本。


E019状态更新（取代上文“下一步P4合同”）：原分片队列failed_stop，额外LSE隔离诊断完成，累计3attention/0DiT；39个元素极小漂移已定位，未执行真实分片/数学对照。报告018完成，不扩此方向。

**当前唯一实施主线**：rCM-Wan主权重QAD与无在线LR的native NVFP4部署。先新增薄主W/A-STE模块及packed导出/安装，复用现成teacher/cache/GEMM；固定4步与BF16 attention，再实际训练—导出—独立视频评价。普通QAD是强基线，不是新claim；创新来自该连续主线的真实失败与成本—质量约束。详见00_state/process_reset_e019.md与implementation_readiness_wan_mainweight_qad.md。当前训练尚未启动，全部研究GPU任务退出；不复用旧参数/字节不变gate作为研究准入。

2026-10-03最新状态（取代上述“训练尚未启动”）：E020训练64updates及E02116视频/全部内容运动评价已经完成，见报告019/020。训练集误差下降但native开发上升7.77%，MJ均值由campus异常主导，当前没有普遍质量收益。现在实施E022现场更充分teacher数据+fresh主权重QAD；具体数据/128updates/accum4/预定native开发选择见06_experiments/E022_expanded_qad_plan.md。当前collector/trainer实现中，未启动GPU；继续该主线，不转回算子小审计。普通QAD仍不是新方法；需要实际质量稳定性与成本证据才形成后续研究问题。

E022已实际采集（GPU0/PID1768590）；训练已排入review后监督队列 research-E022-queue，等待采集complete。新协议/source冻结，后续不重置失败deadline或覆盖已有结果。

E022采集已complete 288BF16/545.43秒，GPU0已释放；GPU5训练PID1890988正在初始开发验证。8prompt×2seed的64视频四臂测试协议已经固定，待训练候选实际产生后执行。
