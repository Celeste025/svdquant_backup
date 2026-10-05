# 候选主张，均未证实

C001（P001，mechanism）：在普通多步Wan的NVFP4 W4A4中，移除每通道bias后误差仍有跨步稳定方向，且固定权重路径解释主要部分。
预测：连续6步、2个prompt、conditional及CFG结果中，中心化相邻误差cosine明显为正；W4A16也有同类结构。
反证：中心化后相关消失、prompt间不稳定、主要由已知输出比例偏差解释。
状态：active exploratory；teacher-state相关不是自由轨迹损伤或新颖性证据。

C002（P001，method/measurement）：如果C001成立，按solver加权的互补误差干预能在不增NFEs的条件下减少闭环偏移，收益不能由简单bias correction解释。
预测：保持单步误差能量近似相同、减少跨步相关即可改善heldout endpoint和视频；契约内真实kernel仍保留收益。
反证：仅oracle有效、扰动相关改善却endpoint不改善、收益来自额外计算或与已有方案等价。
状态：parked，须C001、prior-work collision及closed-loop门槛后才实现。

C003（P002，systems）：某实用动态batch实现的global scale统计域导致可复现质量依赖，且有维持batch吞吐的修正。
反证：合法幅度区间影响轻微、现成per-request实现已解决，或只是GEMM不确定性。
状态：parked；当前证据仅格式相位变化，paper scope过薄。

C004（P003，mechanism）：H3 block0总误差被text主导，但其实际video误差pulse产生的最终video偏差比text误差pulse更大，局部模态能量排序不能指导校准。
预测：text-only pulse的局部误差大而最终video误差小；video-only pulse反之；最终plain/SVD排序与local不同。
反证：text-only也主导最终video，或仅一个prompt/step发生；即使成立仍不证明modality reweighting新颖。
状态：E003在受限诊断内支持；各pulse幅度不同，不能推导内在敏感性。须跨block/heldout及自由rollout后才能提升证据等级；下一步E004简单已知方法对照。

更新D004：C001完整W4A4主导持久相关的强版本被E002 CFG对照否决；C002当前路线停止。上面的初始预测保留作预注册记录。

C005（P004，mechanism，2026-10-02预注册）：原生W4A4 qkv projection与原生FP4 attention的误差存在有害非加性交互，且QKV输入尺度切换是可干预的主要原因。
预测/反证：按E006冻结的12case video主指标门槛；固定teacher尺度只有oracle诊断作用，不可部署方法或创新结论。低交互、抵消交互、固定scale无效或BF16 continuation收益不保留任一项即停止当前路线。
状态：native前置门槛通过，真实factorial尚未运行。

更新D007：C004在p1的受限现象保留，但E004的跨prompt排序相反、简单对照收益未过预设门槛。当前模态校准路线停止，不把固定A/fc2-B阴性推导成一般方法无效。


## C007 / P011 — PV双侧差分编码（2026-10-04，未证实）

Main claim候选：在H3精细结构案例中，独立P舍入丢失相对概率差异是可见对比损伤的可干预来源；同bit/scale预算双侧差分编码可超过V-only中心化。范围为固定SVDQuant+官方Sage3及真实同输入PV；prediction、disproof、强对照和部署限制详报告078。若普通更细scale、V平滑或随机pair解释收益则停止机制新颖主张；若只降局部MSE不救视频则不推进。Move为mechanism/method；fast uncertainty为P量化前后对比是否丢失且影响可见细节。状态proposed, untested；novelty partially verified，中高碰撞风险。

更新D111/E080：C007当前表示停止，预设必要优势未通过。block24差分误差降约13%–14%，但同预算shuffle已解释大部分改善、V中心化更好、总输出误差增加；无视频验证或细节恢复。见[报告080](../reports/080_20261004_pv_contrast_results.md)。上文保留原预测，不代表当前active状态。

## C008 / P011 — 运动输运下的量化误差一致性（2026-10-04，未证实）

Main claim候选：匹配逐帧误差能量/偏差/分布后，误差沿真实运动对应的结构比固定屏幕对应更决定边缘重影；可用耦合舍入而非信号平滑缓解。范围H3视频帧轴，非去噪步缓存。预测是正确对应干预改善轮廓而不减动作幅度；若仅减少闪烁/运动或全由低MSE解释，则不能支持。Move为mechanism→method；fast uncertainty为跨帧误差如何转成单帧模糊。状态proposed, untested；novelty unverified，普通temporal loss已有Q-VDiT覆盖。详报告078。
