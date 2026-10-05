# Claim Ledger

2026-10-04 / D090：E065/E065b必要基线对照已完成并独立核验。carry在200层/1600校准格局部改善，但四点video整模SSE均更差；仅证实局部目标与整模读出反转，不能归因分布差异/传播/模态竞争或质量。停止扩当前固定配方迭代，不分配claim ID；不排除完整官方PTQ。ARHQ另直接覆盖激活残差协方差加权低秩保护，不能把native/H3迁移当新方法。见[报告060](../reports/060_20261004_h3_carry_q_results.md)与[问题筛查](../02_problems/h3_post_carry_candidate_pending_20261004.md)。无active新方法claim。

| ID | Problem | Status | Evidence / next decision |
|---|---|---|---|
| C001 | P001 | rejected in tested setting | 实际CFG6去bias相关0.102/0.184；完整W4A4强持久方向不成立 |
| C002 | P001 | rejected for current route | E002触发预设停止阈值；SR近邻与双权重内存风险 |
| C003 | P002 | parked | scale指数吸收反例；没有实际batch质量失效 |
| C004 | P003 | current route stopped | E003受限现象存在；E004跨prompt取舍方向不稳定，简单目标未过预设收益门槛；不继续调参 |
| C005 | P004 | current route stopped | E006 12case：交互31.71%但组合/加和0.9608且0/12净放大；oracle固定尺度仅改善6.46%<10%；按预设停止，不做continuation |
| C006 | P005 输出通信与SVD投影 | independent paper route stopped; engineering retained | E039−22.85%延迟；E040 side/decoded最终NMSE5.93e−8/1.83e−6，确认已知LR信息合同，不建立新机制/视频必要性 |
| C007 | P011 PV双侧差分编码 | current representation stopped (E080) | 真实H3局部差分SSE改善但输V中心化/未超同预算shuffle门槛，总SSE增加；未进入视频，无细节恢复结论 |
| C008 | P011 运动输运下误差结构 | proposed, untested | 本轮未执行，新颖性未验证 |

未立项候选（不分配正式claim ID）：2026-10-02对“以continuity/Stein残差代替速度L2”的目标重定义做了查新与数学核查。Gauge自由度、候选残差及训练目标由ICLR2024 gauge与FDM直接覆盖，DMD/LongLive已提供不绑定seed的分布匹配及NVFP4应用；普通/STE Jacobian还漏掉实际量化边界通量。朴素目标KILL，宽泛分布校准PARK，无GPU实验。见[碰撞报告](../01_literature/continuity_calibration_collision.md)。当前没有active claim。

2026-10-02新增两项未立项筛选，均无GPU实验：
- [量化×高阶/自适应求解器](../01_literature/solver_quantization_collision.md)：PARK。SAQ/QuAKE及低精度RK直接覆盖宽泛机制；本地H3是固定Euler，无controller，BF16本身也有数值台阶。
- [最小条件编辑的响应](../01_literature/conditional_response_collision.md)：新的差分损失KILL，方向PARK。DASH/GAMP与关系/Jacobian匹配已覆盖核心代数；CFG1语义选择性衰减尚未观察到。后续E012已做一次有界诊断，结果见下。

E012补充：预注册局部诊断已完成26次BF16，方向差分SNR1.032/.642未过teacher稳定性门槛，无合格control，因此native未执行。该具体代理停止，不是量化机制阴性，不注册新claim；见[阶段011](../reports/011_20261002_conditional_response_stop.md)。

E013补充：首对真实50步BF16行为窗口已完成，间距guard导致两例unknown，第二seed/native未执行；只说明本次读出前提不足。接近动作可见，不能否认teacher能力。不新增claim，不扩大toy提示。P002另经有限查新：一般batch-invariance、per-request scale隔离、grouped GEMM/行epilogue新贡献KILL；真实服务残余仍缺需求及损伤/成本证据，C003继续park。见[阶段012](../reports/012_20261002_behavior_readiness_stop.md)、[批独立性碰撞](../01_literature/batch_invariance_collision.md)。

C006范围记录：[paper-unit gate](../05_scope/paper_unit_gate_C006.md)，不据单层阳性扩大完整SP。

2026-10-04未立项候选F1（SwiGLU gate/up乘性交互修正）经E062停止：两teacher×3预选层，同输入真实native fc1后读出的交互净误差全为轻微抵消，去中心/比例及BF16算术对照一致；不支持有害二阶分量动机，不启动adapter。不是所有非线性量化方案阴性。SPEAR/NA-LoRA近邻边界及完整结果见[报告056](../reports/056_20261004_h3_swiglu_interaction.md)。当时QK仅备选，后续结果如下。

2026-10-04未立项候选F2经E063停止：同smooth/shared activationpacket的fresh rank0与legacyrank32对照，六格raw QK改善0.454%–1.383%，主要切向；径向占比四负，两正仅0.373%/2.853%，实际norm后同量级改善。三预选层均未过必要条件，独立CPU复核通过；不追加attention干预或训练，不将有限阴性外推全部QK校准。QuantMLA/HeadQ/SlimDiff等已覆盖宽泛功能目标/联合低秩，完整结果见[报告057](../reports/057_20261004_h3_qknorm_radial.md)。F1/F2均停止，无active新方法claim。

2026-10-04 P009/E064未立项来源筛选完成：实际plain/SVD路径四角，16非零深度video输入依赖项净作用均负，去均值后仍负；稳定净有害放大候选停止。独立104PT/full路径/四角统计核验通过。不能把净负换称quantized map更contractive，因其有限输入响应能量反而更大；不归因A4，不推视频质量。CLQ/跨层联合补偿等已覆盖一般深度累积/有限差分，见[报告058](../reports/058_20261004_h3_depth_four_corner.md)。未注册新claim，无active方法线。


2026-10-04 E066/D091边界更新：H3四teacher输入的800个层×状态video局部SSE均carry更好（M512/full-M同样本/全有效行一致），整模video四点仍更差。局部排序迁移与覆盖控制通过，未识别具体因果机制；不分配新方法claim。固定非负逐层SSE重加权不能翻转两个已有候选，但不否定第三候选的重新训练。工程修复、目标反转观察、H3复现均不单独算创新；报告061。

2026-10-04 D097/报告067：用户要求压缩工程投入。E072a未执行停止；RoPE融合、AdaLN解析shift、Jensen偏置通用方案均因直接近邻未立项。弱运动问题还缺可辨、可干预证据，E014含噪velocity不是合格筛查代理；无新active claim，不伪造signal或实验阴性。


2026-10-04：C007(P011) / C008(P011) 为 proposed, untested 候选，非已验证或已启动方法。C007有限查新、C008新颖性未核实；均0新增GPU，详报告078。导数匹配不另立claim，2ndMatch等已覆盖。

2026-10-04 C007补充：DIDB-ViT也直接保护attention差分，动机重叠强于初筛；仅PV双侧量化表示的窄残余待验证。VC的FP4 V-Smooth与Sage3算法兼容但本地未接入，未确认公开官方kernel；不替换冻结基线。见报告079。


2026-10-04 D111/E080：C007当前表示按预设门槛停止，不将此前proposed状态当active。局部结果与独立复核见[报告080](../reports/080_20261004_pv_contrast_results.md)。差分SSE降低不等于幅值或画质恢复；不扩大group/层/步/head扫描，不以V中心化已知收益认领创新。C008保持未执行。


2026-10-05 D113/D114：V中心化作为已有方法完成E081机制拆分与E082原生/两clap视频检验，局部收益成立但尚无明确细节修复，不新增claim或复活C007。详报告081。


2026-10-05 D115/E083：BF16主干×Sage3/center128两seed4新视频完成，局部阳性未转化为明确稳定细节收益；不能只用SVD主干解释此前无收益。仅已有方法效果验证，不新增创新claim，不复活C007；见报告082。
