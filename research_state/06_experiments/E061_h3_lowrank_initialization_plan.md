# E061 — 固定smooth的H3低秩初始化目标对照

2026-10-04，exploration / 强基线补强的投入判别。不是新方法，不是完整官方SVDQuant复现；在任何新导出或GPU执行前记录。

## 动机 / 假设

现有H3导出使用legacy smooth+LR配方；当前生成脚本对量化残差重新做随机SVD候选，而官方校准器支持从完整平滑权重开始、跨轮更新量化权重。两者可能把相同rank预算用于不同的方向。若直接吸收权重主方向已经一致降低完整预测误差，后续应先补强基线，避免把legacy配方的残余失真当新机制。

H1：固定现有smooth、rank32、量化器和真实native执行，仅将低秩目标由Q0量化误差改为Ws，能改善同输入完整H3预测。H0/竞争解释：误差补偿已更合适；随机SVD或原候选选择带来的差异；smooth与低秩强耦合，单改初始化并不足够。阴性不排除完整官方含LR的smooth/交替优化。

## 三个臂及固定合同

- `legacy_selected`：现存E009导出，不修改。
- `error_init`：`Ws=BF16(W*s)`；`Q0=legacy_nvfp4_qdq(Ws)`；`L≈SVD_r(BF16(Ws−Q0))`；得到BF16 A/B，`Rq=Q(BF16(Ws−BF16(B@A)))`。
- `weight_init`：同样Ws，以`L≈SVD_r(Ws)`生成BF16 A/B，其余导出表达式完全相同。

两个新臂每层重置同一固定seed `310000+1000*block+50*local_index`，沿当前LowRankBranch的FP32 randomized SVD（rank32,q40,niter2），关闭TF32。不是FP64 full SVD；不得称error_init逐位复现历史首次候选，历史RNG环境未绑定。原smooth、rank、attention、低秩执行、native量化、非目标层、模型输入及CFG1全部保持。

流式读取200层原始W，每次仅留一层；两个新臂分别导出全部200层。每层验证原W哈希与E009绑定、s/A/B形状、导出BF16 residual表达式、packed decode与独立旧QDQ逐元素一致。保存源/配置/输入/导出哈希，不覆盖旧产物。导出检查证明表示合同，不证明质量。

## 最小有效读出

使用E014/E015已有p30 s5/s6、p36 s14/s15共四个teacher输入，模型完整50block/200native linears运行。BF16参考原样复用。同输入legacy输出使用E059已验证的六份zero结果，本轮先重放p30 source一次，raw/velocity须逐元素相同；失败即停止。新臂逐调用核对blocking actual input、102 SDPA、200 native及200pack、无disk forward。

主读出为每个teacher输入的video velocity FP64 SSE对相同BF16参考。`weight_init`只有在**四点均同时比legacy_selected和error_init降低至少20%**时，才允许考虑下一步正式matched PTQ。阈值是资源决策，不是显著性；这些旧样本不是独立泛化测试集。报告全部四点，不按均值掩盖异质性。报告audio及去通道均值读出作为辅助，不以之改主判据；已有plain同输入结果一并保留。

最多先9次完整forward（1 legacy replay＋2新臂×4teacher）。只有主门槛通过，再执行两新臂各两个E015 legacy-shifted输入，共最多13次。它们只检查同一旧偏移输入的误差，BF16参考为E057；不是任一新臂的自由轨迹。若未通过，不补这4次。

## 决策 / 边界

主门槛通过：现有legacy残差不足以宣称标准SVDQuant已无能为力，优先准备有正式校准/独立质量验证的强基线；不把初始化差异命名为贡献。

未通过：停止当前固定smooth初始化对照，不扩rank/seed/步数/提示网格；保留完整官方配方仍未验证的限制。即使weight_init更差，也不能以此恢复“现有legacy就是官方最强baseline”的口径。

技术检查失败：保存位置/预算，不能解释为方法阴性。不通过重新开预算或改量化规则挽救。

预算：一张已复查空闲SM120，导出及评估合计1800秒，allocated<60GiB，新增export≤26GiB（旧单份约11.14GB，两份约22.28GB），最多13完整DiT。CPU check先做，0CUDA/0forward。无视频生成、无MJ、无质量/部署速度结论；不自动启动完整PTQ。
