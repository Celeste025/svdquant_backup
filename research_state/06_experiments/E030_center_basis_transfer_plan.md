# E030：单文本固定 query-center basis 的跨文本迁移

2026-10-03，GPU前固定。E029 same-sample rank16三geometry均保留global→block张量误差优势91.66%–96.66%，没有稳定geometry胜者。选择最简单Euclidean basis，检验是否必须依赖当前Q的SVD；不把特定K4-error几何写成收益。

校准只用已经保存的E017 p30/seed49771/step14/block_mean自由轨迹，重放一次完整原生DiT，在block0/24/48的原post-RoPE Q处捕获官方GPU BF16块均值与padded全局均值。p30有效长度22227，pad22272，尾组83；文本501、音频414、视频21312。每head δ=μblock−μglobal，√wδ的FP64 SVD取前16个右奇异向量B（16×128、正交行）。保存均值、B和谱等小工件，不保存新的完整QKV。固定每层/每head basis，不汇合层或head，不追加文本/步数。

测试仅用原E018 p36/seed59526/step14三层QKV，长度22539/pad22656/尾组11。读取p30固定B，当前μ在同B下投影：δ16=(δ Bᵀ)B；FP32重构c=μglobal+δ16，再一次BF16 cast。与E029相同的c同时用于BF16 Q减法和FP32 cKcᵀ，只重pack Q、复用原KV四packets。每层第一臂为上述BF16中心；另加实际factorable FP32中心臂，详见下文GPU前补充。端点与same-sample Euclidean直接引用完整E029输出，不重复GPU调用。全程一次DiT（含原50FP4attention等）加6次额外attention，无新生成视频。

全部三层/56heads报告对BF16 NMSE/cosine、对E029 free-block/global/same-sample rank16的变化，以及保留(global−candidate)/(global−block)优势。重放raw输出与已有E017按数值差异报告，不设逐位科学gate；具体输入、native安装和实际调用计数仍须核实。CPU独立读取输出复算，basis正交性、rank及输入长度可检查，不重复SVD训练。

这是一个探索性跨文本和文本长度的迁移诊断，相同步数与分辨率，不能代表跨步/跨长度泛化或无泄漏论文测试。p30/p36均是历史研究样例。当前仍物化完整correction、逐块BF16重构，非低秩consumer性能测试；不因为basis转移成功就先宣布显存/加速。GPU0单进程，600秒总预算，最大60GiB guard（完整H3转换旧峰值超过30GiB）。模型/env/缓存/工件留DATA1，失败保留，不隐式追加rank/样例网格。

GPU前补充（v1原样保留，尚无E030执行结果）：consumer审查确认逐元素BF16中心舍入使严格rank17分解不成立。因此同一个固定B增加factorable_fp32臂：A由当前δ与B在FP32计算，center32=global.float()+A@B.float()；Qcenter=(Qpad.float()−center32).to(BF16)，T0=global.float()@Kc.float().T、Tb=B.float()@Kc.float().T，correction=T0+A@Tb，仍完整物化后交原kernel。主Q减法与恢复项使用同一因子定义的数学中心，FP32结合顺序带来的舍入保留；不暗称它与BF16中心是同一数值接口，也不增加逐位gate。每层2臂共6probe，实际计数分开capture50与probe6。此臂回答未来17行消费者的算术合同是否还能保留精度；真正消费者尚未实现，当前没有测性能。600秒总预算不变。
