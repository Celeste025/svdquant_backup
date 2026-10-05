# E070 — H3强基线接入与精确SVD资源pilot

2026-10-04。基线工程；不产生新方法或画质结论。E069完整源码审计已经结束，本次运行真实H3输入/权重。

## 动机与竞争解释

完整SVDQuant配方与现有固定尺度诊断并不等价。进入强基线复现前有两个实际不确定性：(A)通用校准器的样本切批能否在不改变H3变长packed sequence的条件下工作；(B)上游完整FP64 SVD在真实最大线性权重上的成本是否可承受。未经测量，不能声称简单接入已正确或完整PTQ可在既有预算内完成。

## A：真实attention接入

固定原64-state校准缓存中的p1/s03、p20/s03；必须确认两者真实M不同。只用两条完整序列验证接口，不将其当作64-state校准或质量证据。

在相同resident BF16 H3上各运行一次前缀，到blocks.0.attn输出后停止，保存完整实际x、rope_freqs、cu_seqlens、max_seqlen和原输出。2次prefix、0完整DiT。独立case-index wrapper的eval cache每份为[1,1]索引，使用实际DeepCompressor TensorCache/TensorsCache、OutputsError+Layer的_parse_ipts/repartition/extract（batch1、sample_size=-1）；wrapper按索引取完整输入与各自kwargs，调用同一真实attention。再调用2次attention，要求输入/kwargs与输出逐字节相同、每例恰好一次、M未拆分。x_acts另保留variable-M列表供实际span读取，不把索引或metadata当作激活统计。

接受条件：实际cache API通过，完整输入身份和无量化输出均byte-exact，0额外完整DiT/量化/TE/VAE/校准；两prefix各4次SDPA、两wrapper各2次，共12次。任一不满足即保留失败，不降低容差、挑换提示或改用AST/玩具替代后称接入成功。成功只允许推进候选评分接线，不证明全部校准流程正确。

## B：真实最大权重精确SVD成本

固定checkpoint中的blocks.0.mlp.fc1.weight，BF16 [28672,5376]。CPU只读safetensors header及历史身份；GPU阶段仅加载这一张权重并验证其tensor SHA，0模型forward。

严格执行一次固定上游调用`torch.linalg.svd(weight.double())`，保留默认full_matrices与driver，不改为reduced、随机近似或其他精度。记录CUDA同步耗时、显存、全部奇异值、top32 FP64因子与对应BF16 A/B，以及预定16个均匀行/列的全谱重构和top32正交性检查。另存对应W行/列，供CPU独立核验top32奇异向量方程；不能把非零rank32截断残差当SVD错误。预定完整重构及奇异方程相对范数容差1e-8、top32 Gram-I Frobenius容差1e-8；finite、S非负降序及默认U/S/Vh形状也必须通过。全U/Vh不落盘。记录原矩阵的成本，不将未平滑权重一次分解时间视作每个候选/完整模型的准确预测，也不外推质量。

成功允许据实制定完整配方资源预算；超时/OOM只判资源不足，保留进度与退出码，不能算算法阴性或自动切回近似SVD。未通过数值有限性/残差检查则不使用因子。

## 预算、执行与保留

- 两臂可在不同空闲GPU并行，每臂单独一次启动与绝对deadline；A 900秒/60GiB GPU/4GiB新数据，B 1800秒/60GiB GPU/256MiB新数据。deadline含加载、计算、核验与落盘，不因工具观测超时重启。
- named tmux＋外部supervisor保存PID、命令、源SHA与真实退出码。GPU6/7他人任务不触碰。每臂启动前重查空闲。B若native调用长时间不返回，由外部监督终止，不能依赖Python信号强中断CUDA调用。
- 大产物放`/data1/models/svdquant-wjq/research/20261004/E070`，报告放`results/research/E070`。执行源/计划/输出不覆盖，修复另版本。
- 环境只复用既有Python、ninja、extension cache；先做0GPU import check，最多300秒，不安装依赖。环境探针不计模型实验。
- CPUcheck通过并冻结source/plan后才运行。独立检查针对实际保存输入/输出/因子与资源收据；不冒称重演完整SVD或未保存的全矩阵重构。

E068人评继续待反馈，不重复索要，不使用MJ或模型观感代替。用户已有研究授权持续有效。
