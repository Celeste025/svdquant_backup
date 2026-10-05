# E072a — 相同H3矩阵的精确SVD执行成本决策

**状态：因用户要求控制工程投入，已于2026-10-04停止本支线，0 CPUcheck / 0 GPU / 0 SVD。以下仅保留未执行协议，不作为下一启动指令。**

2026-10-04。基线工程/资源决策，非创新或质量实验。E071及独立复核已complete，不重演。E072整体校准草案尚未启动；本次只选择其中一个有界成本判别，用户既有研究授权持续有效。

## 动机、竞争解释与决策

E071真实QKV候选默认full FP64 SVD耗时52.572899秒。机械套入完整配方可能耗费百GPU小时，不能在没有资源依据时扩大循环。数学上rank32只用前32奇异向量；全正交补未参与部署，精确SVD也有不同求解driver。检验一个预先指定的执行组合，而不扫描驱动或宣称识别了单一耗时原因。

H1：reduced输出的FP64 gesvd可保留当前候选的数值/部署结果，并显著降低分解成本。H2：速度不足，或有限精度差异导致部署张量不一致。结果决定是否以该组合继续设计实际校准成本；不决定量化方法或画质优劣。

## 最小有效设置

只读E071 candidate.pt的同一 `ws_bf16` [21504,5376]，不重新平滑、不换矩阵。输入SHA `775687de83a0608fc7297fd6b00485864cfcf15679e4b0e85e9baf3cdb60d677`；E071 run SHA `89a086bfbff78f74166f1d6ab9a4b232c6a5d892c78692c3510f276f98751d6d`，独立结果SHA `e86b007f147f6fe957bcf96cb82cddf62b76f8acaefa69604d779ea905bd57d1`。同时绑定candidate、export和正常退出收据及已执行源，不修改旧产物。

唯一分解表达式：`torch.linalg.svd(ws.double(), full_matrices=False, driver='gesvd')`，原生环境torch2.11.0+cu128、GPU0空闲后启动。它仍求完整5376奇异值；不改FP32/随机/rank截断求解，不重新跑默认SVD或模型。计时同E071：同步后开始，包含ws.double与SVD，结束同步。只读旧52.572899秒作单点资源参照，不把非同时、多次统计不足的比值称稳定加速基准，也不把旧整个模型峰值与本次单矩阵峰值混比。

预先使用每对Vh行与旧Vh的dot符号同时对齐U列、Vh行；保存对齐符号与新top32、完整S。不能改旋转/排列或搜索“更像”的因子。计算新A=BF16(Vh32)、B=BF16(U32*S32)、同原GPU BF16(B@A)、residual=BF16(Ws−product)，同冻结zero-SF兼容weight pack(chunk_rows1024)。保存足够CPU核验的张量与身份；不调用NativeH3Linear或任何模型。

## 判据与输出解释

数值有效性：所有S/U/Vh finite，S非负降序，保存top32的Gram Frobenius≤1e-8。shape/非finite/执行故障记录failed；不能截掉失败尾谱冒充成功。

等价门槛同时要求：新旧完整rank32 FP64重构矩阵的相对Frobenius差≤1e-10（按行块直接相减累加，避免两个大trace相减的抵消）；对齐后BF16 A/B、实际GPU BF16 product、residual及weight packet（packed/scales/global）逐字节一致。所有不等保留计数、差值/指纹，不为不等提前终止后续统计。这个门槛只针对当前矩阵，不代表未来所有矩阵逐byte相同。

合法有限数值但不满足等价时，status=complete、verdict=not_equivalent，保留阴性，不放宽阈值/更换矩阵。等价且同步SVD时间≤E071的50%时，允许把该组合纳入下一实际校准的成本设计；仍须考虑64case评分/I/O和其他层形状，不能由此承诺全模型时间。等价但不达到成本门槛则不为此开driver网格，不默认承担全配方大预算。这里50%是投入决策门槛，不是论文贡献的显著性判据。

## 资源与停止

一次GPU0任务，绝对deadline900秒、单设备总使用30GiB上限、新DATA产物≤2GiB；原监督/timeout机制，named tmux，先查空闲。CPUcheck在无CUDA下绑定来源与协议，通过后执行源冻结；修改另立版本/产物。0模型forward/activation pack/TE/VAE/完整视频，1SVD/1weight pack/1BF16 LR-product GEMM。任何超时/OOM/来源不符均停止保留，观察超时不重启。外部监督只清理所属进程组，GPU6/7不动。

结果目录 results/research/E072a；大张量 /data1/models/svdquant-wjq/research/20261004/E072a。独立CPU checker读取保存的因子/符号与旧参照，另算Gram、FP64重构差和BF16 cast，核完整product/residual/packet身份、实际退出/计数/原deadline。它不重演SVD、GPU product GEMM或模型；若packet与已独立核验E071完全相同，不再重复其编码器数学验证。GPU数值收据与独立检查边界分别报告。

Necessary：同输入身份、实际新SVD、完整rank32数值与部署张量、单点时间及资源。Cuttable：更多driver/矩阵、重复默认分解、重新生成视频/校准输入。Future：实际完整候选与LR配方、强基线整模质量/效率，以及独立新机制证据。E068盲评仍待，不重问、不据SSE造质量。
