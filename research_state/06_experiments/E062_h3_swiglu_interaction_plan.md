# E062 — H3 SwiGLU 成对残余的局部判别

2026-10-04，exploration；执行前冻结。主模型 MiniMax-H3 pruned，原 CFG1/20-step 合同。不是新方法验证，不是 kernel 优化，也不从线性 MSE 推断视频质量。

## 动机与竞争假设

H3 每个 FFN 的 fused fc1 同时产生 gate/up，再经过 `SiLU(g)*u`。现有两步残余具有相关性，但不能据此定位误差源；E059 未完成 oracle，不提供来源阴性。当前问题是：**真实 W4A4 fc1 的成对残余是否经乘法产生有实质有害作用、无法由普通偏差/比例解释的二阶分量？** 阳性才值得考虑将有限残余表示预算移至乘法后。

固定同一个完整 BF16 fc1 输入，记 teacher/native 输出为 `(g,u)` / `(g+eg,u+eu)`。在 FP32 计算 SiLU 后：

`a=(phi(g+eg)-phi(g))*u`；`b=phi(g)*eu`；`c=(phi(g+eg)-phi(g))*eu`。

精确实数关系为 `dh=a+b+c`。令 D 为同一原始 BF16 down 权重转 FP32 的线性映射，再乘本次实际 AdaLN gate（转 FP32）。主量 `net_c=||D dh||²-||D(a+b)||²`，同时报告 `||Dc||²` 与带符号交叉项。

- H1：至少一个预选层，在两个状态均有显著正的 net_c，普通中心/比例去除后仍存在。支持进入一次直接干预及可表示性检查。
- H0a：c 小，或 c 主要抵消一阶项；停止二阶表示候选。
- H0b：作用可由普通输出偏差/比例解释；停止复杂表示候选。
- 数值控制不成立：只报告不可判定，不作机制阴性。

宽泛 SwiGLU outlier/权重对齐和高精度通道补偿已有先例：[Dissecting Outlier Dynamics §3.2–4](https://arxiv.org/html/2602.02047v1)。本次只检验实际残余的带符号作用，尚未确认方法新颖性。不能将普通 block reconstruction 或任意新增 MLP 包装成创新。

执行前窄查新补充：[SPEAR §3.1](https://arxiv.org/html/2606.11244v1)已直接采用低秩空间的输入相关门控量化补偿。因此“线性低秩换成门控非线性 adapter”本身已被覆盖。E062 仍只回答具体乘性交互的诊断问题；即使阳性，也不能自动进入普通门控 adapter 训练或建立创新 claim，必须先提出超出该直接先例的结构性干预。

## 最小有效设置

1. 固定 E014 teacher 状态 p30/s5、p36/s14；预选 block 0/24/49，全部六格报告。不根据结果改层或扩提示。
2. 两次完整 BF16 DiT 前向。实际输入与历史 raw outputs / velocities 必须逐字节匹配。teacher 路径不替换；在 fc1 hook 内旁路重放旧 E009 native fc1，完整输入参与量化，保留原 global scale 与真实 NVFP4 GEMM。
3. 量化完成后，才取 video 行中的 512 个固定均匀索引。保存索引、teacher/native gate/up、实际 AdaLN gate、down 权重 hash、投影分量及 teacher 输出。它是局部有限采样，不是全 token 或全模型归因。
4. FP32 SiLU、FP32 down GEMM（TF32 禁用），统计用 FP64。保存 hidden 与 projected 恒等式误差；抽小样本独立 FP64 复算数值控制。实际 BF16 SwiGLU/down 局部计算另存为算术对照，不把其舍入算作 c。
5. 对所有投影分量用同一线性投影 P：先逐输出通道去 token 均值，再去除 teacher 输出的逐通道比例方向。P 由 teacher 决定，对 a/b/c/total 一致应用，保持分解。不分别拟合后再相加。
6. 辅助零假设只有一个：在已选 video 行中循环移动 eu 256 行，保留通道边际统计再构造 b/c。它不保持内容条件关系，不能单独作因果证明，不参与主 gate。

执行前独立审查补充：`net_c/total` 是删除 c 的局部反事实差，可为负或大于 1，不是可加的独立能量份额。循环移位另用原始 total SSE 作共同分母。数值对照保存实际 BF16 隐层四角 `HQQ−HQB−HBQ+HBB` 与代数 c 的差；native/teacher 的 BF16 down 实际对照保留完整输入形状再采样，避免采样改变 GEMM 形状。捕获可在两次 teacher 前向结束后逐层重放 native，不要求 hook 内重放；完整 fc1 输入仅暂存 CPU，结果文件不存全量输入。阴性仅否定本次 gate/up 乘性交互路线，不排除 SiLU 单支曲率。

## 判据与结果对应决策

同一预选层必须在两状态的 raw 和 P-residual 读出均满足 `net_c/||D dh||² >= 0.20`，才推进一次方法相关干预。20% 是预先分配研究资源的门槛，不是统计显著性；六格不是六个独立视频样本。若分母为零，报告零误差且不晋级。

恒等式及小样本 FP64 数值误差相对 RMS 要低于 1%；若失效，保留失败产物并停止，不随结果放宽门槛。BF16 算术对照单独报告，若其残余量级足以改变 net_c 的符号或 20% 决策，标为数值不确定，不据 FP32 阳性晋级。

无论何种局部结果，都不声称已证明生成质量改善、跨步相干来源、线性 adapter 无法拟合或达到发表标准。阳性后还需同预算普通线性 pre/post adapter、block reconstruction 与独立视频质量对照。

## 预算与产物

- 最多 2 次完整 BF16 DiT；6 次完整输入的局部 native fc1；不生成视频、不训练、不新增 checkpoint 导出、不运行 MJ。
- 一张空闲 SM120；启动前重查，不占用他人卡。绝对截止 15 分钟，峰 GPU 分配 <60 GiB，新增张量文件 <4 GiB。原始执行源/结果保留；失败不改源覆盖重跑。
- CPU check 不初始化 CUDA；检查源绑定、输入、预选层形状、样本选择定义。evaluate 输出 `results/research/E062/evaluate.json`，张量 `/data1/models/svdquant-wjq/research/20261004/E062/`；CPU 独立复核后写阶段报告。

必要证据是同输入真实 native fc1、分解/实际 gate 正确与 signed net；样本错配为辅助；全视频/训练在此阶段不执行。QK 几何备选本次不启动，避免并行扩展。
