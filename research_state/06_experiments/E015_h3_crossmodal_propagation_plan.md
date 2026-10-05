# E015：实际单步量化误差的跨模态四角续接

2026-10-02，GPU前限定。模式 exploration；没有新方法claim。E014完整plain/SVD的模态误差排序交叉，只提供问题动机，不证明跨模态传播机制。此实验判别另一模态的实际量化状态误差是否改变本模态的净误差和配方选择，避免再用局部NMSE直接推断部署收益。

## 范围与近邻

只用E014已固定的p30/step5、p36/step14，两套已有量化配方。TurboT2VA已明确cross-JVP，Synchrony-Aware Sparse Attention已有跨模态重要性，联合QAD也隐式考虑传播；普通模态加权、不同sigma或Jacobian不对称不是新意。不称刚性，不假设线性可加。[精确碰撞](../01_literature/joint_av_propagation_collision.md)

不训练、不生成新视频、不搜索prompt/step/rank/scale，不使用E010自由量化轨迹作为脉冲来源。这样观察的是从共同teacher状态出发的一次真实量化误差，不混入此前累计漂移。

## 构造与执行

令n为5或14。读取E014同一teacher输入下保存的BF16/SVD/plain完整scheduler velocity；以冻结E010原pipeline.step、原20步video shift12/audio shift3，从同一x_n各执行一次原更新，得到x_(n+1)^B及各arm的x_(n+1)^Q。保持实际BF16更新算术，不用x_B+h·(v_Q-v_B)近似。CPU重建的B必须逐byte等于E010第n步latents_after及第n+1步latents_before；若CPU/GPU算术不同，不放宽误差，先查清并保持原路径。

在原n+1双timestep、condition、packed metadata下，每个quantarm执行四个角（video第一坐标、audio第二坐标）：BB、QB、BQ、QQ。仅latents_before根据角选择，其他模型输入保持原样。Q为本arm自己的one-step状态，不能混用plain/SVD来源。两个量化臂下游都维持各自完整200层原生量化，attention仍BF16，不在扰动后换回BF16。

各角的velocity再经原第n+1步scheduler得到Y_corner，即x_(n+2)。**第二次更新必须以该corner自己的latents为sample**，不能统一用teacher-next起步。所有完整raw、velocity、Y张量及真实DiT输入签名保存。BF16臂仅重放两个BB next状态，须raw、velocity、Y逐byte等历史E010第n+1步；这是两个新BF16调用。量化两臂各2×4=8调用，总18DiT。

## 主读出与边界

CPU独立FP64累计每个corner、每模态相对共同BF16 x_(n+2) 的NMSE、error/reference energy；同样保留velocity误差及实际输入扰动的幅度。音频oracle恢复对视频的净作用比较 QQ→QB，反向比较 QQ→BQ；报告error能量相对变化、plain/SVD排序是否改变，允许作用为负（抵消被破坏）。

由原张量计算跨作用Y_QQ−Y_QB或Y_QQ−Y_BQ及交互 I=Y_QQ−Y_QB−Y_BQ+Y_BB，保留方向相关/交叉项验证恒等式。不能用NMSE相减替代向量交互，也不能只因cross-delta大就宣布损伤。两个模态维度、误差幅度不同，本轮不估计等幅Jacobian不对称性。

没有BF16模型的QB/BQ/QQ对照，因此不能称量化增加了固有跨模态敏感性；普通联合生成本来也可能传播扰动。oracle恢复包含attention、共享动态quantization global等全部路径，不能单独归因cross-attention。混合状态是因果干预，不是免费可部署方法或自由生成轨迹；两步latent误差不是质量/同步指标。

决策：若两个预定状态中恢复另一模态对净误差和配方排序均无实质影响，或cross-delta只产生不稳定抵消，则不围绕E014交叉继续该机制。实数与数值地板全报，不事后挑百分比或只留有利模态。若仅一个状态有作用，只记该状态探索线索；只有恢复另一模态明确降低净误差且改变配方选择，才值得有限独立输入/传播窗口验证。无论结果如何均不能直接立新loss或顶会claim。

## 正确性与资源

复用冻结E014/E010源、原模型/导出、实际2D embedding与0D FP32时间字段；仅新薄构造器/runner。CPU先验证真实两状态、原step重放、四角选择/元数据，再冻结新源与输入。每次BF16有102 SDPA/0FP4，quant有102 SDPA/200实际FP4/200合法域checks，均0disk。非目标原dtype保留；所有调用真实输入与构造器预期一致，量化flag非法或任何来源/历史重放失败即停止。

首次GPU起总20分钟，最多20DiT，当前与历史allocated≤60GiB；预期18，不为追结果补调用。GPU5空闲后串行BF16/SVD/plain独立进程，不计性能。间隔需要两次稳定空闲读数，保持同一deadline，保留所有失败。大文件/data1/models/svdquant-wjq/research/20261002/E015，报告results/research/E015。GPU运行只由root统一启动。
