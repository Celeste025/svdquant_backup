# E066 后的部署结构边界审查

2026-10-04；仅本地文档/代码与三个一手全文，无 GPU、无实验源修改。**本轮没有可直接立项的新方法残余。** 固定 native NVFP4＋rank32 有可干预结构，但还没有区别于已有分解与扰动校准的实证预测；这不等于证明该方向不可能产生贡献。

**事实。** [E066](/home/wjq/workspace/svdquant-exp/research_state/reports/061_20261004_h3_local_transfer.md)：四个 teacher 状态的 200/200 层，carry 同输入 full-video SSE 更低，完整 DiT video SSE 却更高；真实 M512 不改变排序。这排除了这些 teacher 点的局部收益丢失与采样/形状排序假象，未排除 student 输入变化，未证明方向变化或 packet 切换致害。固定非负逐层 SSE 加权无法翻转这两个候选的排序，不等于该目标不能训练出第三个候选。

**精确结构。** [NativeH3Linear](/home/wjq/workspace/svdquant-exp/scripts/research/h3_native_nvfp4.py:114) 固定 kernel/配方/rank；两臂 packed 权重不同。平滑后行输入 X、L=BA、已解码残差权重 R，在忽略 BF16 算术舍入的理想化下，

$$f_L(X)=Q_A(X)R^T+XL^T+b,\qquad \Delta f_L=\Delta Q_A R^T+\Delta X L^T.$$

沿连续 X 的同一 code/scale 边界取两侧极限，若 J_A 是 Q_A 的非零跳变，则 **J_f=J_A R^T**；连续 LR 项不能抵消这个瞬时跳变。carry 改变 R，因而可以改变跳变几何。这不是有限 student 扰动的全部响应：后者还有 ΔX Lᵀ。实际 BF16 输入/两次 LR linear 自身也有离散舍入，须另分离，不能把理想化等式冒充数值证据。Q_A 含动态全矩阵 global 和 E4M3 block scale，不能假定所有胞元内 Q_A 恒定或总 Jacobian 只有 rank32。

**先例碰撞（方法全文）。** [SVDQuant §4.2–4.3](https://arxiv.org/html/2411.05007v2) 已含相同双支路、分解 W−Q(R) 的迭代与支路融合；[ARHQ §2、§5.3–6](https://arxiv.org/html/2605.00140v1) 已以激活残差协方差分配低秩容量，明确再次量化残差权重的 tradeoff 并提出联合目标；[QDrop §3.2–3.3、Algorithm 1](https://arxiv.org/pdf/2203.05740) 已用量化/浮点输入混合及激活量化丢弃改善输入相关扰动响应。因此改用 jump covariance 做加权 SVD，或把实际 packet 扰动加入重建，首先属于已有路线的校准分布/目标变化。native 合同不同不自动构成算法贡献。

**已有反例。** [E064](/home/wjq/workspace/svdquant-exp/research_state/reports/058_20261004_h3_depth_four_corner.md) 中 16 个非零深度单元均有 ||p+c||²−||p||²>0，却有 net_c=||p+q+c||²−||p+q||²<0，因为 2⟨q,c⟩ 为负并抵消增加。该 c 混合多种来源，也属于旧配方，不能代替 carry/restart 的测量；它足以阻止把“响应/jump 能量更大”直接命名为有害放大，不能改名重开已有阴性候选。

**什么预测才可能增加信息。** 一个尚未观察的必要预测是：预先指定的真实 code/scale 切换结构，能在未参与选择的状态预测共同后缀内的**有符号排序反转**，且同 rank/kernel 的分解改变该结构后反转消失；局部 SSE 缩放与 BF16 舍入对照不能解释它。若只由扰动协方差或总能量预测，ARHQ 式目标已足够；即使在匹配这些二阶量后仍有差异，也只越过协方差解释，**尚未越过可学习非线性扰动效应的 QDrop 式块重建**，仍需同数据/校准预算的直接对照。当前既无这种预测的证据，也无超出常规重建的具体干预，所以不申请新 GPU，不将夹角/缩幅描述或 E066 观察本身晋级为 claim，不重开 F1/F2。
