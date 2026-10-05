# H3：实际激活量化残余支路的预算与先例边界

2026-10-04。0 GPU、0 forward、未改代码。本轮仅保存一个表示方案的窄查新结论；不重开 E064 的输入依赖净损伤解释，也不把 E059 数值控制失败当成激活机制阴性。前序筛查用了三条定向查询；本次保存仅补核对 SVDQuant 原论文入口。

**结论：可以构造同 rank32 参数和小 GEMM 算术预算的实现，但目前没有筛出足以独立立项的核心剩余。** “读取真实 activation residual”及其低秩 side code 已有直接近邻；这不等于某一论文完整实现了下面的 H3 native 配方，也不等于整个高精度表示方向为空。

## 具体输入、输出与预算

在已经应用相同 smoothing 的坐标中，令 `X∈R^(T×d)`、`W∈R^(d×m)`。`Xq=decode(actual_packet(X))` 必须由本次主 GEMM 实际消费的 E2M1 codes、group16 E4M3 scales 和 global scale 还原，不能拿另一套软件 QDQ 冒充。考虑：

```
E = X − Xq
Y_hat = native_nvfp4(actual_packet(X), packed_W) + (E A) B
A ∈ R^(d×32), B ∈ R^(32×m), A/B 为 BF16
```

输出仍为 `T×m`；bias 按同一线性层合同处理。主权重应从原 W 构造所比较配方的 `Q_W(W)`，不能剥掉 legacy 的 LR 后复用旧量化残差矩阵作为 fresh 主权重。

| 项目 | rank32 残余支路成本 |
|---|---|
| A/B 存储 | `2×32×(d+m)=64(d+m)` 字节 |
| 两次小 GEMM | `T×32×(d+m)` MAC，或按乘加计两次的 `2T×32×(d+m)` FLOPs |
| 中间 `EA` | BF16 为 `2T×32` 字节，不含 workspace |
| 残余提取 | 额外解码、减法与数据移动；若显式保存 E，再需 `2Td` 字节 |

这些参数量及小 GEMM 算术量与 BF16 rank32 线性支路相同，**不代表总延迟、带宽或峰值显存已匹配**。将残余提取/投影融入已有 pack 路径可能省去完整 E，但尚无本轮实现或测量；常规融合只计工程。部署时不得保存 BF16 全 W 或额外完整高精度矩阵来帮助支路。

## SVDQuant 已隐含保护什么

对 [SVDQuant 原论文](https://arxiv.org/abs/2411.05007) 的主支路/低秩结构作实数代数展开，省略共同 bias，记 `P=AB`：

```
Y_svd = Xq Q_W(W−P) + X P
      = Xq [Q_W(W−P)+P] + (X−Xq) P
```

因此不能把 SVDQuant 描述为“只保护 W、没有保护激活误差”：同一个 P 既参与有效权重重构，也保护 activation residual 的投影。这是本笔记的代数观察，不声称不同融合/加回次序在 BF16 下逐位等价。

上面的替代方案将主权重固定为 `Q_W(W)`，让 P 单独服务于 E 的读出，因而改变了两项之间的约束；它**并非原 SVDQuant 配方逐参数等价的重命名**。但“把这两种作用解耦”仍需超过下面直接先例，不能凭该等式就宣布新方法。

## 主源碰撞：直接覆盖与非完全覆盖

| 一手来源 | 已覆盖的具体结构 | 未完全覆盖的边界 |
|---|---|---|
| [HeadQ §4，Eq14–17](https://arxiv.org/html/2605.03562v1) | 对真实 key 量化残余 `e=k−Q(k)` 保存低秩坐标 `z=Q_side(Uᵀe)`，读取时以 `qᵀUz` 修正 logits。将 readout 换成固定 W，得到 `E U UᵀW`，即本候选的重要特例。 | 原工作是 KV cache/key-side correction，含查询基、side-code 量化和读写生命周期；不是 H3 全部 linear 的 native W4A4，也未直接覆盖任意独立学习的 A/B。不能说整套 H3 配方已被它实现。 |
| [QUADS §3.3，Eq20–24](https://arxiv.org/html/2607.15810v1) | 读取实际 NVFP4 激活残余，按在线残余范数选通道，以第二遍 FP4 量化补偿，保留 W4A4 主路径。 | 它采用稀疏通道和第二次低位补偿，不是同 rank32 BF16 两小 GEMM 的低秩表示；但“保护真实激活残余、兼容原生 NVFP4”已直接覆盖。 |
| [LoRaQ §3.2–3.4](https://arxiv.org/html/2604.18117v1) | 塑造权重量化误差为低秩，并在同存储预算下量化支路、增加可用 rank；激活侧沿用 smoothing。 | 不直接读取 `X−Xq` 作为支路输入。它是必须面对的同预算强对照，不能将“支路换精度/表示”泛称空白。 |
| [SPEAR §3.1，Eq3–4](https://arxiv.org/html/2606.11244v1) | 对低秩坐标 Ax 加输入相关门控，形成非线性量化补偿。 | 门控只观察 Ax，不直接观察实际量化胞元残余；一般不能由 Ax 唯一恢复 `(X−Q(X))A`。因此不是完全同构，但仅换 adapter 输入也不足以构成可信贡献。 |
| [DeltaQuant 官方项目](https://hanlab.mit.edu/projects/deltaquant) | 时空 cube 的 mean/core 保留 FP8，delta 用 FP4，权重沿用 SVDQuant 低秩。 | 保护的是邻近 token 公共成分，不是实际量化误差的低秩坐标。此次边界依据官方项目页，不声称已核查原文 kernel。 |
| [NSNQuant §3](https://arxiv.org/html/2505.18231v2) | normalization–shift–normalization，保留恢复所需信息，结合 Hadamard 与 KV 向量量化。 | 不直接实现上述 residual branch；不能因为同样保留高精度 side information 就认定完全覆盖。 |

## 窄结论与后续约束

目前能提出的可检验工程假设是：**解耦主权重重构与 E 的高精度读出，能否在实际总成本受限时优于完整 SVDQuant、LoRaQ 与 QUADS 式残余补偿？** 这个假设不要求预先证明 H3 的 activation error 主导；也不能把现有 legacy rank32 收益小当成完整 SVDQuant 已被排除。

但本轮没有找到超过 HeadQ 式残余坐标迁移、QUADS 式真实残余读取及普通低秩回归的独特方法结构。故它至多是一个强基线比较方案，**不据此注册新方法 claim，不自动追加 GPU、oracle 或训练**。若以后要重提，必须先指出比“把 adapter 输入从 X 换成 E”更多的可部署结构，以及能区分该结构与这些成熟对照的预测；不能只换 loss、rank 或 kernel 名称。
