# 领域工作地图（第一版）

更新：2026-10-02；配套 [paper_index.md](paper_index.md)。本文区分论文证据与待检验推断，未在本机复现任何文献结果。

## 当前判断

“视频模型做W4A4”本身已非足够贡献。2026年的近邻已覆盖时空delta、在线混合精度、误差传播校准、稀疏与量化联合、CFG分支变换、H3 attention kernel。第一轮工作应寻找一个**能解释失败且能产生可部署干预的具体机制**，不是先搭方法再堆评价。

## Main problem formulations / Method families

| 问题表述 | 已有方法族 | 最近碰撞 | 需要保留的区分 |
|---|---|---|---|
| 激活/权重范围难以共同压到4位 | smoothing、低秩旁路、rotations、data-free codebook | P01/P03/P08/P09、DVD-Quant | INT4、spherical codebook、E2M1+E4M3 scale不是同一种量化 |
| 同一层跨步分布与敏感度变化 | dynamic bit allocation、时间感知校准 | P02/P04 | 多步CogVideoX证据不自动适用少步Wan/H3 |
| 局部误差影响完整生成轨迹 | 多步重建、传播风险、在线补偿 | P05/P06/P07 | scalar risk只描写部分方向；多源联合扰动的符号与协方差仍需实证，但不能据此声称新颖 |
| 长序列attention昂贵 | 低位GEMM、稀疏路由、linear补偿、训练适应 | P10/P11/P12/P14、SVG2/VSA | dense低位与sparse低位可能改变最优分块和误差传播 |
| 真实硬件不兑现理论收益 | fusion、数据布局、低精度转换、消除旁路访存 | P01/P13/P14 | 看全算子和端到端；不能仅列TFLOPS |
| 多分支/任务保真 | CFG joint coding、guided-output校准 | P15/GAMP | 同seed贴近BF16和感知质量/运动保真是不同目标 |

## Evaluation protocols

1. **机制诊断**：paired prompt/seed/latent，层输出、模型velocity/epsilon、solver后latent都记录；区分单次forward局部误差与完整rollout。P05的pulse是已有协议，可作诊断工具，不能当新贡献。
2. **量化契约**：记录目标线性层覆盖、W/A实际格式、group size、E4M3缩放、global scale、是否保留BF16分支/特殊层、QK/PV精度。所谓W4A4不能隐藏这些差异。
3. **质量**：paired PSNR/SSIM/LPIPS衡量dense-reference fidelity；VBench及人工盲评看运动/语义/画质。近饱和一致性指标不足以证明无损。至少给paired置信区间与具体失败样例；FVD不用于十几个样本的强结论。
4. **性能**：同GPU同shape、预热与同步、latency与peak memory、preprocess/GEMM/attention/communication占比。分别报告DiT step、全部denoising、整段视频含文本编码/VAE；把offloading避免带来的收益单列。所有真实性能必须运行native kernel。
5. **跨设置有效性**：常规多步与4–8步蒸馏至少一对；发现机制后才扩prompt/分辨率/模型。用小分辨率探针可以排错，但不能代替长token硬件与视频时间机制。

## Common claims / Repeated limitations

- “outlier随时间变动”“视频token相似”“attention稀疏”“QAT可恢复精度”均为成熟论点。
- “compatible/orthogonal”常是组合可运行或某一配置上有效，不代表误差统计独立。
- 不少质量表使用Q/DQ，速度另用native设置；对本研究应要求同一checkpoint/格式的质量与速度闭环。
- 大倍数headline常混入少步蒸馏、缓存、CPU卸载避免或旧attention基线。只有增量、边界一致的比较才支持方法收益。

## Unstable assumptions / Potential research tensions

- **局部MSE与可接受生成误差**：不是新问题（P05–P07）；未知的是native block scaling是否制造特定、可压缩的跨步相关误差，以及方向性干预是否超出已有scalar-risk校准。
- **重复调用的确定性误差**：同一权重在多步反复使用可能产生coherent drift；只是候选机制，须区分权重静态误差与每步activation量化，且补查temporal error-feedback/antithetic rounding。
- **共同计算的统计耦合**：NVFP4全tensor第二级scale可能取决于batch组成；若独立请求因此相互改变数值轨迹，属于serving语义问题。尚未查证具体runtime是否如此，不把假设写成bug。
- **少步改变主导项**：多步方案可摊薄的calibration/聚类/cache成本，3–4步可能不划算；此处单做移植很薄，需找到成本/敏感性共同转变的结构证据。
- **全局旋转与局部microscale**：消除全局outlier不必改善16-element group的误差。但LLM已有micro-rotation与scale优化邻居，不能简单把旋转粒度移到视频。

## Closest work clusters / 拥挤区

- 低秩/均值/残差：SVDQuant、DeltaQuant、QVGen、SharQ、VC-Attention。
- 轨迹与精度分配：ViDiT-Q、DVD-Quant、6Bit-Diffusion、AccuQuant、TCEC、PulseQuant。
- 稀疏与量化：SpargeAttn、QuantSparse、SLA2；“1+1有误差放大”也已公开。
- CFG相关结构：GAMP、GCBT。

## Literature coverage status / 下一步

第一轮满足进入signal mining的条件；15篇核心文献与若干工程/架构补充已建立，全文深度不均，见索引逐篇标记。未形成任何novelty或顶会可发表性判决。下一步只做两条定向查新：跨denoising步误差相关/互补舍入，以及动态batch的scale耦合；再按本机现象决定保留或杀掉候选。
