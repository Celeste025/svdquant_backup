# Phase 4：两个部署信号及强基线缺口

2026-10-02；CPU只读真实profile、冻结结果和已安装SM120源码。未运行GPU、安装包、实现新后端。**目前没有证据证明成熟技术不足以解决这两个信号，因而不建立论文claim。** 也不能把单卡、单shape的结果外推为容量/长视频规律。

## 1. 低秩补偿的实测成本已超过部分模型的主支节省，收益价值却未被整模对照确认

以下均为严格`cat=kernel`归因，不重复计入CPU/GPU annotation；kernel工作量不是关键路径时间。两模型形状、架构不同，不能把它们当纯模型规模实验。

| 实测 | Wan1.3B，E007 | H3，E009 |
|---|---:|---:|
| BF16 / native完整DiT中位时间 | 1.770 / 1.975 s | 8.260 / 6.787 s |
| BF16主支 / native主支kernel工作量 | 312.0 / 96.8 ms | 3666.7 / 1008.8 ms |
| native LR / smooth / pack工作量 | 253.4 / 69.1 / 89.2 ms | 473.0 / 160.0 / 200.3 ms |
| native attention工作量占比 | 42.8% | 49.3% |

Wan的LR+smooth约322.5ms，已经大于dense→FP4主支省下的约215ms；这是真实部署负担。H3则仍有1.217×收益。当前完整模型只测rank32 SVD状态，E003/E004的plain仅局部脉冲，**没有证明这些额外成本在相同完整输入上买到了什么误差收益**。[E007数据](../../results/research/E007_profile_summary.json)、[E009数据](../../results/research/E009_profile_summary.json)

这不是低秩算法或SM120的固有限制。已测的现成FlashInfer融合在H3长M QKV组件10.161→6.641ms，但FC2为5.678→5.799ms，短M也更慢；默认策略、动态低秩rescale和形状均影响结果，不能用“一般fusion肯定解决”或“fusion无效”代替完整对照。[已有融合证据](../06_experiments/E009_fused_up_results.md) 当前尚无依据开发新融合、rank选择或并行机制作为贡献。

**排除两个伪信号：** H3 native启动37.62GiB而稳态峰值16.80GiB，是先装BF16再转换的loader选择，不是packed模型需要37GB；52次D2H总共624 bytes，CPU约6.6秒等待包含先前GPU计算，不能当成可直接节省的6.6秒传输。直接packed加载、CPU元数据缓存均是普通部署修复。

## 2. 已能调用的FP4 attention会物化部分二次大小的修正矩阵

E009的attention占3.349s，说明继续只优化linear覆盖不足。E006已经在相同SM120上用官方FlashInfer完成真实H3 QKV和原生FP4 attention；其记录的Q/K/V形状为`[22400,56,128]`，主段有效长度22384。**但这条成熟路径的预处理并非线性显存：** 已安装0.7.0.post1在`per_block_mean=True`时先生成FP32 `qk_correction[B,H,Mpad/128,Npad]`。由实测形状与源码分配可确定主段该单个张量为0.818GiB；若等长tokens增加至44800/89600，公式给3.271/13.084GiB。后两项只是容量推算，未测长视频、峰值、OOM或性能，不能冒充实验发现。[E006实际形状与kernel证据](../../results/research/E006_h3_attention_interface.json)、[官方接口](https://docs.flashinfer.ai/generated/flashinfer.nvfp4_attention_sm120.nvfp4_attention_sm120_fwd.html)

来源是本机`flashinfer/nvfp4_attention_sm120.py`的`_preprocess_qkv`，文件SHA `040738c6d6af5584ca9b375bb59468d7011506862ff0a162f1f51a72149abefc`。这是特定API物化策略，**不是NVFP4/softmax硬件的必要下界**。同一官方API已有`per_block_mean=False`，只生成单行均值修正、容量线性；真实H3下精度与速度尚未比较。已有BF16 FlashAttention也可作为内存对照。因此暂时没有“普通成熟baseline不足”的证据；不应先新写分块修正kernel或宣称低位attention存在固有二次显存壁垒。E006的损伤交互门槛失败，也不等于这条attention部署基线不可用。

## 唯一优先补的强对照：完整plain native，同配方去smooth/LR

优先回答信号1的缺口。原始BF16 W直接按**现H3配方**量化，200主干linear覆盖相同、8个refiner保留，bias保留；输入直接pack，既不smooth也不LR。沿用E4M3 RNE、E2M1 signed ties-lower及已有zero-SF规则，标签`plain_h3_recipe`；不混入标准E2M1 RNE，也不声称最强SOTA。不可对旧残差packet简单删除LR。

最短实现为独立约100行plain模块与逐层`safe_open`导出，复用已有packer/packet/`scaled_mm`，无需新kernel或完整37GB BF16常驻导出。全部200权重须对独立旧QDQ重建数值exact，并做有界同packet GEMM检查。E009旧完整导出含smoke为215.5秒，plain预算5–10 GPU分钟是保守估计、非新实测。

随后只在预先指定E009 p1/s0、E010 BF16轨迹p30/s5与p36/s14三个共同状态，比较BF16/SVD/plain完整双模态endpoint；至少固定p1计时才可谈成本，慢参考packing不能用于速度结论。若plain同样快且误差不劣，优先删除不值成本的复杂度，不立方法claim；若SVD确有稳定收益，则保留这个质量–成本约束，再考虑是否存在普通成熟部署无法满足的目标。三个输出NMSE仍不是视频质量。未建立这个对照前，不建议再扩长视频、分辨率或多卡的研究假设。
