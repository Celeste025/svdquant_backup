# E009b：现成SM120 fused-up结果

2026-10-02。已完成6个synthetic与4个真实shape的有限验证；没有完整H3速度或质量结论。

| H3 block0层 | M | 未融合组件 ms | FlashInfer融合组件 ms | 前者/后者 |
|---|---:|---:|---:|---:|
| attn.qkv_proj | 512 | 0.274 | 0.343 | 0.798 |
| attn.qkv_proj | 22400 | 10.161 | 6.641 | 1.530 |
| mlp.fc2 | 512 | 0.207 | 0.290 | 0.714 |
| mlp.fc2 | 22400 | 5.678 | 5.799 | 0.979 |

计时：GPU0独占；各路径1次warmup+5次repeat，中位host wall（device events另存）。输入已经smooth/pack；包含BF16 down、main/up及add，融合路径还包括动态FP32 alpha和BF16(B/alpha)重算。**不含packing和smoothing**，不是完整linear/DiT计时。未融合组件省略旧helper额外的乘1写回，是可复用的普通实现对照。默认CuTe策略、未做autotune，不代表各shape的最优融合性能。M512 fc2有一个0.505ms离群repeat，完整序列保留。

同codes的fused主干与torch原生主干，四case数值逐元素一致。融合完整输出对原BF16 up分支的NMSE为7.53e-6–8.19e-6，来自B/alpha的BF16舍入和融合epilogue的舍入顺序；单层误差不能替代整模或视频质量评估。所有输出finite，实际SM120 CuTe FP4 GPU kernel已观测。

结论：复用现成fusion在这个QKV形状有潜力；不能对全部层统一套用并宣称加速。FC2与短M缺少收益。普通fusion/形状选后端已有近邻，不作为论文贡献。等E009完整native与快速packing SHA闭环，再决定是否只给QKV接入融合并评估整模传播。

证据：`results/research/E009_fused_up_real.json`及其export前置证据快照；[结构化摘要](results/E009_fused_up_summary.json)；[预注册](E009_fused_up_plan.md)；[隔离环境](../00_state/native_fused_environment.md)。当前`probe_h3_fused_up.py`已与结果SHA匹配并冻结；后续扩展新建文件。

[FlashInfer官方API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_nvfp4_svdquant.html)提供该融合数学约定及SM120接口。
