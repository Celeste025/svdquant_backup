# E007 快速 activation packing 契约

2026-10-02。实现：`scripts/research/wan_nvfp4_fastpack.py`；这是旧配方的部署实现，不是新量化算法。

三个 Triton kernels：分块 absmax reduction；按原 PyTorch CUDA 标量运算次序计算 FP32 global；按16通道计算 E4M3 scale、编码 E2M1、直接写 packed bytes 和128×4 swizzled scales。保留 BF16 smoother 输出，不将 reciprocal 折入低秩权重。尺度与码值中点均向更大的有符号数取舍；并非硬件 RNE 配方。

已通过实际 DeepCompressor CUDA quantizer 对照：7个合成case（含M31200、K8960、M257尾块、E2M1/E4M3中点、subnormal和零分组），加3种真实pre-QDQ输入（self-Q、cross-K、FFN fc2）。10case的codes/global/swizzled scales逐字节差异均为0。真实输入来自E007同一forward的quantizer入口；完整文件SHA和源码SHA见[验证摘要](results/E007_fastpack_validation.json)。首版零SF符号+0/−0不一致的失败记录保留，修正后再执行；没有更改量化尺度或选样本追求通过。

异常检查不是可选的数值结论：

- 逐quantizer离线核对单步signed E2M1、FP32→E4M3、tensor-global+group16、channels_dim=-1、FP32 development，无显式scale/range/custom kernel。
- device flags检测NaN/Inf、失效global和`SF=0但code非零`。后者在旧remove_zero语义下可得到非零QDQ输出，原生格式不能忠实表示，必须拒绝。
- 真正全零输入采用canonical `g=1, SF=-0, codes=0`；与旧QDQ输出逐元素相同，但不声称退化scale leaves逐字节相同。
- 默认adapter逐次验证；`collect_fastpack_checks()`可将同一forward的flags收集并在结束时一次验证，减少host同步。性能runner须在接受测量前完成检查，并报告包括检查在内的wall time。仅底层`pack_legacy_wan`返回flags而不自行同步。

完整模型fast/slow native的checksum仍须另行验证。当前packing计时只用于实现调试，不是完整linear、DiT或视频生成加速比。原生完整模型性能协议及结果由独立runner提供。
