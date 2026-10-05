# H3：校准候选与实际部署一致性

2026-10-04，E071。**本轮完成基线验证，没有新方法或画质提升结论。** GPU任务及独立CPU复核均正常结束，GPU0已释放。

## 动机 / 观察

此前已接通H3变长attention缓存，但尚不能保证校准评分对应真正部署的模型。重复平滑、scale精度或低秩分支输入不同，都可能让“校准更好”变成实现假象。这轮只区分：同一部署合同能否贯穿真实库内校准与导出；是否存在上述接线不一致。

## 实验

固定首层QKV、rank32、一个非恒等平滑候选(alpha=beta=0.5)，复用两份完整校准输入（22400/22464 token）。实际调用SmoothCalibrator的calibrate/reset/ask/score/tell，候选采用原生NVFP4算子；一次BF16平滑后做一次完整FP64 SVD，低秩分支读取未量化的平滑后输入。重新从导出文件构建模块，比较完整QKV输出。

| 项目 | 结果 |
|---|---|
| 两次BF16 attention参照 | 与E070逐字节一致 |
| 校准候选与导出重载 | 两例完整QKV、量化packet及保存的低秩输入/输出证据逐字节一致 |
| 原模块恢复 | 对象、权重、存储及hooks均恢复 |
| GPU成本 | 总96.02秒，其中精确SVD52.57秒；峰值分配44.80 GiB，产物6.67 GB |
| 独立CPU检查 | 24.28秒通过，143份文件身份核验，0 CUDA / 0模型前向 |

共4次attention（2 BF16＋2候选）、2次重载QKV、8次SDPA、4次native GEMM和activation pack；0完整DiT、0新视频。唯一GPU启动按原900秒deadline结束，worker/supervisor均退出码0。

独立程序用NumPy核验完整输入、平滑、量化编码、残余减法、top32奇异方程与两种评分。真实库分数采用BF16相减再FP32平方归约，CPU复算总分相对差4.89e−10；另行FP64差值SSE相对差1.20e−16。两种定义没有混用。独立程序没有重演GPU GEMM、完整SVD或库调用，执行身份仍由绑定源码的收据支持。

## 结果 / 结论

这一候选的校准和部署没有出现所检查的接线差异，可以推进实际多候选配方校准。**单候选不能证明搜索排序、完整SVDQuant配方有效或视频质量改善。** 当前executor使用native适配，w/x/y quantizer=None；也不能称为默认库QDQ hook路径或stock YAML的逐字节复现。

后续直接规划实际校准，避免重复接口smoke；完整强基线仍需补齐候选搜索、低秩优化和整模评价。相关工作中已找到社区H3 calibrated-8x20成品，公开Diffusers加载代码明确选择NVFP4激活。但必需kernel仓库本次匿名访问返回401，尚未在本机运行；因此暂不能直接替代强基线。312对200的模块数量主要含QKV拆分，并不证明底模不同；额外refiner/AdaLN覆盖和低秩预算仍需映射。见[运行合同核查](../04_prior_work/h3_community_runtime_contract_20261004.md)。

完整本地配方的机械成本尺度可能达百GPU小时，不能照搬单候选循环就默认投入；[实施草案](../06_experiments/E072_native_baseline_design_draft.md)比较了公开成品与本地校准路线，并保留一次同矩阵精确SVD执行方式比较的有界备选。尚未启动下一GPU任务，未将备选12 GPU小时或全模型预算选为执行方案。

创新仍要求在强基线下成立的问题、可证伪的机制预测及相对最近工作的实质贡献。伪量化修复、常规kernel优化、已有配方接入均记作工程，不包装为论文贡献。E068仍待人工盲评，无新增质量结论；当前无可投稿核心成果。

[实验协议](../06_experiments/E071_h3_native_candidate_plan.md) · [GPU结果](../../results/research/E071/run.json) · [独立复核](../../results/research/E071/independent.json) · [退出收据](../../results/research/E071/process_exit.json)。
