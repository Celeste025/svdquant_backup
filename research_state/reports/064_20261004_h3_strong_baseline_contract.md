# H3：补正强基线比较合同

2026-10-04。E069完成固定版本的源码及CPU语义核查，0 GPU、0模型前向。**没有新方法或画质结果；本轮发现的是必要基线缺口。** E068盲评仍待人工反馈，不重复提问或推定偏好。

## 动机 / 观察

E065b只改变固定legacy尺度下的低秩迭代方式，E066/E067只诊断这两个最终候选。它们没有覆盖完整SVDQuant校准，因此不能用这些结果断言成熟方法已无法解决H3的量化问题。

## 核查

将DeepCompressor上游固定在commit `69f3473f5e1c1504bae35cc50c7858ef900a9b17`，保存12份源码（243155字节）及逐文件SHA。直接执行从源码AST提取的纯候选枚举方法，与当前H3脚本的指数表达式比较；不导入模型或重新校准。

## 结果 / 结论

| 合同 | 当前H3脚本 / E065b | 固定上游或必要边界 |
|---|---|---|
| 相同grid10平滑候选 | 9个，只有α与1−α组合 | 同网格规则19个，多出恒等尺度及9个仅迁移激活幅度的候选；默认grid20为39个 |
| 平滑候选评分模型 | 当前H3脚本无低秩支路；E065b固定旧尺度 | 上游启用`allow_low_rank`，评分时纳入低秩分解 |
| SVD数值求解 | 本地FP32随机近似，q=rank+8、2次幂迭代 | 固定上游完整FP64 SVD；未检验此差异的实际效果 |
| 评分终点与输入来源 | 当前H3源码已有完整block评分、块间量化输出传播；E065b使用teacher逐linear目标 | 上游按模块选attention/linear或部分block终点，不能统称全block目标；缓存顺序也须单独核查 |
| 原生NVFP4合同 | 本地W/A均有FP32全局尺度与E4M3微块尺度 | 上游该版本NVFP4 YAML的激活尺度层级不同；忠实算法移植到本地native格式须明确披露 |

候选枚举审计耗时0.033秒，结果文件完整。独立复核另用有理数构造候选集合，并重验31份文件，0.027秒通过；没有重跑原审计方法。上述计数是指数候选数，不能保证经过实际span、截断与BF16舍入后每个scale张量都互异。**这些差异只证明比较身份尚未等价，不证明上游更好，也不使旧同实现配对实验失效。** 历史checkpoint的生产脚本SHA仍缺失，当前源码能力不能倒推历史执行过程。

另一个纠正是：不能将旧H3基线概括为“忽略gate、norm与attention的线性层校准”，也不能将把这些模块纳入评分本身认领为创新。独立结构筛查未提出新的方法候选。

## 后续决定

不继续扩大固定旧尺度的迭代或仅凭SSE做机制扫描。下一步先验证完整H3 packed sequence接入校准器的合同，并做一层资源pilot，再决定完整强基线的可执行预算。通用缓存按第0维切批且要求同形状，直接把`[M,C]`传进去会拆坏attention样本；仅增加batch维也未解决不同prompt长度。可用独立case索引驱动真实attention、保留每例完整输入和RoPE/cu_seqlens，再以零干预逐字节回放验证；目前只是接入方案，尚未运行或证明正确。

补齐基线按工程记账。新的研究贡献仍需独特问题、可区分预测、近邻比较以及实际视频质量与成本证据；不会把“改回已有正确流程”算作创新。

[E069协议](../06_experiments/E069_h3_recipe_contract_plan.md) · [CPU审计](../../results/research/E069/audit.json) · [上游源码清单](../../results/research/E069/upstream/source_manifest.json) · [完整差异与接入合同](../04_prior_work/h3_full_recipe_gap_audit_20261004.md) · [结构问题筛查](../02_problems/h3_svdquant_structural_residual_screen_20261004.md)。审计SHA `aa993946fbf01fc428067ba9a0d6cb59befec0cdbf7b5bbb4fd79de10480faf0`。

[独立复核](../../results/research/E069/independent_review.json) SHA `6a98b15801658cb7ca2e5b56072063756b7089005a8892b1c779247d946b167d`；仅覆盖源码身份、枚举语义与声明范围，不证明完整配置加载器、H3接入或质量。

一手配置与算法源：[固定SVDQuant配置](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/examples/diffusion/configs/svdquant/__default__.yaml)、[低秩求解](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/nn/patch/lowrank.py)、[平滑候选处理](https://github.com/mit-han-lab/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/calib/smooth.py)。
