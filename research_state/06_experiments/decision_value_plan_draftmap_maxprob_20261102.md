# DraftMap 的块内最大 softmax 概率命中率

- 模式：探索；为回应用户具体要求，在已有单 prompt 轨迹实验中加入第三种真实块重要性定义。
- 样本和代理：复用 `decision_value_plan_draftmap_hit_20261102.md` 的 prompt 16、同 seed、20 steps、5 layers、8 heads、Q128、MeanMax0.2、global per-head top-k、原生 DraftMap。
- 真实参考：对 video query 的全部 prefix+video key 计算同输入完整 softmax，对每个 Q128×K128 video tile 的有效 query/key token 取最大概率；在每个 head 的视频 tiles 内选前 10%、20%、30%、50%。随机匹配预算参考为相应预算。
- 同时复算原先块质量和 QK 最大值，用以检查运行稳定性；不声称完全独立观测。
- 敏感性和局限：单 prompt、单 seed；BF16 QK GEMM 输出再 FP32 softmax；只做同 timestep 的路由重合，不能解释视频观感或直接证明跨步复用。统计的 timestep 标签交换假设未证实，仅为探索性分析。
