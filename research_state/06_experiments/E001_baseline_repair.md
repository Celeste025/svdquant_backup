# E001：H3低秩输入精度的基线纠错

类型：baseline-correctness；不是创新主张实验。

问题：旧hook用QDQ输入驱动低秩分支，导致对SVDQuant的错误评价。
假设：标准公式与修复hook一致；同一真实H3 block输入、旧state不变时，修复改变输出，但是否改善真实误差仍未知。
最小有效设置：H3 block0、原始576×1024/124frames packed序列，校准prompt 1/20、step 0/19；同BF16参考比较plain W4A4、旧低秩输入、新高精度低秩输入；按video/audio/text分别记录NMSE；BF16重复必须一致。
保持不变：quantizer舍入规则、smooth、rank32 factors、模型权重、attention精度。不能同时改舍入再把收益归因于hook。
决策：公式回归失败则修代码；真实block仍差则检查旧state目标污染并重校准小范围。无论有无改善，都不直接扩大到全量视频结论。
停止标准：4个配对block case完成即停止该诊断，后续自由rollout需要独立计划。
禁用proxy：随机MLP结果不能冒充真实H3效果；只测试前128文本token不能替代全packed序列；旧state不是重新校准后的标准基线。
输出：`results/research/E001_h3_branch_precision.json`，日志 `results/logs/research_E001_h3_branch_precision.log`。

执行补记：首轮环境自动选择SageAttention，故虽arm后端一致但不是预期BF16 attention。已在GPU5固定DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch复验，输出为E001_h3_branch_precision_torch.json，两组结果分开保存。新版脚本显式默认torch并记录backend。
