# E019 失败后的独立 LSE 诊断

2026-10-03。原 E019 已 failed_stop：block0 首次 BF16 full＋LSE 输出记录不等于 E018；实际消耗1次attention、0次DiT。原frozen_contract、launcher、probe_run、日志和已存输入全部保留。原输出在断言前未保存，不能由此判断差异大小。输入packets和完整correction的SHA已复现，88项冻结源未变。

源码只读发现 return_lse 切换 ReturnLSE 模板实例；目前这只是疑点，不能先声称算子错误或数值差异原因。本补充不恢复原12-call队列，也不放宽原exact gate。

唯一补充：固定 block0 原 Q/K/V packets及已保存且核验的GPU correction，按 False→True 顺序，各调用一次 BF16 full attention，其他参数与 E018 一致。完整valid输出先保存，再核对各自SHA、与E018及彼此的变化元素数/max/RMS/相对能量。True一臂另保存完整LSE。两次原生attention、0DiT；加上失败调用累计最多3次。

代码新文件 E019_lse_diagnostic.py；CPU检查和诊断源在执行前冻结。GPU5稳定空闲检查，外部300秒超时及内部deadline；大数据沿用/data1实验目录下新的lse_diagnostic子目录。不得覆盖旧失败输出/冻结源码。不因结果自动扩大调用。False若仍不能复现旧输出，停止并继续定位；若False exact而True不同，报告明确的模板路径差异，并据差异量级决定是否另立新协议。
