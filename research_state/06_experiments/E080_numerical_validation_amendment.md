# E080 数值验证修正（不改变方法或决策门槛）

首次局部运行在第一个case的首个condition结束时，FP32 online/dense maxabs=0.000353813超过实现中3e-4阈值，保留pv_probe.json/log，状态failed_stop；没有完成科学比较。耗时3.6805秒，峰值14,178,287,616 bytes。

新probe_h3_pv_contrast_v2.py增加独立FP64 softmax在线递推（包括max/exp/denominator/output全部double）对同logits的FP64 dense参考，maxabs要求<1e-8。所有量化分支仍沿原FP32模拟执行，原FP32在线偏差继续报告但不拿它判断公式错误。参考输出用FP64 softmax/PV计算后转FP32。原来的10%/5%科学门槛、同bit/scale预算、对照、采样、模拟gap限制、总资源上限保持不变。加上实际视频grid/长度/shape显式断言。此修正验证高精度数学，未声称原生数值等价，也没有放宽科学筛选标准。
