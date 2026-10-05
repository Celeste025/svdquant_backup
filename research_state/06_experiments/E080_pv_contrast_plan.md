# E080 — C007 PV双侧差分表示：有限局部screen

2026-10-04，用户已授权验证想法1。模式exploration，0训练/校准/新kernel；不把局部误差当画质。

问题：相同bit/scale预算下，量化前保留P差别是否能恢复真实H3局部输出对比，收益是否超过仅V中心化与普通重排？竞争解释为V异常值、普通分组改善、QK上游已失真。

采集：E079拍手r0/r1原始prepare和全部原生路线。固定step14，blocks0/24，各完整56heads QKV及原attention output；两轨迹各20完整DiT，总40，最终video/audio latent必须逐tensor SHA精确复现E079。GPU0/1各900秒上限、峰60GiB，DATA最多8GiB。生成0新视频/VAE。

局部：所有56heads分批处理；视频query取grid[37,18,32]上f=(0,12,24,36)、h=(4,9,13)、w=(7,15,23)及w+1，共36对/72query；完整有效keys，保留原始128-query均值。固定原QK与真实pack后QK两条诊断，P模拟保留逆序128key online softmax、2688预放大、未量化normalizer及P/V不同scale舍入。与capture官方output报告模拟gap，绝不称bit-exact全attention模拟；gap若超过base对原attention误差能量5%，不解释official接近性，仅报告isolated PV。

方法：每个完全属于video且32对齐的原块，P用半和/半差、V用和/差；16和16差分别group16，仍32FP4值/2E4M3 scale。其它块原样，允许少量跨图像行pair并记录比例，不称所有pair物理邻居；所有方法保持原normalizer。P差用signed absmax，避免对称sqrt2变换使P scale溢出。

对照：base、P-only变换量化后逆变换+原V量化、V-only中心化（32和128，可带均值存储/补偿额外开销，不冒称完整VC clustering）、双侧pair、相同32块固定随机pair（seed8007）、同块只重排不变换（同预算）；另精确PV上限。精确QK是来源隔离而非native baseline。

数值门槛：先检查官方QKV pack字节重编码相等与dequant相等、pair FP32恒等式、完整高精度online累积对dense attention误差。独立CPU复算保存输出的SSE/邻居差分SSE。

决策门槛（预先）：block24两seed的差分误差相对base至少下降10%，且相对最好V-only/P-only/随机pair/重排对照再下降5%；block0不反向增加>5%，普通输出SSE不增>5%，两QK条件方向相容，才进入有限视频干预。否则停止本候选当前表示，不扫group/step/block/head。更低SSE而不满足强对照只算表示工程现象。Heads/query不是独立统计样本。

局部GPU单进程最多1200秒；若实现或数值验证失败，保留失败记录，不能把失败记作机制阴性。后续视频另存新实验协议，首轮不承诺吞吐。C008本轮不执行。
