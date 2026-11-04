# E205：复用在线 softmax 指数权重优化概率捕获

将 E203 融合概率统计从 update_mdo 前移到后面、P 转 FP8 前。复用已计算的指数权重和每行当前最大值，归约得到局部 logsumexp；补偿 S_FP8_OFFSET 并保留预取前的 K64 stage 索引。保持 Q128×K128 路由及原来的 softmax_mean_fused 模式。

只做同一组真实 480p、124 帧、step0 layer25、全 56 heads 的简单校验和计时；没有重跑视频。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 主 capture kernel，非 profiler events | 40.50 ms | 30.81 ms |
| 最终归约 | 1.32 ms | 1.32 ms |
| 首步 full API | 45.70 ms | 36.06 ms |
| 寄存器 spill requests（Nsight） | 475,465,984 | 167,715,968 |
| registers/thread | 255 | 255 |

Full API 优化前来自 E203/full_api.json；主 kernel 前后来自 E204/E205 分段 events，旧 QK-max 对照同轮稳定在约 28.2ms。首步 full API 缩短约 21.1%，spill 请求数减少约 64.7%。Nsight replay 的 duration 不与非 profiler events 混用。

同轮原 QK-max full API 首步 31.78ms，额外两遍概率版 91.64ms，优化融合版 36.06ms；优化融合版后续复用 28.50ms。局部统计缓冲区格式和容量未变，峰值显存问题仍存在。

## 简单正确性结果

- 独立核函数 attention 输出相对原 kernel NMSE 1.21e-11；full API 首步 NMSE 9.50e-11。由于编译调度变化不再逐位一致，误差极小。
- 与两遍概率参考的 tile 分数对照通过；最大绝对差约 3.81e-6（路由统一缩放 1024 前）。
- 本输入缓存路由的全部指纹与两遍概率版一致，后续复用输出 NMSE 为 0。

仍有 spill，后续可针对寄存器存活区间继续分析。当前结果不说明其他层、长视频或 768p 一定具有同样收益；此轮没有扩大验证范围。

运行源码在独立 Anemoi checkout，本目录 source 保存了主仓库可追踪快照；E203/E204 旧结果保持原样。
