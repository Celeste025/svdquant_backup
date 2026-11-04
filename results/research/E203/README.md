# E203：融合首步 INT8 softmax 概率均值捕获

新增实验模式 `AttentionRouteCache(capture_mode="softmax_mean_fused")`；H3 实验配置可设置 `route_capture_mode: softmax_mean_fused`，同时启用 Q128 和首步路由复用。默认仍为原来的 QK 最大值模式。

实现保持逻辑 Q128×K128 路由。独立编译的 D128/Q128 CUDA kernel 在 dense INT8 attention 原 QK 主循环中计算每行每个物理 K64 的局部 logsumexp，输出整行最终 logsumexp。一个 Triton 归约 kernel 使用这两项统计还原概率总量、合并相邻 K64 并除以有效元素数，生成 tile 概率均值。无需额外重算 QK；需要 FP32 中间统计存储。

## 简单验证和计时

真实赛车 480p、124 帧、step0/layer25、全 56 heads，10% INT8 + 90% NVFP4、不跳块。一次预热、5 次 CUDA event 计时，首次编译不计时。

| 路径 | 首步 full API ms/层 | 后续复用 ms/层 | 捕获增量峰值 GB |
|---|---:|---:|---:|
| 原 QK 最大值捕获 | 31.78 | 28.28 | 1.819 |
| 两遍额外 QK 概率捕获 | 91.61 | 28.55 | 1.819 |
| 融合概率统计 | 45.70 | 28.42 | 2.947 |

- 融合概率捕获比两遍版本的首步 full API 耗时减少约 50.1%；仍比原 QK 最大值多约 13.9 ms/层。
- 首步 attention 输出逐位一致。概率均值与两遍实现对照通过，最大绝对误差约 1.91e-6（未乘路由统一缩放 1024 前）。本输入实际缓存路由的全部指纹与两遍版完全一致。
- 峰值显存增加约 1.13 GB，来自局部统计等中间量；路由缓存大小相同。真实峰值取决于长度和模型其余显存占用。
- 仅验证这一组真实输入；未重跑视频，也未验证 768p 或长序列的耗时及显存，不据此宣称普遍加速。

## 编译与运行

`bash scripts/research/launch_e203.sh` 编译独立扩展并做小范围验证；`launch_e203_full.sh` 跑 full API 对照（报告文件存在时须另指定输出）。工具链复用现有 CUDA 12.8 与 Ninja，扩展首次调用时 JIT 编译并缓存。

运行源码位于被主仓库忽略的独立 Anemoi checkout，本目录 source 保留相关源码快照。`.cu.txt` 与 `.cuh.txt` 是 CUDA 源码快照，避免被主仓库忽略规则漏掉。
