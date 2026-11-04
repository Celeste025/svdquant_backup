# E208：Q64×K64 概率 map 与实际计算粒度对齐

新增实验路径：`AttentionRouteCache(capture_mode="softmax_mean_fused_k64")`，配合 `SparseConfig(query_block_size=64)` 使用 Q64×K64；配合 query_block_size=128 使用 Q128×K64。原 softmax_mean_fused 保持 Q128×K128。H3 实验配置同样支持 `route_capture_mode: softmax_mean_fused_k64` 和 query_block_size 64/128，同时需启用 reuse_first_step_route。

K64 概率统计复用原 dense INT8 指数权重及最终归一化信息，独立为每个物理 K64 计算均值，不合并两个 K64。原生 CUB 路由扩展支持矩形 map；缓存保存物理 video K64 IDs，物化时只插入前缀而不把每个 ID 再展开为两个 K64。完全为空的物理 K64 会被排除在候选和预算之外；当前样本 Q128 布局实际无空半块，此分支为边缘形状准备。

## 同轮 full API 比较

真实赛车 480p、124 帧、step0 layer25、56 heads，10% INT8 + 90% NVFP4，不跳块。2 次预热，8 次 CUDA event 计时，排除首次编译。以下缓存大小为十进制 MB/层。

| map/计算粒度 | 首步 ms/层 | 复用 ms/层 | cache MB/层 | map 行×有效列 |
|---|---:|---:|---:|---:|
| Q128×K128 | 35.46 | 28.55 | 2.59 | 148×148 |
| Q128×K64 | 36.44 | 28.61 | 5.04 | 148×296 |
| Q64×K64 | 34.31 | 26.26 | 7.75 | 259×259 |

该样本 Q64×K64 比 Q128×K128 首步下降约 3.2%，复用下降约 8.0%；缓存约 3 倍。Q128×K64 首步略慢，复用基本相同。

Q64/Q128 采用各自原有的 ragged 2D 空间划分，Q64 划分减少 padding，所以性能变化不只是寄存器差异。Q 的量化分组和路由块形状也变了；不能把本轮当成相同 map 的纯 kernel 替换。未测视频画质，未对 Q64 的 spill 作硬件采样，不能宣称 spill 消失。

## 简单校验

三种路径的首步输出相对各自相同 Q 粒度的原生 dense INT8 输出 NMSE 均小于 1e-8；复用输出有限且正常。小矩形 map 的原生排序与显式全局排序一致；真实 Q64 INT8 softmax 概率均值抽查 2 heads、3 query tiles、所有 key tiles，最大绝对误差 4.66e-7。没有重跑完整视频。

运行入口：scripts/research/launch_e208.sh；简单对照：launch_e208_check.sh。旧实验结果未覆盖，运行代码位于独立 Anemoi checkout，本目录 source 保存相关源码快照。
