# E200：首步 INT8 softmax 块均值捕获（实验版）

在 SM120 Q128 路径支持 `AttentionRouteCache(capture_mode="softmax_mean")`。首步仍运行原有 dense INT8 attention；之后使用已量化的 Q/K 与对应缩放，以两遍 Triton QK 计算每行归一化和视频 tile 的概率均值，生成并缓存现有 ragged 路由。后续 step 使用原有缓存复用路径。默认 `qk_max` 不变。

MiniMax-H3 实验配置可设置 `query_block_size: 128`、`reuse_first_step_route: true`、`route_capture_mode: softmax_mean`，并按原有配置提供各精度比例。当前只支持 Q128、Q/K 相同 head 数、head dimension 64 或 128，SM120。

## 验证

- 合成不规则 tile 和前缀输入与直接 FP32 softmax 对照通过；真实 480p 量化 Q/K，2 个 head、3 个 query tile 对所有 key tile 的均值最大绝对误差为 4.70e-7。
- 真实 QKV 的两种首步输出逐位相同；后续 map 复用在概率计算被禁止时仍正常执行。
- 两项 GPU 回归测试通过；MiniMax-H3 原有 28 项配置测试通过。
- 使用的是 INT8 Q/K 的概率，不是 E196 BF16 oracle 概率。FP16 路由分数统一缩放 1024 后排序，仍可能因 FP16 量化发生并列。

## 全 API 计时（毫秒/层）

| 捕获输入 | 模式 | 首步 | 后续复用 |
|---|---|---:|---:|
| 480p，124 帧，layer 25 | QK 最大值 | 31.80 | 28.46 |
| 480p，124 帧，layer 25 | softmax 均值 | 91.58 | 28.47 |
| 768p，124 帧，layer 0 | QK 最大值 | 117.67 | 96.61 |
| 768p，124 帧，layer 0 | softmax 均值 | 355.05 | 96.73 |

每组一次预热、5 次 CUDA event 计时，不含模型加载。后续计时包含当前 map 的执行；这些是不同 layer、不同输入，仅供观察开销。峰值由现有准备阶段主导，两模式测得的捕获增量峰值在 480p 约 1.82 GB、768p 约 3.65 GB，不能解读为概率统计本身的显存开销。

当前版本是**额外两遍 QK**的可用实验实现，并未融合到原有 dense attention 主循环，而且首步仍执行旧的 QK 最大值捕获，故首步耗时较高。需要优化和完整视频验证才能评价实际收益。嵌套 Anemoi 仓库在主仓库被 gitignore；本目录 `source/` 保存了相关源码快照，不能将快照误认为主运行时文件。
