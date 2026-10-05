# De-biasing Diffusion：方法补缺访问审计

2026-10-04。目标是补齐 [temporal_error_prior_work](temporal_error_prior_work.md) 中 OpenReview `nExUJBF5tR` 的实际方法：权重 stochastic rounding 是否每次 forward 重采样、partial SR 的选择/执行规则、存储/计算合同，以及同轨迹跨 step 负相关。使用 claim-prior-work-triangulation 的逐项覆盖原则。本轮没有 GPU、模型前向、实验代码或根 state 修改。

**结果：仍未取得官方全文；到此停止，不把摘要补写成方法。证据等级仍为 C（官方论文被搜索索引的摘要），没有升级为 A。**

## 实际访问记录

| 官方入口 | 本轮结果 |
|---|---|
| [OpenReview forum](https://openreview.net/forum?id=nExUJBF5tR) | web 返回 browser verification 页面，无论文内容 |
| [PDF id 入口](https://openreview.net/pdf?id=nExUJBF5tR) | web 返回同一验证页；从搜索结果再打开也相同 |
| [固定 PDF 路径](https://openreview.net/pdf/ef26b54e0d9905ba01ede3c62d26e99cc1749b7c.pdf) | web 返回验证页；普通 urllib 请求 HTTP 403 |
| [附件接口](https://openreview.net/attachment?id=nExUJBF5tR&name=pdf) | web 内部错误；普通 urllib 请求 HTTP 403 |
| [OpenReview API v2 note](https://api2.openreview.net/notes?id=nExUJBF5tR)；[旧 API note](https://api.openreview.net/notes?id=nExUJBF5tR) | urllib 均 HTTP 403；v2 web 也未取得内容 |
| [Daniel Soudry Lab publications](https://soudry.github.io/publications/) | 页面可读，查 `biasing` 无匹配；没有取得本论文替代全文链接。这不是论文不存在/作者身份/录用状态的证据 |

共 3 次定向搜索：`"De-biasing Diffusion" "Data-Free"`、`"Yaniv Blumenfeld" "De-biasing" quantization pdf`、`"nExUJBF5tR" "stochastic" weights forward`。查询得到官方 PDF 的摘要索引、第三方聚合页及指回 OpenReview 的论文清单，没有得到作者公开的另一份全文或方法代码。第三方页只用于定位，未作为方法证据；没有登录、联系作者或继续扩展搜索。

## 四项核查结果

| 待核事项 | 本轮能确认什么 | 不能推断什么 |
|---|---|---|
| 每次 forward/每个 denoising step 是否重采样权重 | **未核实** | SR 可以一次离线抽样，也可以在线重采样；论文标题及摘要不足以选择其中一种。也不知道 CFG 两分支是否共享随机数 |
| partial SR 细节 | 索引摘要明确提出 partial stochastic-rounding of weights | **未核实** partial 指部分层、部分元素、部分 mantissa、部分步骤还是其它安排；不知道筛选规则、概率或更新频率 |
| 权重存储与计算合同 | 论文讨论 FP8 量化 | **未核实** 是否保留高精度 master、概率/residual metadata、几份低位副本、在线重编码；不知道真实 GEMM dtype、实现设备、额外带宽或 native kernel 开销。摘要的成本表述不是这些问题的答案 |
| 跨 step 负相关/互补配对 | **未核实** | 去 bias、随机舍入、步数更多时终图偏差下降均不逻辑蕴含 antithetic rounding、互补两权重、跨步负相关或传播加权设计；同时全文未读也不能宣称它没有做 |

## 对当前先例判断的影响

旧记录的宽泛碰撞保留：**降低量化 bias、以 SR 改善 diffusion 已被摘要明确提出，不能声称该上位动机全新。** 但本轮不新增“作者每步重采样”“partial 是某种确定方案”“只有独立噪声所以没有负相关”等方法事实。

“同轨迹两套原生 NVFP4 权重按传播/solver 系数设计跨步负相关”对这篇论文的精确覆盖状态继续为 **unverified**。既不能因此自动 kill，也不能利用未能访问全文而宣布有残余创新。研究方若要推进该具体方法，仍需实际原文或官方代码才能消除这项碰撞不确定性；本审计不授权恢复已停止实验。

停止点明确：访问受限且本轮查询额度已用完，不继续用关键词片段拼接整篇论文，不要求用户批准额外搜索。本次只新增该记录，不覆盖先前文献笔记。
