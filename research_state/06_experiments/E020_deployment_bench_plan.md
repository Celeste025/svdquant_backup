# E020 补充：独立进程部署成本

2026-10-03，训练已完成，计时前固定。GPU5新进程、原开发验证0001/step0完整输入，BF16、现有SVD nativefast、plain_step0、QAD_step64四臂。每臂fresh model，2次预热＋5次同步wall计时，共28DiT。固定BF16 FLASH attention与同一输入，不含模型加载、D2H/hash及flags检查；另列包含flags检查的wall。加载后重置allocated/reserved峰值；常驻存储包括SVD hook拥有的LR/smooth；每臂清理模型引用后验证释放。此处无训练图/optimizer，不能与训练峰值混用。SVD是整套旧配方（含非目标INT4）的参照，不归因为纯LR消融。20分钟/60GiB，输出E020/deployment_bench.json，不生成视频，不代表整段生成加速或质量等价。
