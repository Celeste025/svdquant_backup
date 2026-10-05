# E048 — 匹配校准的 Wan native 完整生成基线

2026-10-03。E047 校准完成后，使用其唯一固定 checkpoint 在既有八个诊断样例上完成生成。此项是成熟 SVDQuant 基线补全，不提出新方法。

## 输入与执行

- 完整继承 E043 四提示 × 两 seed 的实际初始 FP32 noise、BF16 正/负 embedding 和身份。直接加载已保存 tensor，不重新抽噪声或编码文本。
- 同 E044 v2：原版 Wan 1.3B，81 帧 480×832、16 fps，UniPC 50 步、CFG 6、flow_shift 8；FP32 sampler/VAE，BF16 FlashSDPA，300 个 native NVFP4 线性层与既有 smooth/LR 配方。仅 checkpoint 换为 E047 匹配校准产物。
- 薄 wrapper 直接调用冻结 E044 v2 的 prerequisites、run（其中安装仍为原 install_model）。标准 arm/variant 保留 `svdquant_nvfp4`，以 experiment=E048、独立目录与 checkpoint receipt 区分旧基线。
- 每个 prompt worker 保存两条完整生成：实际初态/embedding 引用、50 步标量、100 次 DiT 内核计数、final latent、81 帧 MP4 和固定九帧图。DATA1/E048/svdquant_nvfp4/{case_id}，不覆盖旧实验。

## 前置状态与预算

任何权重检查或生成均要求 `results/research/E047/ptq_run.json` 为 complete，并绑定其 `checkpoint_identity.json`；五文件身份检查沿用 E044 原入口，不重复构建模型 hash 树。本次准备只进行 Python 语法编译；E047 完成后才运行隐藏 CUDA 的全八例 `--check-only`，然后由 root 启动 named tmux launcher。

四 worker 各一个 prompt：161→GPU0、192→GPU1、269→GPU5，316 排队 GPU0。单 worker 1800 秒，launcher 3600 秒；仅空闲卡启动，失败保留并停止自身进程组，不重试。共 8 视频、800 DiT、240000 native GEMM/fastpack checks、48000 DiT SDPA、400 scheduler、8 VAE decode、0 TE。完整 wall time 含加载与输出，非性能 benchmark。

## 读出边界

后续仅评价新增八媒体，E043 BF16、E044 旧 SVD 与 E046 bf16_last 继承已有完整媒体和评分，不重新推理。本文件不实现评价适配。全部八例保留，以逐例原始视觉与 MJ 四项为主、AMT/RAFT/DINO 为辅助，安全/偏差题 raw-only。八例已经用于诊断，不是新 held-out 测试；帧长与采样 schedule 同时改变，不能把改善归因于单一因素。无改善也不证明所有 NVFP4 配方失败。

固定协议：[manifest](E048_matched_wan_native_manifest.json)。
