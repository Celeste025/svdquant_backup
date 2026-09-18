# New-server recovery handoff prompt

Give the following prompt to an agent on a new server. It verifies every
published checkpoint can be downloaded, loaded, and used for a minimal output;
it intentionally does not attempt an exact PTQ rerun.

```text
在新服务器恢复并验收 SVDQuant 工作区：

1. 克隆并初始化：
git clone --recurse-submodules https://github.com/Celeste025/svdquant_backup.git svdquant-exp
cd svdquant-exp && git submodule update --init --recursive
设置 SVDQUANT_DATA_ROOT 到大容量数据盘。

2. 阅读 README.md、docs/environment_matrix.md、docs/modelscope_release.md，按文档创建两个核心环境：
- PTQ/rCM/FLUX/VBench
- MiniMax-H3 generation
MJVideo、ConvRot 暂不配置。

3. 合规下载/准备 README 中列出的上游基模：Wan+rCM transformer、MiniMax-H3、FLUX dev/schnell。

4. 用 tools/fetch_published_artifact.py 下载并校验全部量化 checkpoint：
- rcm-wan: real-nvfp4-g5-r32, int4-g10-r32, real-nvfp4-g20-r32, real-nvfp4-g10-r64
- flux1: dev-int4-r32, schnell-int4-r32
- minimax-h3: nvfp4-g10-r32, nvfp4-g10-r64（需要有权限的 MODELSCOPE_API_TOKEN）

5. 用 tools/fetch_published_video_dataset.py 下载并校验 rCM 与 H3 的正式视频集；H3 数据集同样需要 token。确认 metadata/videos.jsonl 可用于映射视频到 case/variant/prompt。

6. 每个 GPU 任务放进独立 tmux。逐个运行 smoke：
- rCM 四个 variant：scripts/infer_rcm_wan_4step.py，固定 480x832、77 frames、4 steps、固定 seed，并传入对应 --quant-ckpt。
- H3：先 minimax_h3_bf16_smoke.py；再对 r32/r64 各用 run_minimax_h3_vbench51.py --smoke --variants svdquant 跑一个 case。
- FLUX：按 registry 所列 loader 对 dev/schnell 各生成一张确定性量化图片。
确认每项的 SHA、量化状态加载、输出文件和日志均正常。

7. 最后只跑一个 VBench case smoke，确认评测环境正常；不要重跑全量评测或 PTQ。

最终横向表格汇报：每个模型的下载校验、基模状态、smoke 状态、输出路径、GPU/显存/耗时和失败原因。不要提交 token、模型、视频或缓存。
```

## Scope boundary

This validates published-artifact inference and evaluation reproducibility.
Exact PTQ reproduction additionally requires the original 64-sample
calibration cache and the recorded CUDA/PyTorch runtime. MiniMax-H3 model and
video repositories are private; the receiving account needs explicit
ModelScope read access.
