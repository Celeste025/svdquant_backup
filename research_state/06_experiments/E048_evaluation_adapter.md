# E048 新增八视频评价接口

`wan_matched_native_evaluation.py` 只绑定本轮八个 `svdquant_nvfp4` 媒体，直接调用已运行的 81 帧 AMT/RAFT/DINO loop；MJ 仍用冻结 `eval_mjvideo_e021.py`。E043 BF16、E044 旧 SVD、E046 bf16_last 的媒体与完整评分仅作为历史参照，不重新生成/评分。无新增指标、安装、下载或样本筛除，安全与偏差题 raw-only。

准备阶段只在隐藏 CUDA 的现有 MJ 环境运行 `--phase check`，不要求生成文件、不加载模型。正式 `prepare` 必须看到 generation launcher complete、所有实际 worker rc0/complete，固定八媒体存在并与 generation SHA/81 帧 metadata 对应，且 initial_noise/embeddings 为 E043 实际引用。只有全部产物通过才写 `evaluation_manifest.json` 和 `temporal_svdquant_nvfp4_manifest.json`。

生成完成后由 root 在 repo cwd 的命名 tmux 启动：

```bash
/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python -u scripts/research/launch_wan_matched_native_evaluation.py
```

launcher 先做 CPU prepare 与完整八媒体 `temporal --check-only` 软件解码，再复核 GPU0/1 空闲；GPU0 跑 temporal，GPU1 跑 MJ。总外限 900 秒，各 CPU 阶段 120 秒，保留失败与日志，无自动重试，只清理本 launcher 的 GPU 子进程组。`results/research/E048/evaluation_launcher.json` 保存两 evaluator 的实际 PID/GPU、命令、returncode、各进程 wall 秒数和总 wall 秒数；这些含模型加载与检查的耗时不是部署 benchmark，实际值运行后填写，不预估为测量结果。

采样不变：MJ 八段 `[0,10,20,30,40,50,60,70]`（模板 `{variant}/{case_id}/video.mp4`）；AMT 41 偶帧输入/40 奇帧参考，RAFT 41 帧/40 flow，DINO 全 81 帧/80项。480×832 无媒体补帧/补边，AMT 内部实际 scale 沿原算法记录。新分数为 `temporal_svdquant_nvfp4_scores.json` 和 `mjvideo_scores.json`，都是八 case × 单 arm。CPU summary 另由 audit 实现，逐 case/四 prompt/均值报告七主项及历史差分，不由局部 latent 误差或均值单独声称等质量。

CPU 独立汇总执行入口改用 `scripts/research/summarize_matched_wan_native_v2.py`（SHA `0a3fb425408704745ef64255f78288ff18983ff741b0e4c2fbb253acf6b279bb`）。冻结 v1 保留未执行；v2 仅将 E047 PTQ identity 的扩展文件记录与 worker 简版记录按共同 `file/bytes/sha256` 字段及同一解析路径比较，其他完整记录比较不变。已实际 CPU 读取 E043/E044/E046 三份小 summary，全部八 case 的 identity、metrics/video/media 以及所引用 E043 noise/schedule 字段均存在并对齐；未运行尚未产生的 E048 数据或任何 GPU。
