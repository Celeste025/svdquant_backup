# E049 评价入口与独立归约

本轮仅评价 `wan14b_bf16` 新八条视频，保留四提示各两 seed。E049 的 81帧/480×832/16fps/50步/CFG5/shift3 直接从本轮 manifest 校验；不继承 E043 的 CFG6/shift8。评价公式复用冻结的 81帧 AMT/RAFT/DINO 实现和 MJ-VIDEO 原入口，不重跑历史模型媒体。

三个新增源：

- `scripts/research/wan14b_reference_evaluation.py`：`--phase check|prepare|temporal`。`check` 在 generation/download 尚未完成时也可运行，仅检查协议、既有评价资产存在与结构；`prepare` 要求四生成 worker 完成并绑定全部八条实际媒体，写 `evaluation_manifest.json` 及 `temporal_wan14b_bf16_manifest.json`。`temporal --check-only` 仅 CPU 全媒体检查；正式 temporal 同原模型与公式。
- `scripts/research/launch_wan14b_reference_evaluation.py`：root 在 named tmux 启动；完成 generation 后，先独立 CPU prepare / temporal check（各120秒），再检查 GPU0/1 空闲并分别启动 temporal/MJ。整体900秒 deadline，记录 PID、命令、日志、rc、结果 SHA；失败保存并停止所属进程，无自动重试、不覆盖旧结果。
- `scripts/research/summarize_wan14b_reference.py`：CUDA隐藏的独立 CPU 汇总；仅所有生成与评分 complete/rc0 后执行。读取新原始指标、八份实际初态与末态、四份复用 embedding；核实际输入 signatures、40层配置/小来源记录与来源模型身份、400 callback /800 DiT/64000 SDPA/400 scheduler/8 decode/0 TE/0 native。仅核中间标量时间表，不重新推理或重演中间轨迹，不重新哈希模型权重。

固定输出位于 `results/research/E049`：`evaluation_check.json`、`evaluation_manifest.json`、`temporal_wan14b_bf16_{manifest,cpucheck,scores}.json`、`mjvideo_scores.json`、`evaluation_launcher.json`、`evaluation_summary.json`。生成输入沿 `worker_<prompt_id>.json`：case 的 `initial_noise` 为 artifact+tensor，`embeddings` 为 E043 artifact 三字段，`actual_inputs` 保留三份实际张量签名，`final_latents` 为 artifact+tensor，`video` 为文件记录。汇总输出 `per_case`、`per_prompt`、`mean_metrics`，以及全部 MJ 五 aspects /28 criteria 原值、两种时序数组和 DINO sum；先每提示两 seed 均值，再四提示等权，无综合分或筛例。

```bash
CUDA_VISIBLE_DEVICES='' /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python scripts/research/wan14b_reference_evaluation.py --phase check
# 以下两条由 root 在完成各自前置后执行，本次准备未启动：
/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python scripts/research/launch_wan14b_reference_evaluation.py
CUDA_VISIBLE_DEVICES='' timeout 180 /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python scripts/research/summarize_wan14b_reference.py
```

正式结构检查已 complete，`cuda_initialized=false`、`model_loads=0`、`generation_files_required=false`；三源 py_compile 和 summary `--help` 通过，尚未运行正式评价或归约。AMT40项/RAFT40 flows/DINO80项，MJ索引固定0/10/20/30/40/50/60/70。低运动/模糊也可能得到高平滑度，分数不直接证明动作完成。该8例为已观察诊断集；与1.3B同时改变模型及CFG/schedule，不作尺寸或量化单因素因果判断，也无语义阈值准入。
