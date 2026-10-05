# E044 薄评价接口

`scripts/research/vanilla_wan_native_evaluation.py` 仅更换病例/媒体绑定与报告标签，实际 AMT/RAFT/DINO 执行仍为已运行 E043 所复用的 `evaluate_fastwan_qad_temporal.py`；MJ 沿用 `eval_mjvideo_e021.py`。不修改旧源、不重新评价 BF16、不挑样本。

- `--phase check`：GPU 前 CPU 结构检查，不要求新媒体存在。
- `--phase prepare`：所有量化 worker 完成后，按 launcher 的实际 worker 记录取文件，精确覆盖两臂各八例；生成 `results/research/E044/evaluation_manifest.json`，其 `cases` 是固定八个身份、`rows` 为 BF16/两量化臂共 24 行。每行绑定实际视频 SHA 与 E043 noise/embedding artifact。
- 同时生成 `temporal_plain_nvfp4_manifest.json`、`temporal_svdquant_nvfp4_manifest.json`，各八例，次序固定为原 prompt/replica 顺序。
- `--phase temporal --arm ARM --check-only`：CPU 完整解码该臂八媒体，输出 `temporal_ARM_cpucheck.json`。去掉 `--check-only` 并提供 `--deadline-unix` 后才运行 GPU 指标，输出 `temporal_ARM_scores.json`。

在 repo cwd 下，CPU 阶段使用 `CUDA_VISIBLE_DEVICES=''`，Python 为 `/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python`：

```bash
python scripts/research/vanilla_wan_native_evaluation.py --phase prepare
python scripts/research/vanilla_wan_native_evaluation.py --phase temporal --arm plain_nvfp4 --check-only
python scripts/research/vanilla_wan_native_evaluation.py --phase temporal --arm svdquant_nvfp4 --check-only
```

GPU 评价由 root 单独启动，沿原环境、离线资产与 deadline 监督。两次 temporal 的 CLI 为上述命令去掉 `--check-only`、加 `--deadline-unix UNIX`。MJ 一次读 8 case × 2 variants 共 16 个新视频：

```bash
python scripts/research/eval_mjvideo_e021.py \
  --manifest results/research/E044/evaluation_manifest.json \
  --samples /data1/models/svdquant-wjq/research/20261003/E044 \
  --model /data1/models/svdquant-wjq/models/MJ-VIDEO-2B \
  --tokenizer /data1/models/svdquant-wjq/research/20261003/E021/tokenizer \
  --mjvideo-repo /data1/models/svdquant-wjq/third_party/MJ-Video \
  --output results/research/E044/mjvideo_scores.json \
  --variants plain_nvfp4 svdquant_nvfp4 \
  --video-template '{variant}/{case_id}/video.mp4' --num-segments 8
```

81 帧媒体不增补帧；480×832 为 AMT16/RAFT8 的整数倍。MJ 实际索引固定 `[0,10,20,30,40,50,60,70]`；AMT 41 个偶数输入帧/40 个奇数参考，RAFT 41 个采样帧/40 flow，DINO 81 帧/80 项。AMT 内部自适应 scale 沿原算法并记录实际值，不新增 resize 规则。

全部新评价完成后，CPU 运行 `summarize_vanilla_wan_native.py`，默认写 `results/research/E044/evaluation_summary.json`：绑定 E043 已完成分数，核 24 媒体、16 同输入引用、16 新 FP32 final 与 800 步标量链、1600 个 DiT 调用收据/300 模块覆盖；复算新 AMT/RAFT 数组与 DINO 已保存总和归一化，保留 MJ28/5 原始输出。按每 case、每 prompt 两 seed、全体给七主指标和三种配对差分；安全/偏差仅逐例 raw。初终自由轨迹差不解释为局部量化扰动，中间 tensor 未保存因而不声称重放。SVD 的旧 smooth/LR/非目标 PTQ 与旧校准域差异均属于整套配方比较。

本轮仅结构 check/编译/help 完成；prepare、媒体 CPUcheck、GPU 指标及最终汇总等待正式产物。
