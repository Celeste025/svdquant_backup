# E046 同轮两臂评价接口

新入口 `scripts/research/wan_terminal_repair_evaluation.py` 复用 E043/E044 已运行的 81 帧 AMT/RAFT/DINO loop；MJ 仍为原 `eval_mjvideo_e021.py`。两臂是 `native_full` / `bf16_last`，8 case × 2 = 16 新媒体全部重新评分，历史 E043/E044 结果只绑定为参照。无新增指标、资产下载或样本筛除。

`--phase check` 已在现有 MJ 环境 CPU 完成，结果 `results/research/E046/evaluation_check.json`。生成未完成前不运行 prepare。全部生成完成后，评价 launcher：

```bash
/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python -u scripts/research/launch_wan_terminal_repair_evaluation.py
```

由 root 在 repo cwd 的命名 tmux 中启动。它依次运行 CPU prepare、两臂 `--phase temporal --arm ARM --check-only`，完整解码 16 视频；确认 GPU0/1/5 空闲后，0/1 分别运行两臂 temporal，5 运行 MJ。总外限 900 秒、进程组清理、失败保留，无自动重试。MJ 原模板 `{variant}/{case_id}/video.mp4`，一次 8 case × 2 variants；不使用 `--only-complete` 或 case cap。

产物：`evaluation_manifest.json` 的 cases 为八个固定身份、rows 为16媒体及原noise/embedding绑定；两个 `temporal_ARM_manifest.json` / `temporal_ARM_scores.json`；一个 `mjvideo_scores.json`；`evaluation_launcher.json` 记录真实三个评价进程。输出路径 `results/research/E046`，媒体仍在 DATA1/E046/{variant}/{case_id}。

采样保持原算法：MJ `[0,10,20,30,40,50,60,70]`；AMT 41 偶帧输入/40 奇帧参考；RAFT 41帧/40 flow；DINO81帧/80项。不改视频/补帧；AMT内部自适应 scale 沿旧算法并记录实际值。

全部评价完成后，CPU readout：

```bash
timeout 180s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python \
  scripts/research/summarize_wan_terminal_repair.py
```

默认 `evaluation_summary.json`。复算保存的 AMT/RAFT 数组和 DINO 总和，报告同轮两臂七主项、每case/每prompt双seed/总体配对差分；MJ安全与偏差仅保留raw。历史 E043 BF16 为质量上下文；实际 E044 SVD final 与本轮 native final 的FP64差是复现/漂移描述，无逐位科学门槛，不替换本轮baseline，也不计算不同完整轨迹的native–BF16 NMSE来冒充质量。

执行核对包括实际102次DiT收据/每case、300原native模块、50步记录、保存的pre-last sample/cond/uncond/timestep/原embedding、复制前history位置与sigma、真实独立BF16输出、16终态/媒体。原worker明确复用同一已捕获输入并从同一保存history恢复scheduler；未另存第二份BF16调用输入/history，CPU summary不把源码复用合同说成第二份独立tensor追踪或数值scheduler重放。

备用BF16模型每worker一次的CPU加载/H2D秒数、参数/缓冲字节、allocated/reserved增量和每case阶段峰值原样汇总。第二seed的native阶段已有teacher驻留也保留；这些含诊断的双模型控制成本不构成成熟部署benchmark，新增2次DiT不能换算为2%总延迟。
