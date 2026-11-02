# MiniMax-H3 / DraftAttention2 工作交接

更新时间：2026-11-02。面向在其他机器或新会话接手的 agent；先读本文件和仓库根目录 `AGENTS.md`，再根据具体问题查看原始报告。

## 仓库和文件

- GitHub：<https://github.com/Celeste025/svdquant_backup>；近期工作在 `operator_exp` 分支，基准提交 `99e819d`。先确认当前分支与后续提交，勿假定 `main` 包含这些工作。
- 代码在 `scripts/`、`scripts/research/`；可读结果在 `results/research/`，实验计划与决策在 `research_state/`。项目自带 `third_party/deepcompressor` 为本地修改过的代码，不要用上游覆盖。
- 模型、checkpoint、QKV、生成视频、日志和编译产物没有提交。原机器共享数据根目录是 `/data1/models/svdquant-wjq`；prompt 表是 `/home/wjq/workspace/178866172854036`（JSONL，未跟踪）。按 `prompt_id` 查原文，不要从文件名推断 prompt。
- Git 克隆不是完整运行环境。历史脚本含原机器绝对路径；新环境先修改或配置路径，准备模型、数据、Python/CUDA 和 GPU，勿直接运行历史批量脚本。

## 近期工作与结论

1. **首尾 step 保护**：E134 在赛车和游泳两个 case 中，SVDQuant 推理时前 10% step 保持 BF16，比仅保护后 10% 更接近 BF16 参考；效应因 case 不同。E136 比较首尾 BF16、W8A8 INT8 等；INT8 是 K128 分组 Q/DQ、BF16 GEMM 数值模拟，不是原生 INT8 速度。报告：`results/research/E134/README.md`、`results/research/E136/README.md`。
2. **DraftAttention2 预算**：E139–E148 包含不同稀疏率、INT8/NVFP4 比例及首步保护实验。`results/research/E142/tradeoff.png` 是真实 QKV 算子的延迟/误差图，不能直接当成整段视频加速；查看 `results/research/E139/README.md`、`E140/README.md`、`E142/README.md` 和 `E144`–`E148` 的数值报告。
3. **首步捕获与复用 map**：E154 在每层首步以稠密 INT8 QK 捕获 tile 最大分数，选路并缓存，后续复用，减少 DraftMap 构建/排序。E159–E162 测了每步生成、首步一直复用、每 5 步刷新；质量随 case 变化，不能假定首步路由一直有效。详见 `results/research/E154/summary.md`、`E159/summary.md`、`E160/summary.json`、`E161/summary.json`。E188/E195 是算子级计时，不能直接当作整段视频加速。
4. **七组联合配置**：两批各 4 个 case × 7 组均已完成。配置分别在 `results/research/h3_r64_draft_schedules/plan.json` 和 `results/research/h3_r64_draft_secondset/plan.json`，结果看各自的 `summary.csv` / `summary.json` 与每组的 `metrics.json`。注意第二批赛车 p016 使用 5 秒、576×1024 的 E124 BF16 参考，其余用 E150 768×1344；不同分辨率与 5/10 秒时长不要直接混成速度结论。`plan.json` 的 `queued` 是启动时快照，完成状态以 `summary.json` 为准。
5. **原版 DraftMap 命中率**：`results/research/draftmap_maxprob_20261102/README.md` 比较同一步前 10/20/30/50% 视频块与三个 oracle：块内概率质量、最大 softmax 概率和最大 QK logit。赛车 p016、480×832、20 steps、5 层、8 个固定 heads；单 prompt 结果和探索性统计不能外推总体。它衡量同一步选块准确性，不直接证明跨步 map 复用有效。

## 实现与关键限制

- 改版 Anemoi 不直接作为嵌套仓库提交。按 `third_party/patches/README.md` 获取指定上游 commit 并应用 `third_party/patches/anemoi-local.patch`；补丁包含首步捕获/复用和本地算子优化。重建 CUDA 扩展；不要假定其他机器有现成 `.so`。MixAttention 等的基准版本和补丁也见该文档。
- 七组实验入口：`scripts/research/run_h3_r64_draft_experiment.py`、`run_h3_r64_draft_secondset.py`；同目录有 `launch_...`、`queue_...`、`summarize_...`。先审阅绝对路径、输入、任务规模和输出目录。
- rank-64 SVDQuant 的 NVFP4 主 GEMM 与低秩 up 采用 SM120 FlashInfer CuTe DSL 融合路径，低秩 down 是 BF16 GEMM；普通 W4A4 使用原生 NVFP4 GEMM。第 5 组 W8A8 是 `scripts/research/h3_w8a8_group128.py` 的真实 K128 INT8 GEMM，但没有证明性能最优。DraftAttention2 INT8 路径为 INT8 QK + FP8 PV，不是整个模型统一 INT8。
- 七组视频测试为适应显存在 BF16/NVFP4 阶段切换 CPU/GPU 权重；传输和可能的首次编译计入 timestep 耗时。长序列 Sage3 按 8 个 heads 分批。结果是单次运行，不可把不同长度、分辨率、不同轮次的耗时混为稳定加速比。
- 选块实验入口：`scripts/research/analyze_draftmap_hit_rate.py`、`analyze_draftmap_maxprob.py` 及对应 `summarize_...`。Q128 下一个 block 是最多 128×128 的 query/key 区域，不是单个 key；块内最大 QK 与最大 softmax 概率、概率质量是三个不同排序目标。首步捕获的是 INT8 QK tile 最大值，BF16 QK 实验只是近似对照。

## 运行和版本控制

原机器主要模型是 MiniMax-H3；GPU 为 RTX PRO 5000 Blackwell（SM120）。常用 Python 路径及其他模型环境见 `AGENTS.md`；FlashInfer、CuTe DSL 和 Anemoi 需要核对实际版本、架构和编译路径。运行长任务前检查 `nvidia-smi`、磁盘和现有任务，使用具名 `tmux` 并将日志写入 `results/logs/`。

不要声称 Q/DQ 模拟为真实低精度 GEMM，也不要把相对 BF16 的 LPIPS/NMSE 当作主观画质。提交前审阅 `git status`、忽略规则与暂存差异；权重、视频、checkpoint、私有映射和大缓存不进入 Git。除非用户明确要求，不要 push。
