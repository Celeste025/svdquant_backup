# 旧实验与可复用资产审计

审计日期：2026-10-02。读取当前仓库代码、原始 JSON/CSV 与日志；本审计未启动 GPU。以下区分已有报告、代码事实及本次 CPU 验证。**旧观察可用作筛选假设，不能直接继承为论文结论。**

## 1. 会改变研究决策的发现

1. **H3 的 SVDQuant 低秩分支存在实质实现错误，必须先修复基线。** `scripts/minimax_h3_svdquant_common.py` 的 `install_runtime_hooks` 先通过前置 hook 把输入变为 `Q(x/s)`，再在普通 forward hook 中计算 `branch(inputs[0])`。后者收到的也是 `Q(x/s)`，因此旧代码实际计算 `Q(x/s) W4 + Q(x/s) AB`，而非标准 SVDQuant 的 `Q(x/s) W4 + (x/s) AB`。`infer_minimax_h3_svdquant_standard.py` 的 `_install` 与 PTQ 临时替换共用该函数。CPU 隔离加载原函数验证：与“低秩分支也量化输入”的输出 max diff 为 0；与正确高精度输入输出 max diff 为 1.792817（固定随机用例，NMSE 0.008279）。这不是 H3 真实误差大小，而是足以证明 hook 语义错误的最小反例。既有 H3 结果仍是旧实现的实测，但不能用于声称“标准 SVDQuant 不如直接 NVFP4”。
2. **`real-NVFP4` 是格式语义，不等于原生 FP4 kernel。** Wan/rCM 的 `load_quantized_transformer` 插入 DeepCompressor quantizer；`quant/weight.py` 明确 `return_with_dequant=True` 后 `module.weight.data=result.data`，执行普通 `nn.Linear`。H3 common 明确标注 fake quant，weight/activation 均 QDQ 后 BF16 GEMM。已有性能正例是 FLUX 的官方 Nunchaku **INT4**，不能作为视频 NVFP4 加速证据。`verify_rcm_real_nvfp4_checkpoint.py` 仅验证 300 个 block Linear/306 个原生 Linear 和有限输出，没有验证 Tensor Core FP4 指令。
3. **旧 idea 文档的关键基线前提已被后续报告否定。** `docs/VIDEO_W4A4_RESEARCH_IDEAS.md` 把 Q/K 独立 linear 误差最小化当成 SVDQuant 局限；9/28 的 `nvfp4_scale_jitter_20260928/REPORT_zh.md` 已纠正：DeepCompressor 的 QKV smoothing、weight/low-rank calibration 可使用 attention 或父 block 的 eval module，且支持共享输入低秩基。因此“改用 attention output loss”不是新贡献。
4. **局部指标的漂亮数字多次没有兑现。** scale-jitter 冻结网格局部改善 35.9% 不代表生成改善；轨迹恢复 top10 在四个案例中的平均视频 NMSE 反而增加 4.18%；signed 梯度 top/bottom 也无可靠排序。不要继续把这些阴性实验包装成正面机制主张。

## 2. 实际测过什么、可相信到什么程度

| 旧实验 | 数值证据 | 可支持的有限结论 / 混杂因素 |
|---|---|---|
| 早期 Wan INT4 group64/rank32、扩校准、W4A16/W8A4/rank64 | 校准均值 NMSE r32=0.911%、r64=0.765%；单个 seed44 视频 PSNR 13.87→11.84 dB；W8A4=15.70 dB | 局部拟合不保证单样本轨迹接近；不足以证明“提高 rank 系统性有害”。这是 INT4 fake quant，不能直接当 NVFP4 observation。 |
| Wan vs FLUX activation / weight / outlier | **修正版** postsmooth A4 NMSE 中位数 FLUX=1.430%、Wan=1.278%；权重重构 1.770% vs 0.834%；某 FFN-down outlier channel Jaccard .752 vs .903 | 当前采样不支持“Wan 更难因为 outlier 更严重”。跨模型架构、CFG、kernel/checkpoint和采样配置不同，不能完成因果归因。早期 FLUX 2.635% 含 smooth 未 unpack/shift 未加的 bug，不应引用。 |
| Wan CFG 与首步传播 | 单分支输出 NMSE 平均 .400%，CFG6=5.411%；绝对 MSE 放大 32.3 倍。modulation NMSE .712→2.991%，但绝对 MSE 下降 | CFG 放大有具体数值支持；**modulation 放大只是分母变化**。后者不能称放大机制。 |
| rCM 四步早期 INT4 | 输出 NMSE 13.097→38.492%；最终 latent 65.771% | 只有严格配对单案例；少步大更新与传播是合理解释但尚非独立控制因果结论。 |
| 轨迹单层注入，all300/2prompt | local-vs-final Spearman −.0280，block-vs-final .1993；normalized final p90/p10=33.0；`enter_trajectory_aware_lowrank=false` | 归一化相同输入误差后层间影响差异存在，但 prompt 数=2、confirmation 仅1。300层/1200层步不是300/1200个独立样本。该实验在BF16背景注入单层扰动，不等于全量化网络恢复价值。 |
| 实际 restore top10，4视频 | local top10 平均 NMSE 改变 −.40%、2/4改善；trajectory top10 平均 +4.18%、3/4改善；airplane个例 +42.815% | 均值/中位数冲突，样本太少。足以终止“敏感度排序即有效混合精度”的未经验证假设；不足以证明跨层抵消机制。 |
| signed restore | airplane signed top10/bottom10 均恶化约40%；fastmotion top改善18.0%、bottom恶化4.73%；top1 α=.01预测latent NMSE .7801，实际 .8754（基线 .7846） | 梯度预测不可靠；需检查STE、BF16有限差分、量化不连续性，不能把一阶近似失败直接当“抵消”的证明。恢复实现已有 smoothing 坐标处理，不能简单归因为忘记smooth。 |
| low-rank / smooth训练 | fullmodel 10epoch train NMSE .1077→.06447；smooth+lowrank 20epoch .1180→.07493 | 降训练损失确实发生；不能用训练曲线代替heldout rollout。on-policy retry日志因CSV列不匹配报错，不能声称完成20epochs。 |
| 20视频 lowrank all30 | 配对 NMSE .28589→.26019（−8.99%），LPIPS .49270→.46561（−5.50%）；VBench subject .91225→.92223，但 aesthetic .64950→.64801，background .92611→.91910 | 存在小幅真实配对改善，质量指标有取舍，没有通用优化成功；需核对训练/测试prompt重叠并给置信区间后再提升证据等级。 |
| H3 restore，p2/p16/p26 | p16 raw W4A4视频NMSE .2999，旧SVD .5191；restore10=.4483、restore20=.3557 | 受上述低秩输入 bug 污染；仅3 prompts且pixel接近BF16不等于质量。不要据此设计专门修复“SVD无效”的新方法。 |
| H3 activation audit | 8 prompts、固定step8、200层；fc2 max/RMS中位数392，A4 NMSE .4815%；QKV max/RMS26.7、A4 NMSE .8569% | 大absmax不自动意味着较大QDQ NMSE。A4用**前128 token**，H3 text/video/audio packed序列可能偏向前段模态；不能外推全部视频token。 |
| H3 SageAttention2/3 | 有适配、kernel benchmark、相似度和VisionReward脚本；当前本地未发现 `results/h3_sage_attention_similarity.json` 或对应完整结果 | 不能把pull到的脚本当已在本机复现实验。SA2 guard针对padding零值scale→NaN；SA3 wrapper按cu_seqlens切段调用kernel，并验证固定1/√d scale。多数新启动脚本硬编码 `/home/admin/...` 旧服务器路径，需改启动环境。 |

主要原始证据：
- `outputs/EXPERIMENT_SUMMARY_zh.md`；应优先用 `outputs/quant_error_diagnosis/REPORT_rerun_zh.md` 的修正数值。
- `results/reports/rcm_trajectory_sensitivity_audit/all300_2prompt_actual_normalized_1pct/correlations.json`。
- `results/reports/rcm_trajectory_actual_restore_top10/{comparison.json,multi_prompt_summary.json}`。
- `results/reports/rcm_signed_restore_top_bottom/{changes_vs_nvfp4.csv,step_size_check.json}`。
- `results/vbench/rcm_lowrank_8prompt_e10_vbench20/{video_similarity/summary.json,evaluation/summary.csv}`。
- `results/reports/minimax_h3_svdquant_restore_top/metrics.csv` 与 `results/reports/minimax_h3_activation_outlier_audit/summary.json`。

## 3. scale-jitter：应保留的机制发现与明确的阴性结果

9/28实验是当前证据记录最完整的一组：28次BF16捕获与原cache逐元素一致；NVFP4为E2M1/group16/E4M3/FP32 tensor scale；比较RNE与DeepCompressor中点规则；聚合使用误差能量和/参考能量和。rCM 4 prompts×16层×4步；Wan仅2 prompts、三个相邻步位置，branch0。

- rCM受控1%扰动：冻结block scale使response NMSE 30.5095→19.5684（−35.9%），point NMSE .8642%→.8676%。response分母是真实微小变化，不能说整体误差超过3000%。
- 普通Wan 25→26步真实activation变化中位数3.47%，局部响应下降24.6%；1→2及49→50反而恶化133.6%/33.9%。
- 单层功能干预：Wan最终denoiser响应冻结仅Q −.29%、FFN +.43%；迟滞 −.38%/−1.74%。DC兼容舍入复测同量级。没有完整W4A4 rollout、视频解码、VBench/FVD。
- 公共global scale的局部实验用输入对最大absmax，是归因oracle，不能直接作为因果在线方法。functional实验有避免未来信息。
- **决策：保留量化网格不连续性观察，停用“local response NMSE即质量代理”，不再大规模扫冻结scale。**

完整来源：`results/reports/nvfp4_scale_jitter_20260928/REPORT_zh.md`、`summary.json`、`quantizer_validation.json`、`reproduce.sh`。

## 4. 可复用资产地图（本机确认存在）

| 用途 | 路径与建议 |
|---|---|
| Wan/rCM BF16 | `/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers`，`/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer` |
| Wan/rCM NVFP4 rank32 | `/data1/models/svdquant-wjq/ckpts/{wan2.1-1.3b-real-nvfp4-s16,rcm-wan2.1-1.3b-real-nvfp4-s16}`；各约3GB，含model.pt/scale.pt/wgts.pt，smooth/branch是指向runs的相对链接，复制时需保留依赖 |
| rCM配对变体 | 同ckpts目录的 `rcm-wan2.1-1.3b-int4-s16-g10`、`...real-nvfp4-s16-g20-r32`、`...g10-r64`；勿混入带smoke/aborted的目录 |
| BF16轨迹cache | `/data1/models/svdquant-wjq/datasets/torch.bfloat16/{wan2.1-1.3b/unipc50-g6.0-f33,rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77}/vbench/s16/caches`；首选rCM快速闭环，普通Wan确认可迁移性 |
| jitter小激活cache | `results/reports/nvfp4_scale_jitter_20260928/{0000,0005,0010,0015}_s{0..3}.pt`，单文件34.7MB，含16层320token；`wan50/`含普通Wan端点。适合CPU/GPU小型格式与局部机制实验 |
| H3 BF16 | `/home/wjq/workspace/DiffSynth-Studio/models/{MiniMax/MiniMax-H3,Comfy-Org/MiniMax-H3}`；common默认路径不一定指向这里，需env覆写 |
| H3校准cache | `results/calib/minimax_h3_svdquant_standard_8p64s`：64样本、8prompts、8steps，约0.876GiB；`results/calib/minimax_h3_smoke_2p8s`约0.059GiB |
| H3量化state | `results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt`，r64及smoke sibling存在；低秩分支bug修复后旧state只作固定权重诊断，不直接视为正确重新校准基线 |
| 评测链路 | `scripts/eval_rcm_vbench_subset.py`、`scripts/eval_mjvideo_rcm_vbench51.py`，VBench/MJVideo结果已在results下；pixel/latent相似度需与质量、运动程度同时报告 |
| 默认PTQ Python | `/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python`；`scripts/env_svdquant_ptq.sh`、`docs/environment_matrix.md`。H3用单独env；不能直接合并依赖 |

## 5. 下一步门槛

1. 修复并回归验证H3高精度分支输入；先对保存state做固定输入的单层/块诊断，再决定是否重校准。把修复记为基线工程，不作为创新。
2. 建立至少一个原生FP4 GEMM的可验证性能基线；只报QDQ质量时明确标注，不从dtype/文件名推断硬件计算。
3. 新机制必须同时通过局部预测、heldout功能影响、匹配prompt/seed的自由rollout；若只改善local指标，立即停止扩量。
4. 新颖性核查必须重新读真实论文与基线实现；旧ideas的标题、arXiv链接和“未发现同类”判断本审计未验证，不可作为已查新证据。

本轮审核不删除任何旧文件，不改现有数据。后续修复另行记录。

## 修复与回归补充（2026-10-02，额度恢复后）

已修复 `scripts/minimax_h3_svdquant_common.py`：低秩输出在前置hook内使用未量化的平滑输入计算，主路径仍输入NVFP4 QDQ；forward hook通过 `always_call=True` 在主Linear或其他hook异常时清理暂存输出，`RuntimeHooks.remove()`也会清理。低秩模块使用同一个对象引用，因此兼容H3 `_install`在注册hook后将branch迁移CUDA。未修改旧检查点、量化器格式或无低秩分支的数值路径。

CPU回归脚本 `scripts/research/test_h3_runtime_hooks.py` 的 **10项全部通过**；结果见 `research_state/06_experiments/results/h3_runtime_hooks_regression.json`。覆盖冻结的旧错误语义反例、FP32/BF16与独立线性公式逐元素一致、bias、smooth、重复2D/3D输入、无低秩分支、安装后dtype转换、主forward/后续pre-hook/branch异常后状态清理与恢复、remove幂等性、临时量化context异常恢复。测试AST隔离加载实际源函数，使用仓库真实LowRankBranch，未加载H3模型或使用GPU；不把该结果解释为生成质量提升。真实H3单层/块级配对由主研究任务另行开展。
