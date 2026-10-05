# E010：VisionReward 辅助视频诊断契约

只准备代码与CPU检查，未启动模型/GPU。新脚本 `scripts/research/evaluate_h3_paired_visionreward.py`，不修改用户原 `tools/evaluate_visionreward_video.py`、E010生成脚本或官方源码。

范围为已预定p30/p36的BF16/native四个解码视频。严格核对caseJSON中完整prompt、原seed、variant、video绝对路径与SHA；四个组合必须齐全，不按结果挑选。唯一case_id包含prompt、seed、variant和video SHA，避免旧工具label碰撞。此项只作两例辅助诊断，不能支持质量等价、泛化或统计显著性；**不评价音频**。

## 保留的官方行为

- 官方源 `/data1/models/svdquant-wjq/third_party/VisionReward/inference-video.py` 的 `load_video` 与 `inference` 函数通过AST单独提取、原样执行，避免执行其顶层下载/模型加载。模型改为本地离线加载，其余生成契约保持。
- 使用全部29行官方questions，**保留原readlines()末尾换行**，按原 `replace('[[prompt]]', full_prompt)` 展开；原29个weight原值不改，不截题、不裁prompt。
- 原chat取帧：每整数秒找最近frame，最多24帧；保留每题实际indices、timestamps、原总帧数与FPS。124帧/24FPS预计取 `[0,24,48,72,96,120]`，以实际文件读数为准。
- 原 `max_new_tokens=2048, pad_token_id=128002, top_k=1, do_sample=False, top_p=0.1, temperature=0.1`；官方只用首个新token解码值作判断。新脚本额外保存完整生成token IDs/解码文本、输入token IDs、首token ID与未经strip的解码文本；不把旧工具max_new_tokens=1混称原设置。

## 两种分数与未知回答

1. `strict_official`：原公式 `mean(weight * (1 if first_token_decoded == 'yes' else -1))`。例如大写`Yes`在这一列仍为-1，保留官方大小写行为。
2. `normalized_yes_no`：仅对首token解码文本做 `strip().casefold()`，只接受`yes`或`no`，按相同weight均值计分。

任何无法归一化为yes/no的回答都列出question index，**两种可用score均设null、valid=false**。为审计保留 `raw_formula_value_even_if_invalid`，但它明确不是有效分数，不能把未知回答静默当no使用。官方严格值与归一化值分列展示，不声称它们等价，也不挑较好的一列。

## 本地资产、验证与持久化

模型 `/data1/models/svdquant-wjq/models/VisionReward-Video`，6个shard共25,015,199,008 B；本地model code包含现有适配改动，所以记录实际代码SHA，不声称原始远程仓库逐字节未改。官方questions/weights/source和旧工具SHA绑定已有 `visionreward_contract_audit.json`。CPU检查模型shard header/完整文件长度与index、tokenizer大小写映射、已知/未知回答判定、官方取帧函数合成fixture，并要求CUDA未初始化。

真正评价启动前，完整SHA全部模型文件并保存源快照；模型加载后，实际dynamic-module cache代码必须逐文件SHA匹配本地代码。每题与每视频保存一次JSON，异常保留此前结果；不覆盖已有报告。报告保存Python/Torch/Transformers/Decord版本、GPU/dtype、generation_config与attention实现。运行前仍须查GPU空闲、named tmux与timeout/log；本任务尚未授权启动。

```bash
# CPU契约检查；已存在报告需显式换output保留先前证据。
CUDA_VISIBLE_DEVICES= /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python \
  scripts/research/evaluate_h3_paired_visionreward.py --check

# 四个E010视频均已完成、另获启动安排后才运行；GPU编号由当时空闲情况决定。
/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python \
  scripts/research/evaluate_h3_paired_visionreward.py
```

CPU通过不代表模型完整加载/生成已通过。四个真实文件尚未生成时，check报告仅标记case文件未齐，不伪称视频契约通过。评价可能生成多token直到EOS或2048上限，当前不估造实际耗时。
