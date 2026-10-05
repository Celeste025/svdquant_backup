# E003：H3 block0 模态误差 pulse 的 BF16 continuation

状态：complete；类型：exploratory / causal diagnostic，不是论文证据。

## 决策问题

E001中text只占少量token却占88–99.7%的block0输出能量；固定旧state的corrected SVDQuant相较plain W4A4可降低整体误差，但video token误差更大。整体NMSE是否在后续49层中低估video误差的重要性，或text误差仍通过attention主导最终输出？仅凭局部模态NMSE不能决定。

## 固定契约

- 同E001缓存prompt1、step0/19，完整原始packed shape；不采样token，不改变分辨率或序列长度。
- 全BF16 DiT teacher完整前向，保存其block0输入/输出与最终video/audio denoiser输出。其余49blocks、final layer在所有arm保持BF16；同一输入、权重、attention实现、dtype和mask。
- 旧state rank32/g10/smooth/low-rank factors不变，仅使用已修正的高精度low-rank输入hook。量化是旧tie规则的NVFP4 QDQ，绝非native FP4 kernel性能实验。
- block0四个目标线性层换为普通nn.Linear，避免disk offload覆盖权重干预；计算donor后立即恢复全部BF16权重并移除quant hooks。
- pulse通过block0输出hook实现：选定token行替换为donor输出，等价于注入该行 `donor - BF16` 误差，并避免BF16额外减加舍入。每次hook确认未干预block0输出与teacher逐元素一致。

## 最小arms

1. BF16 zero pulse replay：必须最终video/audio与teacher一致，否则停止因果解释。
2. plain W4A4全部block0 pulse。
3. corrected SVDQuant全部block0 pulse。
4. corrected SVDQuant仅text token pulse。
5. corrected SVDQuant仅video token pulse。

text/video/audio由sample中的各自position_ids定义；验证唯一、边界有效、互不重叠。剩余行单独命名pad_or_unassigned，并记录token_tags分布，不能并入text。全量pulse包括audio和pad，其贡献不在本轮独立识别，不把full-text-video差值解释为audio影响（非线性不可加）。

## 指标、审计和停止标准

记录各donor局部all/video/audio/text/pad_or_unassigned的err²、ref²、NMSE、max_abs与token数；各pulse的实际注入energy；最终video/audio denoiser NMSE，相同teacher作reference。记录输入/量化state/脚本/common/模型源码sha256、原始模型文件完整SHA256、环境版本、attention实现、原始case元数据和分区索引SHA256。原始数据只读。

只完成2case×5arm即停止；无视频生成、无新校准、无超参搜索。若text pulse主导最终video误差，则不能因text局部energy大而忽略text；若video pulse主导且full corrected比plain差，则可继续审查校准目标，但普通重加权本身不是novelty。混合/步间相反结果则降低候选方向优先级，不扩大结论。

## 显存和执行

原模型38GB磁盘、GPU71.7GiB；使用既有disk offload、vram_limit=35GiB，E001单block峰值11.51GiB。保守预计<55GiB（resident weights≤约35GiB、block临时量+保留donors约15–20GiB），保留>15GiB余量。未加载文本编码器/VAE，无第二份完整模型。CPU可用约704GiB。若OOM则先记录失败，降低resident limit，不缩减packed输入。非性能benchmark，不做缓存冷启动速度比较。

脚本：`scripts/research/probe_h3_modality_pulse.py`。
输出：`results/research/E003_h3_modality_pulse.json`；每个arm写入partial，异常记录failed。
日志：`results/logs/research_E003_h3_modality_pulse.log`。
tmux：`research-E003-h3-modality-pulse`；GPU0，启动前再次检查空闲。

## 启动审计修正（模型forward前发现）

第一次启动检测到恢复环境默认SageAttention，在任何模型forward前停止，原始failed JSON/log保留为`E003_h3_modality_pulse_startup_rejected_sage.*`。H3 `_sdpa_varlen_attention`确实经`attention_forward`的global dispatch进入Sage，并非只有配置标记。第二次启动明确`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`，包装真实F.scaled_dot_product_attention入口，逐次验证Q/K/V为BF16并记录调用数。E001默认backend因此并非BF16 attention；E003与E001数值不能混用，所有donor/reference必须在E003重新计算。本实验内部backend固定。

模态pulse保留原始误差幅度，没有等能量归一化；其结果回答实际误差分布的下游影响，不能称为独立于误差幅度的内在模态敏感性。

## 完成结果（2026-10-02）

状态：complete。GPU0已释放，峰值allocated显存36.225GiB；两case的完整BF16 zero-pulse replay误差均严格0，各完整forward实际调用102次torch SDPA，Q/K/V均BF16。

| step | pulse | final video NMSE | final audio NMSE | 注入block0误差energy |
|---|---|---:|---:|---:|
| 0 | plain_full | 0.00117165069 | 0.00180212744 | 935883498 |
| 0 | corrected_full | 0.00133185241 | 0.000838622755 | 833149564 |
| 0 | corrected_text_only | 0.0002923049 | 0.000427807436 | 809384349 |
| 0 | corrected_video_only | 0.000959611638 | 0.000178726498 | 23697169.1 |
| 19 | plain_full | 0.000221520022 | 0.000394318974 | 462596227 |
| 19 | corrected_full | 0.00024038457 | 0.000326439494 | 458028069 |
| 19 | corrected_text_only | 0.000120975235 | 0.000189410386 | 364027974 |
| 19 | corrected_video_only | 0.000199823094 | 0.000125610548 | 93825558.1 |

解释：两步的corrected SVDQuant都比plain降低全block局部误差，但最终video endpoint分别差13.67%和8.52%；最终audio endpoint则更好。仅video pulse造成的video误差分别是text pulse的3.28倍和1.65倍，尽管text注入energy分别大34.15倍和3.88倍。于是，当前实际误差的全block能量排序不能可靠代表video目标；观察到目标模态之间的取舍。

text有658token（2.94%），video21312、audio414、未分配pad16；其token_tags分别1/0/2/-1。audio/pad没有混入text。full pulse包含二者，不能从full-text-video计算其独立贡献。

下一决策：保留“模态与下游作用不匹配的校准目标”作为需要进一步证伪的机制问题；不直接启动大规模重加权校准。应先核查最近prior work、另一prompt/深层block能否复现，并比较廉价的简单baseline。当前仅一个校准prompt、两步、block0的真实error pulse；既不证明heldout视频质量提升，也不证明所有层具有同一机制，更不是等能量敏感性比较或论文novelty。

审计说明：`pulse_local.nmse`分母是完整block参考energy；分模态局部NMSE在`local[donor][modality]`。两类分母不能混用。运行后校验了脚本SHA与JSON记录一致、5arm×2case完整、各模态局部err²之和等于全blockerr²。首轮Sage保护性停止文件仍保留。
