# E023：固定初始权重 outer global 的单因素 QAD 对照

状态：`prepared_not_started`。2026-10-03；仅计划，尚无新训练器、GPU 检查或实验结果。E022 继续原定运行，本计划不改变其源码、终点或视频选择。

**问题与用途。** E022 每次从更新后的 FP32 master 全矩阵 amax 重算 W outer global。此对照只问：在同一训练预算下，保留初始 W outer global 是否改善实际 native 的开发集拟合。它补充成熟尺度政策基线，不是新方法，也不预设“scale jitter”是失败原因。当前 ModelOpt 默认保留校准 W/A outer global、动态计算组尺度；这里仍用动态 A 和 legacy 舍入，故**不是完整 ModelOpt 复现**。版本依据见[已固定源码的配方核查](../01_literature/qad_strong_baseline_recipe_check.md)。实验属于基线完善，无新 claim。

## 唯一变量与固定条件

| 项目 | 固定约束 |
|---|---|
| 唯一变量 | 每个目标层保存初始 FP32 master 的 `weight_global_initial`，训练、QDQ开发评估、导出均使用同一个值；禁止更新、EMA、学习或导出时重算 |
| 初始化/覆盖 | 与 E022 相同原 rCM BF16 模型重新开始，300个 Linear、1,391,984,640 个 FP32 master；不续训 E022，不加 smooth/LR/rotation；bias/非目标状态冻结 |
| 量化 | W group16 的 SF 仍随本次 master 动态计算；A 完整沿用 E022 动态 pack；E4M3 ties-up、E2M1 signed ties-larger、BF16 QDQ输出、identity STE 不变 |
| 数据/顺序 | 直接读取 E022 已完成 `results/research/E022/data_inventory.json`，256训练/32开发，无重新采样或 teacher 调用；复用 `build_micro_schedule`，同512个 record index及顺序，每条训练状态两次 |
| 优化 | AdamW lr=3e-6、weight_decay=0、clip=1、128更新；每次step0/1/2/3各一个micro，NMSE各除4，保持 optimizer/FLASH backward/checkpointing/RNG 配置 |
| 评估/选择 | 0/32/64/128各在同32开发状态做QDQ与native评估；最小native pooled NMSE选候选，完全相等取早者；保留固定128终点，与E022同时报告，不事后挑新终点 |

开发集已用于选择，不作为新的独立质量测试。本轮不新增prompt、网格、损失、attention精度、学习率调度器或生成视频。

## 现有代码允许的最小实现

未来仅新增独立 fixed-global 模块和薄 E023 训练入口；冻结的 [QAD模块](../../scripts/research/wan_mainweight_qad.py)、[fastpack](../../scripts/research/wan_nvfp4_fastpack.py)、[E022训练器](../../scripts/research/train_wan_mainweight_qad_expanded.py)不修改。无需新量化 kernel。

1. 初始化每层时，用原 `pack_nvfp4` 所调用的 `_amax_parts`、`_global_scale` 一次性求 global：FP32 `(amax * (1/6)) * (1/448)`，保持 `enable_fp_fusion=False` 和全零 canonical=1；不能替换成一次 `/2688`。将 FP32[1] 注册为非训练 buffer，保存初始值及层名。
2. 新 W pack 入口接受该 buffer，跳过动态 global 两个 kernel，直接调用冻结 `_encode_groups`。由于旧 `_global_scale` 原本负责初始化flags，新入口必须显式将两个flags清零；初始global检查正且有限，恢复checkpoint时复核。W仍直接读FP32 master，不先cast BF16。A继续原 `qdq_ste(x)`；W的新 autograd Function返回真实decode值，backward对master返回原identity梯度，对global返回None。
3. 继续复用原单kernel `decode_packed` 与 native主支。导出调用同一个固定global W pack，保存 `global_scale` 和初始global绑定。原 `install_packed` 会严格拒绝不同 `RECIPE`，因此新增薄安装入口明确接受新recipe再复用 `PlainPackedWanLinear`；不能伪装成原dynamic recipe绕过检查。部署时仍只有packed W/bias与动态A，无BF16主W、FP32 master或训练buffer引用。

**有限饱和已有实现。** `_encode_groups`先计算组内 `ideal=amax_group*(1/6)`，再 `u=min(div_rn(ideal,g_fixed),448)`，按旧E4M3 ties-up编码；七个E2M1边界最多产生magnitude code7，即±6。更新后超过固定global覆盖的值将饱和，不应造成FP8溢出NaN。有效SF=0仍沿用Wan `remove_zero=1`，并拒绝“零SF、非零code”这类native不能表达的情况；不能采用H3的zero-SF兼容规则。冻结global不是冻结组SF，也不是把所有W数值先裁到同一BF16范围。

## 必要检查与输出

- **起点一致**：300个初始 global及packed codes/SF与E022 step0包逐字节相同，bias/非目标状态相同；这是起点合同，不要求训练后两臂接近。人工有限样本覆盖相同global的普通值、超过448的组尺度饱和、E2正负中点、零组；独立软件解码核对，不新增完整DiT调用。非有限输入和不可表达零SF按原规则报错。
- **global贯穿一致**：保存300层初始global摘要；0/32/64/128逐层检查buffer、master checkpoint中buffer、导出packet global均与初始bytes相同。训练和导出的W pack共用入口。记录初始及各导出点的饱和组数/比例，作为描述量，不设“允许比例”质量门槛，不插入每层host同步。
- **训练/部署合同**：沿用E022首更新真实梯度、权重变化、finite、样本顺序与调用计数；开发QDQ每次600份W/A pack flags，native每次300份A flags及实际300次FP4 GEMM、60次BF16 SDPA。训练checkpoint重算调用另计，不冒充额外样本。native安装后无master/LR/hooks。
- **比较输出**：保存各检查点32条QDQ/native输出、pooled与逐prompt/逐step NMSE、相对各自step0变化、固定128和预定候选；原始master/packed/Adam与global buffer按既有格式留存。比较E022与E023的同数据、同顺序结果，不将QDQ单独改善解释为部署改善。

最小有效设置就是完整128×4训练；单层重构、少量更新或仅QDQ不能替代它。预计512训练micro＋4检查点×32状态×2评估模式＝768次完整前向，另含checkpoint反向重算；无新teacher/VAE调用。

## 资源与结果解释

root后续统一实现/启动，GPU5须在E022相关任务结束后查空闲，命名tmux、日志及外部绝对deadline；**90分钟、peak allocated 60GiB**。以E020约7.8秒/micro保守估计训练67分钟，加评估/导出约80–90分钟；只是预算估算，不作速度结论。仍是microbatch1，global buffer总计1200字节，不改变训练显存级别；另预留约30GiB磁盘给三份master、四份packed、Adam和输出，数据继续引用E022。拟报告 `results/research/E023/`，大文件 `/data1/models/svdquant-wjq/research/20261003/E023/`，拒绝覆盖。

只因超时/显存、非有限、量化表示错误或合同不符停止并保留失败；不以任意NMSE阈值提前停。若native开发误差改善，可将其作为更合理的普通基线候选，后续质量验证另定；若持平或变差，保留E022配方，本对照只说明固定W-global在这份数据/预算下未带来收益。无论哪种结果，都不能据此宣称动态尺度是新机制，亦不能排除固定A、不同数据分布或更充分优化的成熟做法。
