# 阶段042：BF16末步有一致改善，但不能恢复完整量化轨迹

2026-10-03。E046全部八例、16媒体及独立评价完成。**完整SVDQuant轨迹只将末步换回原版BF16，八例的MJ总分、对齐和细节均提高；明显碎片/形变仍在，且新增约2.65 GiB显存。** 这是应保留的强控制，不是新方法或等质量优化结论。

固定四动作×两seed全部保留，真实E043初始噪声/embedding，原版Wan2.1-1.3B、81帧480×83216fps、50步UniPC/CFG6/shift8，FP32 sampler与VAE。每例只跑一次完整native轨迹：最后一步继续native得到native_full；另用同一末步前输入和完整history，独立原版BASE BF16模型执行cond/uncond与CFG6，再更新一次，得到bf16_last。替换的是整套SVD末步配方，不能单独归因FP4主支或低秩。两终态同轮解码和评分，旧结果仅作参照。

| 八例均值 | native_full | bf16_last | 原E043 BF16 |
|---|---:|---:|---:|
| MJ total | .56357 | .64060 | .62983 |
| Alignment | .59595 | .67993 | .66248 |
| Fineness | −.04128 | −.00396 | .04080 |
| Coherence | .54492 | .57617 | .60107 |
| AMT | .97034 | .97473 | .96686 |
| RAFT动态判定 | 6/8 | 6/8 | 7/8 |
| DINO | .90127 | .91779 | .92700 |

MJ是模型原始输出，不是概率。按每文本两seed平均、再对四文本等权；八例是已观察诊断集，不作总体显著性推断。相对native，coherence与DINO各7/8上升，例外分别为喷水r0和r1；RAFT逐例全部不变。高AMT不能当作动作正确。

| 文本 | bf16_last−native MJ | bf16_last−原BF16 MJ |
|---|---:|---:|
| 拍手 | +.06601 | −.16425 |
| 叠衣 | +.11835 | +.19820 |
| 转弯 | +.07695 | +.10145 |
| 喷水 | +.04680 | −.09230 |

尽管总均值略高于原BF16，**逐例总分仍6/8低于原BF16**；两个超过原版的例子仍是原版自身较差的叠衣r0、汽车r1。fineness与coherence各6/8低于原BF16，均值差−.04476/−.02490。因此不能将总均值称为质量恢复。

root在评分结束前查看全部16张固定九帧图。BF16末步常令皮肤、天空、草地和衣料更平整；拍手掌指/腕部碎片、叠衣后段衣物变形、车辆几何及末帧第二辆异常车仍在。两喷水例变化较轻，未见决定性动作修复。全部观察见[配对记录](../06_experiments/E046_visual_observations.md)；非实时播放/盲评，不声称精确动作频率。

成本：每worker备用BF16的CPU载入＋H2D为.695–.891秒，模型参数＋buffer为2.64989 GiB，实际additional allocated为2.65063 GiB；两次BF16前向＋末步更新3.308–3.450秒。双模型含解码allocated峰16.972 GiB、reserved峰23.611 GiB。加载时reserved变化−2.932 GiB受缓存状态影响，不能当负模型成本。第二seed的native阶段已驻留备用模型；以上是带诊断的直接双驻留执行，不是成熟部署benchmark，也不能从2/100调用比例推出2%总延迟开销。

生成1126.01秒，评价72.58秒，四生成/三评价进程均rc0退出，GPU0/1/5释放。实际816DiT（800native＋16BF16）、240000主FP4GEMM、48960BF16 SDPA、408scheduler、16VAEdecode、0TE/0训练。八个native终态与E044逐元素零漂移，MJ四项亦相同，AMT仅约1e−9数值差。独立CPU汇总核16媒体/终态、400标量步、816调用收据、末步输入与条件及CFG操作；没有补造第二份history trace或独立scheduler重放。[汇总](../../results/research/E046/evaluation_summary.json) SHA e1129e856ada4029df64e9b64fc0e8145e97869cb989dfee2167f30a3cf31a72。

决策：保留bf16_last作为更强质量基线；停止“单末步保护足以修复”的预期，不自动扩大保护步数/训练/参数网格，也不据此宣称decoder或多步求解器根因。完成的[基线审查](../00_state/baseline_readiness_after_E046.md)核实：旧33帧缓存的实际step0/25/49为t=[999,749,57]，本轮81帧/shift8对应[999,889,146]，形状与实际时间表均不匹配；旧时间点与collector默认shift3一致，但完整历史shift配置未直接保存。rank32符合所读官方通用默认，现有rank64资产分别为rCM祖先或INT4 fake格式，均不能直接替代。下一先补同原版81帧/CFG6/UniPC50/shift8的匹配校准，保持rank32及原生量化设置、校准文本与诊断集分离，再做完整生成比较。失配是否导致这些质量缺陷仍待验证，匹配校准本身不是新贡献。空间patch相位线索目前只有[入口审查](../02_problems/wan_patch_phase_entry_audit.md)，没有测量或新claim。

并行查重停止了CFG两支“共同/差分换基量化”候选：[GCBT](https://arxiv.org/html/2610.00930v1)已直接覆盖该方法族、线性后逆变换，并报告Nunchaku上的系统成本，不能将native接入本身当空白。[核查说明](../01_literature/CFG_branch_hadamard_duplicatecheck.md)。**总体目标active，仍无可投稿核心贡献。**

协议：[E046](../06_experiments/E046_wan_terminal_repair_plan.md)；[生成](../../results/research/E046/launcher.json)、[评价](../../results/research/E046/evaluation_launcher.json)；上一阶段：[041](041_20261003_terminal_intervention.md)。
