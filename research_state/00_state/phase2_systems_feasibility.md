# Phase 2：H3机制、heldout与深层pulse可行性（2026-10-02）

本轮是静态审计，未启动新模型推理、未编辑E004。结论：**尚未找到已被证实“普通per-token/group scale与共享SVD都无法覆盖、且SM120可免费利用”的新结构。** H3有值得验证的归一化几何与低维AdaLN结构，但幅度差、普通模态重加权、解析均值补偿都不能直接包装成创新。

## 真实结构与可以排除的解释

- p1完整packed布局由原sample position_ids核实：text `[0,658)`，audio `[658,1072)`，video `[1072,22384)`，pad `[22384,22400)`。cu_seqlens=`[0,22384,22400]`，即pad是独立attention段。block内norm/MLP/final layer逐token操作，最终只选video/audio rows；故**E003全BF16 continuation中的纯pad输出pulse没有通向video/audio的路径**。这不排除“全量化时pad影响tensor global scale”的另一种机制。
- H3每block的norm1/norm2是共享RMSNorm，不是三套模态独立norm。差别来自`combined_indices=time_index*3+modality`，选择channelwise AdaLN shift/scale/gate；text还经过专属token refiner的final RMSNorm。不能把text原始残差能量大直接等同linear输入outlier大。
- `h = D_(m,t) RMSNorm(x) + b_(m,t)`，attention/MLP输出再乘模态gate并加回residual。RMSNorm对径向误差近似不敏感，但residual、epsilon、后续gate使全网络不具备任意逐token缩放不变性。必须直接测误差的径向/切向部分及后续传播，不能单凭RMSNorm公式解释E003全部现象。
- NVFP4 block scale按**每token、每16通道**独立，M轴text/video不同token不共用block scale。只有tensor global可能共享。per-token/group scale已覆盖粗幅度差；系统审计已展示2的幂倍率可精确被E4M3指数吸收，不能重新把这个问题叫模态动态范围新机制。

## 几类候选结构与已知限制

| 结构 | 普通scale/SVD能否表达 | kernel可行性及未证明部分 |
|---|---|---|
| 模态整体幅度不同 | group/token scale已覆盖；任意global有格点phase或极端下溢 | native支持，通常不是新增贡献 |
| 模态/time的channelwise AdaLN尺度在16组内不同 | 单一group scalar不能表示任意组内对角变换；标准channel smoothing可部分覆盖 | 可在activation quantization前处理，但常规模态重缩放不是新发现；不同模态weight变换会要求多个权重或gather/分GEMM |
| residual误差径向/切向的下游作用不同 | scale改变格点，SVD提供共享输出子空间，都不自动优化下游RMSNorm几何；但更好的既有目标也可能足够 | native GEMM只执行给定codes/scales，不免费实现几何敏感补偿；需证明“why not simpler” |
| additive AdaLN shift低维子空间 | 本机pruned checkpoint每模态≤9维、三模态≤27维；rank32共享branch原则上足以覆盖其联合子空间 | 可预算Wb或将shift留在高精度补偿，但与centering/低秩补偿重合；不能声称SVD表示能力绝对不足 |
| modality/time专属误差方向超出共享rank预算 | 固定rank32可能不足，增加rank/改shared basis是必要对照；未测联合子空间秩/主角度 | 分模态补偿可用小GEMM/epilogue，但rank与带宽开销需计费，当前API不支持免费路由 |

低维AdaLN证据来自safetensors header与真实模型源码：`adaln_t_table=[1025,8]`，blocks0/24/49的`adaln_proj.linear.weight=[96768,8]`且有bias。每一模态shift是`t_emb∈R8`的affine map，故shift族span≤9；拼接三模态≤27。对qkv/fc1可用`W(Dz+b)=W(Dz)+Wb`解析分离。**这只是本机Comfy pruned表示的结构，不能外推原始所有H3/Wan；W diag(D)一般仍是高秩，不因D来自8维参数就变成低秩矩阵。**

已验证native接口为torch `F.scaled_mm`，BlockWise1x16+TensorWise全局scalar以及128×4 swizzle；不接受BlockWise1x16+RowWise二级vector组合。不同模态连续M区间可以分GEMM，任意M已小规模测过；代价是launch、padding与小M效率。单GEMM中absorbing modality global进E4M3只在可表示时精确，否则再次舍入；自定义row epilogue尚未实现。RNE/packing/native parity应在复杂方法前完成（E005）。

## heldout缓存：核实结果

不是只看config：实际`per_block_stage_nmse.json`含9600条记录=50blocks×3stage×64samples，其prompt集合严格为：

`{1, 11, 20, 25, 46, 48, 105, 116}`，step集合`{0,3,5,8,11,14,16,19}`。

与8份calibration manifests及`configs/minimax_h3_svdquant_standard_8p64s.json`一致。quant_state本体只有64sample数量、不含prompt IDs，因此必须保留该外部证据链。

对项目`results/`及`/data1/models/svdquant-wjq`使用包含ignored文件的扫描，只发现：

- 64份576×1024/124frame/20step的标准raw DiT calls，全部在上述calibration集。
- 8份smoke raw calls，prompt48/105、256×448/39frame/30step，也不是prompt-heldout，而且shape/schedule不同。
- quant_state/progress、视频/metric manifests；**没有发现heldout raw DiT calls或可复用的中后层BF16 block input缓存**。`collect_h3_single_layer_activations.py`存在，但其默认`runs/h3-singlelayer-dq-vs-svdq`数据路径在本机未找到；即使存在，单fc2输入也不足以重放完整block。

旧PTQ代码把量化block输出递推为下个block输入，但只在内存维护`block_samples`；保存的progress是量化state，不是BF16 teacher block cache。因此不能将中部state的训练输入等同BF16上游分布。

原prompt表`/home/wjq/workspace/178866172854036`中16/26/42/51均存在，且归一化文本不与上述8个校准prompt重复。26/42/51原config已指定evaluation，故可称**calibration-heldout**，不能称研究过程中完全未看过的blind test。建议先固定42（seed52386）与26（seed63583），不根据测得收益挑prompt。

## 最短扩展路径和成本

1. 深层扩展：复制E003为独立runner，增加`--block-id`与`--prompt-id`，state key改为`blocks.{id}.*`；仍在完整BF16 teacher一次前向时捕获目标block输入/kwargs/output，再做独立donor和output-hook pulse。无需持久化全50层activation，且不改common/E003已完成版本。
2. 中部用block24，晚部用block45或48。**不要把block49 text pulse作为一般化证据**：其后只有逐token final layer及输出row selection，text-only在这里理论上无法影响video；可作为负控制，但结果零是结构必然。
3. 优先在已有p1两步上测24/48，保持E003五arm；若不复现则停止扩张。E003实测每完整forward约8.7–8.8秒，每case含teacher/donors/五arm约58–61秒，峰值36.225GiB。4新case保守预算4–8 GPU分钟（不计冷启动/磁盘hash），比修改continuation API跳过prefix更省工程验证成本。
4. heldout缺cache，需要新收集器。复用`collect_minimax_h3_calib_standard.py`的raw pre-hook，保存0/19；在最后所需pre-hook保存后用专门异常退出，跳过该step的DiT与所有VAE decode。step19输入仍需要之前19次BF16 denoising，不能把step19 timestep贴到初始noise上替代。必须固定torch SDPA、采样schedule、seed和rand_device，并显式保存backend provenance。
5. 本机全模型组件实际位于`/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3/FL2VA/`，text encoder/processor/video VAE/audio VAE均存在。`SVDQUANT_DATA_ROOT`下默认`models/MiniMax-H3`不存在，需设`MINIMAX_H3_MODEL_ROOT`为上述真实目录，prompt文件显式设原始表。原collector会一直生成到VAE decode，不宜直接无修改运行作纯诊断。
6. 每heldout prompt获得0/19输入的理论DiT部分约19×8.8≈167秒，另有text encoder/初始化/磁盘offload；预算4–8 GPU分钟/提示，非性能承诺。两prompt约8–16 GPU分钟，缓存约4×14MB。随后每prompt×2step×2block的五arm约4–8分钟；先小规模做一个prompt/一个block即可决定是否继续。
7. 需修改/新增的文件仅：独立`collect_h3_heldout_calls.py`、泛化E003的独立pulse runner、对应计划和JSON。无需重校准旧state、无需改common历史量化器/原模型文件、不动E004。本审计未实现或启动上述新实验。

### E005准备核查

recovered Python实际导入`/home/wjq/.conda/envs/convrot-wan/.../torch`，2.11.0+cu128；CPU实测确认`F.scaled_mm`、`float4_e2m1fn_x2`、BlockWise1x16/TensorWise/SWIZZLE_32_4_4存在。真实qkv权重[21504,5376]、fc2权重[5376,14336]均满足native K/N对齐。可仅跑一个完整BF16 block0获取两个linear输入，原生GEMM无新依赖。E005只做数值契约，不把prepacked GEMM速度当端到端收益。
