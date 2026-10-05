# E042：既有拍手 PCM 的固定读出

2026-10-03。**六条能量包络已可读取，声音语义及音画事件对应仍 unknown。** 没有听音、自动找峰、阈值/时移搜索、同步评分或音频模型调用。

单次 CPU 信号读出 complete，2.540 s，CUDA 未初始化、0 GPU / 0 模型。使用全部六份既有原始 stereo PCM，每份 5.175 s；原文件和 waveform SHA 均匹配 E038。严格按协议计算 10 ms 双通道 RMS（2583 窗）和 20 ms Hann、1 kHz 至 Nyquist 的 rFFT 平均功率平方根（2578 窗）及其正差分，hop 均为 2 ms、无 padding。原 L/R、原始曲线、逐曲线最大值与归一化值均保存于六个 NPZ。每条曲线用自身全长最大值归一化，图不支持响度比较。

E041 的两种 JSON schema 已统一为各自视频的闭合外观区间；没有用 BF16 时间替代 FP4 视频时间。绿色仍仅表示闭合外观、接触不确定；灰色表示原 E041 不可计数或不完整区间。无标注不等于无动作/无接触。

| 已实际查看的图 | 全长 | 固定 0–1.4 s |
|---|---|---|
| replica0，三臂 | [全长](/data1/models/svdquant-wjq/research/20261003/E042/vbench0161_r0_full.png) | [放大](/data1/models/svdquant-wjq/research/20261003/E042/vbench0161_r0_zoom_0_1p4_layout_v2.png) |
| replica1，三臂 | [全长](/data1/models/svdquant-wjq/research/20261003/E042/vbench0161_r1_full.png) | [放大](/data1/models/svdquant-wjq/research/20261003/E042/vbench0161_r1_zoom_0_1p4_layout_v2.png) |

replica0 三臂均有许多从低底座升起、随后回落的分离能量脉冲，部分发生在相近时刻。但一个视觉闭合区间附近可能有多个脉冲，较强峰也会落在区间外，不能自动一峰一拍。replica1 的 BF16 包络包含较宽的能量团与窄尖峰；global_mean 的底座较持续，coarse16 的曲线呈较分离脉冲。这里描述的是各自归一化后的结构，不能据图判断单人拍手、人群掌声、音乐、混合声或谁更响。更不能把音轨形状变化直接称为同步破坏。

尤其 BF16 自身还没有建立可信的一一事件对应；低位视觉不可读处也无法补出事件。故本轮不继续同步归因，不输出更精确的 offset、接触次数或量化损伤结论。E004/E015 的停止决定保持不变。

产物：[readout.json](/home/wjq/workspace/svdquant-exp/results/research/E042/readout.json) 绑定原始源、六个 NPZ 与初版四图；[render_zoom_v2.json](/home/wjq/workspace/svdquant-exp/results/research/E042/render_zoom_v2.json) 绑定两张最终放大图。初版 zoom 的自动布局截断部分标题/图例，保留原图；v2 **仅从相同 NPZ 重绘固定边距，不重算信号**。原读出源码执行后未改。recovered 环境缺 matplotlib 的前置导入失败保留于 [preflight.json](/home/wjq/workspace/svdquant-exp/results/research/E042/preflight.json)，实际复用已有 PTQ 环境，没有安装依赖。
