# 双球相对投影运动：teacher-only 可行性

2026-10-02；CPU检查完成，未写runner、未启动GPU。**建议最多2提示×2 seed=4个H3 BF16视频。双颜色区域的相对间距比单物体绝对位移更直接，省去光流/相机估计；尚无teacher能可靠完成的证据。** 只测投影接近/远离，不证明3D距离、完整提示遵循或新量化机制。

**固定小试。** 公共提示：`A single continuous five-second wide shot of two matte toy balls on a flat white tabletop, viewed from a fixed overhead camera. One saturated blue ball stays at center-right. One saturated red ball starts at center-left. [MOTION] Both balls remain fully visible and separate, at constant size, on the same plane. Uniform lighting, ample empty space, no other objects, no cuts, no text.` 两个[MOTION]分别为 `The red ball rolls steadily in a straight horizontal line toward the blue ball, without touching it.` / `The red ball rolls steadily in a straight horizontal line away from the blue ball.` seed固定49771、59526；同seed两提示共享原双模态噪声。先第一seed一对，有效但静止/反向=behavior_fail；无法可靠读出=unknown；任一出现即停，不补seed。两例都有效正确才做第二seed。四例全过只获得后续配对试验资格，不能宣称统计可靠。球/颜色/简单运动预计比光束与产品文字更可测，但模型可能仍生成静物。

**直接读出与防伪。** PyAV解全部124帧，OpenCV uint8 HSV：红H≤10或≥170、蓝H∈[100,130]，两色S≥120、V≥60；`connectedComponentsWithStats`给centroid和等效直径 `d=2√(area/π)`。记录原始中心距D、两直径及 `ρ=D/mean(d_red,d_blue)` 的全时间曲线。ρ消除共同平移/统一缩放，但不能证明蓝球静止；旋转/深度/形变下也不等于物理距离。

拟定有效性门槛须先用CPU合成静止/相机平移缩放/真实接近远离/双物体等样例验证，随后在看生成结果前冻结：≥90%帧两色各唯一主区域，首尾10%窗口各≥80%有效；单色主区域占该色总面积≥90%、画面面积0.2%–8%，不触边、不接触/遮挡（ρ≥1.5）、不瞬移；两球直径比和各自直径相对首窗变化≤20%。连通域不能确认语义身份，保存mask叠图、完整轨迹和接触图，人工核实只有两球、颜色身份不交换。任何分裂、颜色丢失、裁切或严重形变为unknown，不能填零或从汇总中隐藏。

方向用首尾10%帧ρ中位数：toward要求`ρ_end/ρ_start≤0.8`，away要求≥1.25；同时原始D须同向变化≥10%，避免“两球膨胀导致归一化距离下降”冒充接近。若ρ与D矛盾则unknown；真实运动叠加强zoom可能因此被拒绝，这是保守边界。保留全曲线查短暂跳变，不将一帧配对或相机运动当作物体行为。检测有效却未达到方向/幅度是behavior_fail；这是行为标签，不是整体质量分。音频不评价。

**本地接口/预算。** `/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python` 已CPU导入PyAV18.1、OpenCV5.0、NumPy2.5（CUDA未初始化），`cvtColor/connectedComponentsWithStats`齐备，无需新VLM。复用 [TE-only API](/home/wjq/workspace/svdquant-exp/scripts/research/run_h3_native_paired_video.py:208)、E009 `make_h3_resident`、原 [H3 __call__](/home/wjq/workspace/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:93)、[video VAE decode API](/home/wjq/workspace/svdquant-exp/scripts/research/run_h3_native_paired_video.py:392)，TE/DiT/VAE分进程。资产根 `/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3/FL2VA`，DiT/TE/VAE完整路径与SHA见 `results/research/E010_h3_prepare.json`。E010入口硬绑旧prompt，不能直接改manifest重用；若获准需独立薄封装，不改冻结源或采样算法。

576×1024、124帧、24FPS、CFG1、双shift12/3，建议**50步**：这是默认步数，分辨率仍为本地576协议而非默认768×1344。按已测8.26s/步，4例约27.5 GPU分钟，加TE/加载/解码约30–35分钟；第一对失败约15–18分钟止损。20步只需约11–15分钟，但属旧校准/render快速协议；不能据其失败判量化问题，也不能观察失败后悄悄改50步。Wan rCM虽便宜，却加入另一4步蒸馏teacher变量，首轮不换模型追求成功。**Root仍可依据成本/文献判断完全不启动；本页不授权实验、不提出loss或新颖性声明。**
