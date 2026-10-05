# H3 解码上下文与最终混合：只读可识别性核查

2026-10-03。仅阅读 E038 实际入口及 `old.DS=/home/wjq/workspace/DiffSynth-Studio` 源码，用整数几何计算索引；没有加载模型、解码、重评分或新增截图。**“E038 手部重影全部由最后的时间线性 blend 产生”与已见非 blend 帧不相容；这不证明或排除更广义的解码器/上下文作用。**

## 实际入口与时间合同

[E038 decode 入口](/home/wjq/workspace/svdquant-exp/scripts/research/prepare_decode_h3_center_video.py:206) 实例化 `MiniMaxH3VideoVAE`，以 BF16 解码、`tile_size=256, tile_overlap=64`。已完成 `decode_bf16.json` 等记录的 latent 为 `[1,24,37,36,64]`，视频为 124×576×1024、24 fps。此模型映射没有覆盖构造参数（[model_configs.py](/home/wjq/workspace/DiffSynth-Studio/diffsynth/configs/model_configs.py:1506)）。

VAE 的默认 `clip_length=17, token_drop=3`、空间倍率16、时间倍率4，导出 `frame_pre_padding=3, tokens_chunk_size=5, token_overlap=2, frame_overlap=5`（[构造函数](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:330)）。37 latent 加回 token_drop 后恰为40，无尾部复制 padding；`num_chunks=7`。七个输入窗分别为 latent **0–6、5–11、10–16、15–21、20–26、25–31、30–36**，每窗完整推理7 latent。

每窗 decoder 输出28帧，先分成20+8，再分别丢前3帧，形成17帧主段与5帧暂存尾段；下一窗的主段前5帧才和这个尾段 blend。最后一窗的5帧尾段直接输出。见 [decode_temporal](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:512) 和 [streaming 拼接](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:489)。

下表均为最终视频 **0-based、闭区间**：

| 窗口 | 主段输出 | 与前窗尾部调用 blend 的帧 |
|---|---|---|
| 0 | 0–16 | 无 |
| 1 | 17–33 | 17–21 |
| 2 | 34–50 | 34–38 |
| 3 | 51–67 | 51–55 |
| 4 | 68–84 | 68–72 |
| 5 | 85–101 | 85–89 |
| 6 | 102–118 | 102–106 |
| 最终尾段 | 119–123 | 无 |

`blend(a,b,L)` 为 `a_tail*(1-w)+b_head*w`，`w=j/L, j=0…L−1`，在张量 dtype 下计算（[blend](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:385)）。因此每个5帧带的**首帧 w=0 只取前窗**，真正混合两个非零贡献的是 **18–21、35–38、52–55、69–72、86–89、103–106**。完全没有调用最后时间 blend 的区间是 **0–16、22–33、39–50、56–67、73–84、90–101、107–123**。

## 空间覆盖与线性混合规则

`tile_overlap=64` 是最低值，不保证两轴都恰好64。`split_tiles` 会把剩余长度按16像素轮流补入重叠（[源码](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:368)）。本例是 **3×5=15 个 256×256 tile/时间窗**：

- 高576：起点 `[0,160,320]`，实际相邻重叠 `[96,96]`；latent 起点 `[0,10,20]`，每块16行。
- 宽1024：起点 `[0,192,384,576,768]`，实际相邻重叠均64；latent 起点 `[0,12,24,36,48]`，每块16列。

以空间坐标半开区间表示，调用空间 blend 的矩形并集为：

- 横带 `[y=160:256, x=0:1024]`、`[y=320:416, x=0:1024]`；
- 纵带 `[y=0:576, x=192:256 / 384:448 / 576:640 / 768:832]`。

每带首行/首列同样 w=0；严格双贡献从下一行/列开始。所有帧都可能有这些空间重叠带，和是否处于时间 blend 带是两件事。非重叠区域也已经经过相应 tile 的完整 decoder。

实现先纵向、再横向 blend，最后裁掉当前 tile 的下/右重叠再拼接（[tiled_decode](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:417)）。注意 `rows` 保留原始 tile，左邻不是已纵向混合的成品：横纵重叠交叉处是 `左原tile*(1-wx) + [上原tile*(1-wy)+当前原tile*wy]*wx`，不能擅自当成四邻 tile 的标准双线性拼接。

## 冗余上下文推理不等于最终 blend

7-latent 窗口、相邻空间 tile 在推理前已经重叠。decoder 是36层 ViT，窗口内时空 token 一起参与 attention（[ViT3DDecoder.forward](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:277)、[Attention.forward](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:111)）。因此即使某帧最后没有 blend，它也不是“无时序上下文独立解码”的输出。切换窗口大小会改变推理上下文，不是仅去掉一条最终加权公式。

还有一个实际接口边界：此版本 `decode_video(..., tiled=...)` 虽接收该参数，但视频路径无条件进入 `decode_temporal`，后者调用 `tiled_decode`；不能把传 `tiled=False` 当成已经实现了无空间分块对照（[入口](/home/wjq/workspace/DiffSynth-Studio/diffsynth/models/minimax_h3_video_vae.py:540)）。

## 已见非时间 blend 反例与决定

来自已完成的 [E041 replica1 全帧观察](/home/wjq/workspace/svdquant-exp/research_state/06_experiments/E041_clapping_replica1_observations.md)：global_mean 的 **12、40、64、94、120** 等帧仍可见明显多重手掌/手指轮廓或纹理拖影；coarse16 的 **80、94、116** 等帧也有此现象。这些帧全部落在上述完全无最后时间 blend 的区间。这里沿用已看过的24页，不增加截图或事后解码。

这些反例足以否定“全部重影只发生在/由最后时间 blend 产生”的强解释。它们不能排除窗口 ViT 上下文、空间混合、BF16 数值、输入 latent 或其交互；共同 decoder 也不保证对不同 latent 产生相同伪影。当前没有把残余具体定位到某个空间缝或上下文边界的证据，**不建议据此启动 decoder 参数网格**。保留解码器作为尚未定位的可能因素，不把“尚未排除”当成新主线。
