# H3社区成品基线：有可复用线索，尚非已验证替代

2026-10-04，只读接入核查，0模型权重下载/0运行。此前“未找到H3 Nunchaku adapter”的检索范围不完整；不能继续用该表述排除现有社区成品。

发布者的[模型卡固定版本](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/README.md)列出免数据与8提示×20步校准两套Nunchaku Lite NVFP4 checkpoint，后者有Diffusers加载形式；卡片指定Blackwell运行环境。模型库最后修改时间为2026-09-13，这不是今天新发布的结果。卡片声明不等于本机运行或质量已经验证。

保存固定模型revision `1d59adb350cc915b2bf96d5a41d7abd238842772` 和producer仓库HEAD `09658748fdb63b6150a7621566a46fcb2d3b830f`，6份配置/说明逐SHA落盘，未执行远程代码。检查发现：

- [校准版config](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/calibrated-8x20/config.json)列312个SVDQ W4A4 target及50个AWQ W4A16 target，包含refiner及分开的Q/K/V；这与本地pruned模型200个主线性层合同不同，不能直接混合速度或画质数字。
- [生产sidecar](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-nvfp4/blob/1d59adb350cc915b2bf96d5a41d7abd238842772/svdq-nvfp4_r32-minimax-h3-t2va.config.yaml)同时记录FP4 weight与INT4 activation metadata。[producer映射说明](https://github.com/rootonchair/diffuse-compressor/blob/09658748fdb63b6150a7621566a46fcb2d3b830f/docs/deepcompressor_mapping.md)也说明A4及部分平滑策略尚非完整同等实现。未读完实际校准/runtime路径，不能据此宣布它使用错误kernel；但也不能认定其校准A4与部署NVFP4已经一致。
- 模型卡所指H3特定生产脚本在固定producer HEAD返回404；不能反推历史没有该脚本。[当前README](https://github.com/rootonchair/diffuse-compressor/blob/09658748fdb63b6150a7621566a46fcb2d3b830f/README.md)还保留额外runtime分发条件，未据此更改或安装本地环境。

**决定：** 将这套发布物保留为外部产品基线候选，后续性能比较不能只对本地粗糙实现。当前不下载整套权重、不据模型卡宣称强基线已复现；继续E071同一候选的实际评分/部署一致性验证。待决定采用该成品时，再明确checkpoint身份、目标范围、runtime可用性及匹配BF16参照；无需为它重复发散文献调查。本轮没有新方法claim或画质结论。

[下载清单与SHA](../../results/research/h3_community_baseline_intake_20261004/manifest.json) · [结构化检查](../../results/research/h3_community_baseline_intake_20261004/intake.json)。
