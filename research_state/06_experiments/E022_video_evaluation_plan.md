# E022 独立视频评价补充协议

2026-10-03。仅 CPU 文本/目录元数据预选；未读取这些新轨迹的模型输出或质量分数，未生成视频。本补充不修改已冻结 E022 训练计划、源码或 data_manifest。

固定池为已有 251 条 VBench manifest。文本按 Unicode NFKC、casefold 后只保留字母数字归一化；排除 E022 的 32 个训练/4 个开发文本、E020 已知 20 个校准文本、E021 四例和已知 toy 文本。对 E020 inventory 列出的四个本地媒体根，排除其历史已存在 case 目录及本次可见 MP4 case ID 对应的文本并集；只检查路径与文字，不看视频/分数。不把整个 planned manifest 都视为已经生成。

源 case_id 排序、归一化去重后剩 125 条；Python Random(20261110).shuffle 后取前 8 条，不按题材配额、预期难度或结果换样。这是本研究已列本地历史之外的 held-out，不声称全球未见或语义完全不重叠。完整排除来源、候选顺序、SHA 与文字保存在 [video_test_manifest.json](/home/wjq/workspace/svdquant-exp/results/research/E022/video_test_manifest.json)。

| 序号 | prompt_id | 两个 seed | 原始 prompt |
|---|---|---|---|
| 0 | vbench_128 | 20261110, 20261111 | A space shuttle launching into orbit, with flames and smoke billowing out from the engines |
| 1 | vbench_134 | 20261112, 20261113 | Iron Man flying in the sky |
| 2 | vbench_095 | 20261114, 20261115 | time lapse of sunrise on mars. |
| 3 | vbench_232 | 20261116, 20261117 | science museum |
| 4 | vbench_133 | 20261118, 20261119 | Gwen Stacy reading a book |
| 5 | vbench_191 | 20261120, 20261121 | courtyard |
| 6 | vbench_204 | 20261122, 20261123 | harbor |
| 7 | vbench_197 | 20261124, 20261125 | football field |

每 prompt 两条轨迹，seed=20261110+2×index+replica。16 个 seed 不同且不与 E022 train/dev seed 值重叠；同一轨迹四臂共享保存的实际初态、四份更新噪声及文本 embedding，随后各自自由递推。固定 rCM 四步、sigma80、guidance0、480×832、77 帧、16 FPS、BF16 FLASH attention 和 E021 BF16 VAE core128/halo0。

主评价仅四臂：BF16、E022 packed_step0000 plain、冻结 SVD+LR、QAD-native-dev-selected，共 64 视频/256 DiT。QAD 严格绑定已冻结 trainer 在 {0,32,64,128} 中 native pooled dev NMSE 最低者，平分取更早步；完成后另写 video_checkpoint_binding.json 记录 selection 文件和导出权重 SHA，不修改样例。若选中 0，照实保留，不能改用非零 checkpoint。此次不加入固定 128 次要臂。SVD 是整套已有配方参照，不能将差异独因果归 LR。

评价与汇总沿用 E021 分开的 AMT/RAFT/DINO 和 MJ 总分、alignment、fineness、coherence；逐 prompt/replica 展示，再对每 prompt 两个 replica 求均值、最后对八个 prompt 等权汇总。BF16 失败样例也保留，不按分数或视觉失败替换；不将高 smoothness 或单一均分等同整体质量。主差值为 selected−plain 与 SVD−plain。运行预算和 generator 由后续安排，本轮未实现新 generator 或启动 GPU。
