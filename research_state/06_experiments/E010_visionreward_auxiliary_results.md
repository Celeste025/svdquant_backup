# E010 VisionReward：四个配对视频的有限辅助结果

2026-10-02，完成，`EXIT_CODE=0`。原始结果 `results/research/E010_visionreward.json`，SHA256 `8f4ad4b1bfe37f7b7b965532963497a91a93a6ab64b311cde99cc7a09ad44eab`；日志 `results/logs/E010_visionreward.log`。脚本、官方questions/weights、模型六shards与实际加载代码的SHA均已保存。事后CPU独立重算分数、四视频SHA、29题数量和取帧位置通过。

| Prompt | BF16 | Native | Native − BF16 | 首token yes/no（BF16；Native） |
|---|---:|---:|---:|---|
| 30 | 0.0104732160 | 0.0104732160 | 0 | 11/18；11/18 |
| 36 | 0.1960319278 | 0.1802847953 | −0.0157471325 | 27/2；25/4 |

全部116个首token均为原始小写`yes`或`no`，未知回答0，因此本轮严格官方与归一化分数一致。p30的29条回答逐题相同；p36仅第14、15题从BF16的yes变为native的no，两题询问**开头物体形状**。这是评价器回答的差异，不是量化影响机制的证明。

官方规则使用首个生成token。115条回答共生成2个token（含EOS）；p36 native第3题生成18个token，首token仍为`yes`，额外文本完整保留，未以清洗/截句改写官方规则。实证说明旧用户工具的uppercase `Yes` token比较不适用于本轮输出；该旧工具未修改或运行。

四视频均按官方chat策略取实际帧`[0,24,48,72,96,120]`，覆盖124帧中的6帧。单模型评估全部四例，实际耗时 **73.295秒**（包括完整模型文件hash和加载），峰值allocated **24.646 GiB**，GPU0已释放。named tmux `research-E010-visionreward`保留退出状态。

两例没有统计泛化能力；相同分数不代表视频逐帧或感知等价，六帧辅助评价也不能代替连续观看。p30文字任务和p36产品细节可能受BF16本身生成能力限制，不能把双方共同失败归因量化。**音频未评价**。
