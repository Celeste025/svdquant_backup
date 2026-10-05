# E068 — carry/restart 完整视频的任务相关性核验

2026-10-04；执行前计划。小规模探索，不是新方法实验或画质基准。

## 动机 / 观察

E065b的完整teacher-state video SSE四点变差，E066却确认相同teacher输入的全部局部层SSE改善。二者都是数值保真指标，不等于用户感知质量。在继续解释或“修复”SSE反转前，需要确认两份冻结配方生成的视频是否真有一致的质量差别。E067的保存向量几何也不能替代这个核验。

这不是根据坏结果扩大校准或迭代搜索。E065/b原停止规则已执行；本次另设任务相关性问题，不改变其来源结果、候选或成功条件。

## 假设、决策与评价边界

- 若隐藏方法标签的比较中carry明显更差，支持进一步调查这些案例上的任务相关退化，但不确定机制，也不外推总体。
- 若carry明显更好，停止把四点SSE反转称为画质损伤，重新审视用该代理值淘汰方法的作用。
- 若难区分或两例不一致，不宣称等效，也不继续仅凭SSE反转投入修复。
- 若历史BF16本身不满足文字/商品需求，共同失败不能归因量化。相对参照保真和相对prompt质量须区分。

主材料是完整配对视频，保留完整音轨，隐藏restart/carry身份；提供同条件BF16参照和原prompt。主判断需人工完整观看，偏好选项A/B/难分及自由说明。模型看帧、MJ或任何自动分数都不能冒称人评；本次不启动MJ或新视频打分器。尚无人工意见时只报告材料已完成，不填造偏好结论。两个原prompt、各一个seed、已经反复观察，只是探索性任务核验，不是新质量heldout或统计显著性样本。

## 冻结设置

使用E065b最终restart/carry各200个selected导出，原smooth、rank32、NVFP4 recipe、BF16支路、torch attention不变。两个原prompt及完整文本取E010锁定manifest：p30/seed49771，p36/seed59526。重用原E010 prepared PT中的BF16文本embedding/tags和video/audio初始CPU噪声，不重新编码或抽噪声。

原MiniMaxH3Pipeline.__call__、CFG1、双scheduler.step保持不改，20steps、124帧、576×1024、video/audio shift12/3；每臂从相同初态独立递推，绝不喂teacher后续latent。每臂先在p30/s5原teacher输入做一次native重放，实际input/raw/velocity须与E065b本臂逐byte一致，再生成两个完整视频。使用原pipeline传入prompt=None/text_embedding，避免重新拼接文本。

解码独立VAEs-only进程：原video VAE BF16、tiled=True/tile_size256/overlap64，原audio VAE 32kHz stereo，24FPS H264/AAC。两新臂使用相同环境/参数，保存实际PyAV/codec版本、PCM和完整媒体；验证124帧、1024×576、24fps、音轨、采样率。历史BF16视频与终态、prepared输入已fresh SHA通过，源与VAE size/mtime符合原完整SHA清单。历史未记录独立PyAV/libx264版本，不能保证编码字节与旧BF16完全一致，亦不声称本次重hash全部VAE权重。

## 实现、记录与复核

新runner scripts/research/run_h3_carry_pair_video.py；不修改已执行E010/E065/E066源或产物。阶段为CPUcheck、每臂denoise、每臂decode；准备数据、来源、manifest、所有selected导出及E065b独立报告绑定。拒绝覆盖。安装从原resident BF16直接install_native_h3读取E065b对应manifest，包含前39层旧E065绝对路径；不先安装legacy或引入中心化/FP4 attention。

每臂自由轨迹40次DiT＋一次历史重放；合计82full、16400native/pack、8364SDPA（每次由实际cu核对）、160次模态scheduler更新、4video＋4audio decode，0TE/0新BF16 forward。保存初态身份、原schedule、逐step before/prediction/after、原始DiT输出签名、最终latent、完整媒体/PCM、实际counts与finite检查。复核按缓存独立检查所有scheduler输出相邻衔接、初态/终态、文件hash与媒体信息；不冒称CPU重新执行DiT/VAE。

盲评材料使用中性A/B命名，生成前固定随机分配并将方法映射另存；所有四视频均展示，不选较好seed。页面不公开映射，BF16作为标明的参照。人评反馈到来之前不解读偏好；技术完整性和方法身份仍由私有清单可追溯。

## 资源 / 停止

启动前复查两个空闲SM120，各负责一臂的denoise→decode。两臂共用一次root指定的开始时间与**900秒绝对deadline**，decode继承原值，不重启预算；最多1800 GPU秒。历史native两视频约288.5秒，四视频VAE/编码约44.1秒，预计并行5–10分钟含I/O；不是本次性能预测结果。每卡峰allocated<60GiB，总新数据≤10GiB；代码/CPU检查在GPU窗口之前完成。

任何源/数值/重放/计数/资源失败保留现场；另版本修复须另说明资源，不静默延长或重新生成。两臂各用named tmux及独立监督退出码，阶段之间确认前进程成功并释放显存。

数据 /data1/models/svdquant-wjq/research/20261004/E068；小产物 results/research/E068；日志 results/logs/E068_*。本实验的贡献是防止错误追逐代理指标，普通生成工程和视频对比界面不算创新。
