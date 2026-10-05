# E043 evaluation adapter

2026-10-03。只准备入口；`evaluation_check.json`已在MJ环境CPU运行通过，CUDA未初始化、0模型加载。未合并尚未完成的worker、未执行媒体验证或GPU评价。

新入口：[vanilla_wan_evaluation.py](/home/wjq/workspace/svdquant-exp/scripts/research/vanilla_wan_evaluation.py)。四个complete worker按冻结manifest顺序合并8条，核实际MP4 SHA/bytes；无样例过滤。三项时序指标直接运行冻结E024的81帧循环，仅替换输入绑定与报告说明，**不继承E024产品身份或生成协议**。本地AMT/RAFT/DINO权重与公式继续沿E022。旧源/worker/媒体不修改。

- MJ：8帧 `[0,10,20,30,40,50,60,70]`，来自原`get_index(0..80, endpoint=False)`，不是9帧contact sheet。保留全部28criteria/5aspects，主读出total/alignment/fineness/coherence；safety/bias仅raw，不用来筛样。
- AMT：41个偶数帧插值，和40个真实奇数帧比较；无补帧。
- RAFT：16fps下间隔2，共41帧/40flow，20次迭代；原阈值11.25，原数量阈值`round(4*41/16)=10`。
- DINO：全81帧，逐视频80个相邻/首帧余弦平均；同prompt两条真实seed一起调用，不复制媒体。
- 480×832可被16和8整除，AMT scale=1与RAFT均无空间padding；AMT实际运行scale由旧循环记录，原resize/normalization不变。

四份worker都complete后，CPU命令：

```bash
CUDA_VISIBLE_DEVICES='' /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python /home/wjq/workspace/svdquant-exp/scripts/research/vanilla_wan_evaluation.py --phase prepare
CUDA_VISIBLE_DEVICES='' /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python /home/wjq/workspace/svdquant-exp/scripts/research/vanilla_wan_evaluation.py --phase temporal --check-only
```

分别生成`results/research/E043/evaluation_manifest.json`和`temporal_cpucheck.json`。CPU结构检查结果为[已有evaluation_check.json](/home/wjq/workspace/svdquant-exp/results/research/E043/evaluation_check.json)。

以下为root监督进程内的GPU命令，**本次未启动**。外层选择空闲GPU并设置`CUDA_VISIBLE_DEVICES`、DATA1缓存与`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；temporal建议900秒、MJ600秒，截止时终止整进程组。`E043_DEADLINE`由外层设置为绝对Unix截止时间。

```bash
/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python -u /home/wjq/workspace/svdquant-exp/scripts/research/vanilla_wan_evaluation.py --phase temporal --deadline-unix "$E043_DEADLINE"

MASTER_PORT=29643 /data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python -u /home/wjq/workspace/svdquant-exp/scripts/research/eval_mjvideo_e021.py \
  --manifest /home/wjq/workspace/svdquant-exp/results/research/E043/evaluation_manifest.json \
  --samples /data1/models/svdquant-wjq/research/20261003/E043 \
  --model /data1/models/svdquant-wjq/models/MJ-VIDEO-2B \
  --tokenizer /data1/models/svdquant-wjq/research/20261003/E021/tokenizer \
  --mjvideo-repo /data1/models/svdquant-wjq/third_party/MJ-Video \
  --output /home/wjq/workspace/svdquant-exp/results/research/E043/mjvideo_scores.json \
  --variants vanilla_wan_bf16 --video-template '{case_id}/video.mp4' --num-segments 8
```

temporal输出`temporal_scores.json`，含每例原始AMT/RAFT数组、DINO原sum及实际输入绑定。MJ旧入口按case逐条落盘；若进程失败保留部分结果，不能当作8条complete。没有添加总体质量合成分数或teacher评级门槛。
