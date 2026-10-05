# E009 H3 resident profile 协议

2026-10-02，frozen / ready to execute。E009完整slow/fast native证明已完成；三arm CPU-only binding全部通过，root已授权直接启动GPU5串行测量。部署工程基线，不作研究贡献或生成质量结论。

固定p1 step0原PTQ calibration raw cache，Comfy-pruned H3、50主干blocks/200目标linears，8个refiner linears与非目标保持原精度。复用 `bench_h3_native_nvfp4.make_h3_resident` 的已验证迁移；其最终source hash必须与完整correctness一致。torch版本、明确torch SDPA实现与flash/math/mem-efficient/cuDNN enabled配置一致，不额外切换attention kernel。

三个arm在**同一空闲GPU5、串行独立进程**执行，不混用不同卡比较：

- BF16：原权重全部resident，无disk-onload。
- QDQ：export中exact旧量化W只在startup解码一次，建立普通nn.Linear+原common smoothing/QDQ/BF16 LR hooks，权重常驻。禁止用每forward临时decodeW的reference模式作为计时baseline。
- Native fast：相同export codes/SF/global、相同BF16 smooth/LR，H3 fastpacker+torch原生scaled_mm。使用独立zero-SF compatibility adapter：保留原fast helper flags0非法拒绝，显式允许并统计已证实数值等价的flags1（非零输入SF0）。不加入FlashInfer融合，不修改既有配方。

每arm先1次完整warmup（允许JIT，不进入steady），再3次无profiler完整DiT重复，最后1次独立profiler。warmup、每次重复、profiler的video/audio输出必须精确匹配对应参考SHA；首个不符即保存失败、停止性能结论。BF16/QDQ继承 `E009_h3_full.json` 的三个完整阶段，保留该报告整体failed_stop状态；native从 `E009_h3_native_resume.json` 新slow阶段取参考。新报告必须以SHA绑定旧失败报告和SF0真实证明，并通过全部50个block和两个endpoint的slow/fast SHA相等检查。compat/E005原slow/original fast/原native模块的source SHA全部绑定，新files与inherited_reference.source_files共同核对，不宽松接受任意status=complete文件。前置fast检查为200 calls、invalid0、affected-call索引[7]；这不改变95个SF0 group来自真实失败输入的独立计数定义。

Native每次forward使用compat collect_fastpack_checks：CUDA结束event同步置context内部，scopeexit统一验证200份flags。每次保存summary的affected-call数量及索引（不是SF0组数），非法flag仍停止。分别报告CUDA event、forward host、flags验证耗时以及**包含检查的完整host wall**。输出CPU复制/SHA在计时外；计时内不采集50个block、局部误差或逐层CPU输入。禁止隐藏检查成本。

显存记录startup/load/convert峰值、warmup后reset的3次steady allocated/reserved峰值与前后状态。常驻model storage去重统计registered参数/buffers及hook LR/smooth，不能漏hook外的低秩分支，也不将其冒称全部CUDA常驻开销。

独立profiler的语义范围：dense main、native GEMM、smooth、pack、旧activation QDQ、LR、attention；LR包含两次GEMM和原1.0 multiplier，最终main+LR add统一留在other。这些子范围不嵌套，完整forward父范围不再累计。通过Chrome trace严格cat=kernel与唯一launch correlation→CPU语义范围归因；gpu_user_annotation排除，copy/memset另列。work总量不是关键路径时间，不能机械外推加速。patched profiler前向也要通过相同endpoint SHA。

Chrome trace/endpoint文件写 `/data1/models/svdquant-wjq/research/20261002/E009/profile/`；聚合JSON写 `results/research/E009_profile_{bf16,qdq,native}.json`。所有已有结果拒绝覆盖；named tmux、results/logs独立日志。每arm timeout 1800秒，不运行rollout/text encoder/VAE、参数扫掠或额外quality测试。

启动入口（尚未运行）：

```text
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/profile_h3_native_nvfp4.py --arm bf16 --fast-proof /home/wjq/workspace/svdquant-exp/results/research/E009_h3_native_resume.json
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/profile_h3_native_nvfp4.py --arm qdq --fast-proof /home/wjq/workspace/svdquant-exp/results/research/E009_h3_native_resume.json
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/profile_h3_native_nvfp4.py --arm native --fast-proof /home/wjq/workspace/svdquant-exp/results/research/E009_h3_native_resume.json
```

环境沿E009：`DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio`、`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`、`MINIMAX_H3_DIT_PATH`指同pruned文件、recovered Python及同DeepCompressor PYTHONPATH。先核查GPU空闲和完整前置报告状态，再冻结脚本/协议启动。
