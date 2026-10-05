# E051 首四条完整视频的运行成本读出

2026-10-03；仅读取 E049/E051 的 `worker_161.json`、`worker_192.json`、launcher 元数据及已执行源码，未读取任何质量评分、未运行 GPU。本记录只覆盖两个 prompt×两个 seed，E051 其余样本尚在执行。

| 配对样本 | E049 BF16 pipeline秒 | E051 plain pipeline秒 | BF16/plain耗时比 | 耗时减少 |
|---|---:|---:|---:|---:|
| vbench0161_r0 | 911.356 | 727.026 | 1.2535 | 20.23% |
| vbench0161_r1 | 911.705 | 726.190 | 1.2555 | 20.35% |
| vbench0192_r0 | 930.380 | 726.387 | 1.2808 | 21.93% |
| vbench0192_r1 | 931.824 | 726.441 | 1.2827 | 22.04% |

四条 **pipeline时长之和**为3685.265→2906.044秒，比值1.2681、减少21.14%；这是配对样本时长汇总，不是并行任务的总wall或服务吞吐。prompt161两臂均物理GPU0、192均GPU1，型号相同RTX PRO 5000 72GB/SM120。每例均100 DiT、8000 BF16 FLASH SDPA、50 scheduler、1次FP32公共VAE decode、0 TE；plain另有40000真实native GEMM及40000 activation pack。输入、尺寸81×480×832、50步CFG5/shift3及解码合同相同。

**计时范围。** [BF16 worker](../../scripts/research/run_wan14b_reference.py:230)和[plain worker](../../scripts/research/run_wan14b_plain_native.py:291)都在pipeline前后CUDA同步，以host单调时钟测量。包含完整去噪、FP32 VAE decode、pipeline后处理转numpy，以及每步finite/RMS/min/max归约、末步CPU latent复制和原始VAE输出检查；不包含加载、plain安装、计时前输入签名计算、计时后final文件保存/hash、MP4编码、contact sheet及JSON写盘。plain还包含每DiT400层计数检查、400组flag合并读回、每GEMM dtype/计数检查等额外instrumentation；没有单独测量或扣除这些开销。首forward的潜在JIT/冷缓存开销也未隔离，没有预热/重复稳态benchmark。两组独立任务的整体并发安排不同（E049四卡同时worker，E051本阶段两卡生成），没有控制时钟/热状态/共享host负载。因此可称本次实际完整pipeline耗时较低，不能归为纯GEMM加速或成熟优化serving结果，更不能宣称质量匹配。

| 内存口径，GiB（2^30字节） | BF16 | plain |
|---|---:|---:|
| 加载/安装后模型registered parameter+buffer去重storage（DiT+FP32VAE，无TE） | 27.142 | 8.331 |
| 加载/安装后实际allocated | 27.242 | 8.405 |
| 加载/安装后实际reserved | 27.262 | 27.623 |
| 每条完整pipeline allocated峰值 | 40.0345–40.0347 | 21.19714–21.19717 |
| 每条完整pipeline reserved峰值 | 48.8301 | 48.2031 |

registered模型storage减少69.30%，完整pipeline allocated峰值减少约47.05%。每例前`reset_peak_memory_stats()`重置峰值计数；它不清空allocator，第二seed也继承第一seed保留的reserved块。因此reserved是实际进程allocator历史的峰值，不能据此推出最小必需显存，也不能把storage下降直接说成同幅度整模峰值下降。

**转换与推理不可混合。** plain由完整BF16 teacher驻GPU后逐层pack，两个worker的转换期allocated峰值均 **27.3536GiB**、reserved峰值27.6230GiB；这是包含加载历史的“自进程上次reset以来”峰值，不是安装增量。其最大单层FP32 temporary为283115520字节（270MiB），400层原W释放28101836800字节；没有保留整模FP32 master。转换完成allocated降至8.4052GiB，随后每例才reset推理峰值。因此本次plain加载→转换→推理所见最大allocated为27.3536GiB，而非仅21.1972GiB；reserved最终仍为48.2031GiB。未说明loader/offload优化后的最低容量需求。

两个worker的load+H2D为BF16 8.946/9.118秒、plain 8.114/9.434秒；plain安装（含逐层pack flag验证）2.196/0.960秒，load+conversion为10.328/10.409秒。完整worker总时长分别BF16 1834.523/1873.884秒、plain1466.192/1465.837秒，包含两seed及文件/媒体等工作；这些不同进程的一次测量不是加载或转换性能稳定性证据。

来源：[E049/161](../../results/research/E049/worker_161.json)、[E049/192](../../results/research/E049/worker_192.json)、[E051/161](../../results/research/E051/worker_161.json)、[E051/192](../../results/research/E051/worker_192.json)。本记录不讨论输出质量，plain亦非SVDQuant强基线。

## 全八例完成后补充（23:49）

余下四条 BF16/plain 秒：汽车r0 936.748969/726.262054，r1 938.477029/725.983705；大象r0 925.094051/726.065440，r1 926.611369/726.722123。全部八条时长和7412.196392/5811.077265秒，比1.275529，减少21.6011%；推理allocated峰仍为40.034483–40.034718 / 21.197140–21.197174GiB。原计时、并发、reserved及安装峰值限制不变。完整E051质量已读：MJ均值下降.096401，不能宣称等质量加速，见报告048。
