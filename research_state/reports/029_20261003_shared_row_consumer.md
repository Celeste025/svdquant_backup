# 029：共享校正行可以真实消费，但聚类尚未证明必要

2026-10-03。E033完成，独立CPU汇总完成。**K16共享中心已能在原生NVFP4 attention中直接消费；完整路径allocated峰值比自由块中心低21.7%，但局部误差更高，速度仅小幅变化。** 三层的五个方案均位于本次观测的误差—时延—显存前沿，没有全面胜出的方案，也未形成可投稿结论。

实现只改变校正表寻址：每个Qtile读C[id]对应的一行T，保留原QK/P/PV、TMA事务和同步。官方源码的小型私有副本使用独立JIT，原环境与既有消费者未改。三层共享表、block identity和global zero共5组同pack/table新旧对照，10次原生调用；全部输出相同，maxULP=0。这是本次工程观察，预定接受规则仍是最多一个BF16邻步，没有改成逐位门槛。

完整计时复用H3 p36/seed59526/step14的层0/24/48。输入是resident连续HND BF16 Q/K/V，终点是valid连续NHD BF16输出；每次包含Q/K/V各一次量化、中心/校正计算、必要复制和输出拼装，codebook每次重新做K16/6轮Lloyd。每层每臂3次预热＋10次测量，无CUDAgraph。表中时延是三个层各自中位数的范围，峰值为allocated，不能当成整模型或视频时延。

| 完整路径 | CUDA中位范围 | allocated峰值 | 保留global→block张量误差优势 |
|---|---:|---:|---:|
| global | 28.609–28.642 ms | 2.684 GiB | 0% |
| freeblock | 31.689–31.770 ms | 3.518 GiB | 100% |
| 精确Qchunk4096 | 33.912–33.957 ms | 2.601 GiB | 100% |
| 精确Qchunk8192 | 32.150–32.194 ms | 2.822 GiB | 100% |
| K16共享中心 | 31.480–31.538 ms | 2.755 GiB | 55.97%–71.74% |

共同resident baseline为0.91072GiB；codebook增量峰值1.84469GiB，freeblock为2.60775GiB，reserved峰值分别3.02148/3.85742GiB。共享表相对freeblock少0.763GiB allocated，但相对chunk4096多0.155GiB；其时延相对freeblock只低0.21–0.23ms（约0.66%–0.73%）。同轮重复范围未重叠，但只有一次固定顺序运行，不能据此宣称跨运行稳定加速。原event与同步wall全部数组已保留。

共享表最终三层NMSE为.00280712/.00804822/.00749995，与E032展开后旧消费者输出一致。全部168head优于global；相对freeblock，只有4head改善、164head更差。freeblock和两个精确分块输出相同。优势保留比例仅描述张量误差差值，不能当成视频质量保留率。

共478次原生attention、零DiT、零新视频，运行110.37秒含冷构建与I/O，worker3062111已正常退出，GPU0释放。[独立汇总](/home/wjq/workspace/svdquant-exp/results/research/E033/summary.json)重读25个输出、5组控制、15臂全部计时/显存和调用数。原validation receipt漏记三次means_and_key的预处理/均值计数，benchmark逐repeat计数及总native次数完整；未为修记录重跑或修改执行源。

下一步先补E034：相同16行表，按b_j=floor(j×177/16)把连续Qtile均匀分组，直接对每组padded Q求BF16中心。它无需聚类，是目前缺失的简单粒度强基线。固定同三层，global/freeblock/codebook/coarse16四臂完整路径，共156次attention；不扩K/视频/模型网格，不再重跑已验证的消费者控制。它与codebook的分组约束和拟合目标不同，比较回答实际取舍，不能伪称纯Lloyd消融。

经典聚类、centroid共享和更粗的中心粒度本身均不是新算法。只有先排除简单粒度解释，才值得决定是否扩大真实状态及视频验证；目前仍无等质量加速或可投稿核心贡献。
