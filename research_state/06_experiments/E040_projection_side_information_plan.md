# E040：同一FP4主支下的低秩信息来源控制

2026-10-03，exploration。E039四臂完成后固定，不改变其已执行源或结果。目标是判断其down32状态所保留的量化前信息是否有可观察价值；这不是性能、视频质量或新颖性实验。

E039候选完整边界比BF16回传快22.85%，但尚未回答普通FP4回传后解码输入再做LR是否足够。最小有效控制固定实际packed主支和原生main输出，只改变低秩支可用的信息。不得用旧最终输出减branch伪造精确main。

使用同一E039完整O[22592,56,128]、E009 block0 out_proj权重、两个原目的token段11264/11328；后者含53真实modelpadding。各段原BF16 smooth、原fast H3 encoder一次，保持原native配置与TF32关闭；真实M不补零。每段一个packet、一次实际native main，保存实际packet和main供独立审查。

同一个main各加三种BF16低秩branch：
1. E039 bf16_return保存的原full-K down；
2. E039 fp4_side保存的两源BF16 down合并结果；
3. 对该同一packet按已有独立signed-nibble LUT解码至BF16所得Xhat，新算Xhat乘原lr_a。
每个down都按原BF16 lr_b、1.0乘数及main+branch次序产生最终输出。前两者保存状态实际来自E039，不重新拟合或重算源partial；第三者只丢掉量化前输入。三种都共享相同main，不能额外量化低秩权重。

仅GPU0单次执行，300秒共同进程组期限含I/O/JIT，具名tmux与日志；总2次native GEMM、2次packet decode、2次新LR-down、6次LR-up、0通信/attention/DiT。无计时warm/repeat、无新kernel。保存三种末输出和实际down、packet、main到DATA1/E040（约1GB）；小结果/代码在仓库。保留失败，不改旧源。

CPU独立按有效22539／modelpad53／allreal分别复算三输出对原down参考的NMSE/cos/RMSE/maxabs，报告down差异及原E039两个输出的重放漂移。不同数值路径不设byte相等准入；若出现足以妨碍解释的重放变化，先报告原因，不以小测试反复重试。量化decoded-X控制的实际main固定可由保存main和输出合同核实。

决策：若decoded-X与side一样保真，状态必要性不成立，停止以此包装贡献；若明显丢失原native输出保真，确认这个具体接口中的信息缺失，下一仍必须面对成熟融合FP8/低位通信实现及跨输入、完整模型价值。单层精度信号不等于视频质量，线性代数结合律本身不是新颖性。
