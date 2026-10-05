# E009 零 scale 兼容域恢复计划

保留原 `E009_h3_full.json` 的失败状态与冻结源码。它已完成全50block resident/offload逐byte一致、200W exact、完整旧公式QDQ与四真实linear门槛；native在block1.fc2非零组SF0处按strict guard停止。本文件不修改原预注册或其哈希。

真实失败输入已在 `E009_h3_sf0_domain_v2.json` 复现：block0 SHA与原失败轨迹相同，95组SF0均在audio，common与E005 canonical decode共321,126,400元素数值完全一致，非零符号以外byte差0；真实fullshape native主支NMSE=4.56099e-6且有限。SF0丢失能量同样存在于旧量化公式，不能解释为硬件新增失真，也不据此作研究机制主张。

恢复流程固定同p1step0、完整token、同torch BF16 SDPA：

1. 校验原报告及其所有source/model/state/rawinput哈希，继承已完成的三个reference阶段，明确未重跑这些forward。
2. 新独立 `h3_nvfp4_zero_sf_compat.py` adapter不改被冻结的原量化器/导出。先在保存的失败xs上要求slow E005与fast codes/global/含padding SF逐byte一致；flags必须checked1、invalid0、SF0 affected1。
3. 新进程重新load、resident化、安装200旧export；非目标tensor哈希须与原参考相同。一次完整slow native；block0必须仍与原strict轨迹SHA一致，200真实FP4 GEMM/102实际BF16 SDPA、全50block与video/audio有限；逐层和endpoint比较复用原BF16/旧QDQ参考。逐call记录slow SF0组数。
4. 同进程切换新fast adapter，再一次完整forward；逐50block和双endpoint SHA须与刚生成的slow native相同。collector退出后200flags验证，invalid仍拒绝，SF0仅按受影响call记录，不能把call数称group数。

GPU1先检查空闲，named tmux、日志、timeout3600；显存门槛60GiB。不新增prompt/step、不重校准、不解码视频、不据诊断计时报告性能。源 `resume_h3_native_zero_sf.py`；输出 `results/research/E009_h3_native_resume.json`，大张量在 `E009/native_zero_sf_resume/`。通过后仅为systems后续独立性能测量提供正确性前置。
