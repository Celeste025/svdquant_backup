# Wan14B完整PTQ的阶段依赖与可并行范围

2026-10-04，systems只读定位、root核读实际源。用于E052结束后的执行组织；不是新研究claim，也未实施/启动分层并行。

**阶段内的输入不随已完成的上游校准更新。** Wan结构在[struct.py:1708](../../third_party/deepcompressor/deepcompressor/app/diffusion/nn/struct.py#L1708)返回40层、recomputes全False、use_prev为[False,True×39]。基类[cache.py:397](../../third_party/deepcompressor/deepcompressor/dataset/cache.py#L397)先执行当前层并把输出保存为下一层输入，418才yield。调用方随后在[smooth.py:662](../../third_party/deepcompressor/deepcompressor/app/diffusion/quant/smooth.py#L662)或[weight.py:377](../../third_party/deepcompressor/deepcompressor/app/diffusion/quant/weight.py#L377)校准当前层。因而同一阶段的X(k+1)来自该阶段入口的F(k)，并非校准后的F(k)。

**必要阶段屏障。** smooth入口为原模型。全部40层smooth完成并合并/应用之后，LR必须新建loader重采完整平滑模型的输入。LR全量完成、合并之后再量化权重和导出。保持每block内部原搜索顺序及共享QKV/KV组。E052 smooth后重采block0仍遵守这一局部依赖，但partial不能直接喂给原ptq冒充全量。

**分层并行的条件。** 可让每worker保留完整iterator/prefix传播，仅校准所负责层；每层仍收到该阶段入口模型产生的真实输入和kwargs。不能把block0输入复用给其他层；不能非连续筛选层列表后仍令use_prev=True而跨过中间层。是否采用这种执行组织须等E052完整资源结果，尚无分层worker/实验编号。

随机SVD的全局RNG消耗受早停/层顺序影响，跨worker不能承诺逐位复刻旧串行。若采用明确的per-layer随机政策，必须记录这一差别，保持64/b4/g10/r32/LR100原早停/OutputsError及原共享组，不把比特相同作为科学等价的唯一标准。

单个64条BF16隐藏状态边界约20GiB；E052现已测得内部activation采集CPU峰236.736GiB、GPUallocated峰42.135GiB。并行可行不意味着能随意开多个worker，须看完整LR峰与服务器其他占用后决定最多两个worker是否合理，不从首两搜索线性承诺整模耗时。

**00:06资源更新：** 首block完整smooth已963.0823秒完成（7共享组×19候选），GPUallocated峰61.8546GiB、CPU RSS峰367.4615GiB。这高于仅cache的236.7GiB，两个此类峰的简单叠加约734.9GiB，已接近整机742GiB、未计其他进程。不能从阶段独立就直接开两worker；现有loader还会在主机使用率>90%时主动失败。LR仍待实测，当前不实施并行。
