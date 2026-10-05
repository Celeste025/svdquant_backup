# E020：rCM-Wan 主权重 QAD 的首个训练—导出闭环

2026-10-03，训练前记录。上一轮为progress：定位并量化LSE路径差异、完成报告018与主权重训练就绪检查。顶会目标仍active，无已成立claim。本轮直接实施成熟强基线，不将普通QAD/STE/去在线LR当新方法。

固定本地rCM-Wan transformer（实际1,418,996,800参数），30block各10个Linear主矩阵可训练，共1,391,984,640 FP32 master参数。其余权重、bias、norm维持原BF16，不继承旧extra INT4 hooks；无smooth、无LR。从完整原W开始，不能删除SVD残差图的LR冒充原W。W/A使用同一已验证legacy-Wan NVFP4动态两级尺度合同：group16、FP32 tensor scale、E4M3 ties-up、signed E2中点取较大有符号值。FP32 master直接编码，导出前不能先转BF16。

训练前向是实际QDQ值的BF16 linear，反向为常规identity STE；不开发FP4 backward。native部署消费完全相同的packed W和在线A配方，主路径实际scaled_mm，QDQ与native累加差异单列，跨实现不强求全输出byte exact。BF16 attention统一用torch FLASH_ATTENTION，保持完整31,200 video tokens；不裁分辨率/长度以通过显存检查。

数据采用E020_cache_inventory.json预先固定的4个开发训练prompt×4step、2个开发验证prompt×4step。它们来自旧缓存，明确不是独立最终测试；现场原rCM BF16 teacher重算每条目标，不假定缓存outputs的来源。目标为同输入单步denoiser输出NMSE，4步均匀采样。首段64次AdamW更新，lr=1e-5、weight_decay=0、grad_clip=1、seed=20261003，batch1、block checkpointing。第0/16/32/64步保存验证读出；保留初始与末尾packed模型及FP32 master/optimizer checkpoint，可续训而非重复初始化。当前NMSE仅为闭环与学习诊断，不是生成质量结论。

先做一次小矩阵真实GPU检查：BF16包与既有packer一致、FP32 master不预先BF16舍入、STE前向与独立decode一致且梯度流向主W、无LR native确实调用scaled_mm。再执行上述固定训练和初始/最终native完整验证。必要的输入/源/数据/checkpoint manifest和失败日志各保留一份；不复制多级88文件冻结。新增源允许在CPU/GPU smoke发现实现错误时显式修订并保留失败，训练开始后记录使用版本。

资源：GPU5启动前稳定空闲/外部进程检查，tmux，日志results/logs/E020*；45分钟首段训练总上限，60GiB allocated告警/停止，不干预其他用户卡。大产物/data1/models/svdquant-wjq/research/20261003/E020。实际训练OOM/非有限/梯度断开时保存失败并修实现；无任意收益百分比/teacher语义门槛。后续用未用于训练/PTQ的prompt与新seed执行完整4步自由生成及时间质量评价，先不扩大到H3训练或修改attention/步数。
