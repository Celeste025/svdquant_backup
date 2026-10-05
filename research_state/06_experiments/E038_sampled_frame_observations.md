# E038固定抽帧观察

2026-10-03，root在全部媒体完成、MJ原始评分已可见时查看，非盲评。每case三行BF16/global/coarse16，固定帧0/18/35/53/70/88/105/123，共8张图、24条视频的192帧。只有静态抽帧观察，无完整播放或音频评价。下述描述不独立建立时序机制或质量排名。

- 拍手r0：三臂均为深色背景中的双手近景，手掌姿态随抽帧变化；BF16轮廓较清楚，两FP4臂可见更强颗粒/重影样纹理。没有看到coarse明确消除global的问题。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0161_r0.png)
- 拍手r1：三臂均为黑衣人物的手部动作；global手部颗粒/重影明显，coarse若干帧较清楚，但仍有异常纹理且构图、动作幅度改变。不能从八帧判断完整拍手频率或动作连续性。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0161_r1.png)
- 叠衣r0：三臂都展示人物操作衣物及衣物堆，视角/照明和布料形状有所改变；八帧不能确认完整折叠动作的物理正确性。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0192_r0.png)
- 叠衣r1：人物在床面操作深色衣物；coarse的床面褶皱与人物位置变化较明显，局部细节呈不规则纹理。与BF16构图不同本身不等于错误，也不单凭这张图给总体排名。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0192_r1.png)
- 汽车r0：三臂均为弯道上的车，抽帧中可见沿弯道的位置/朝向变化；相机和主体尺度不同，没有据此确认某臂失去“转弯”语义。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0269_r0.png)
- 汽车r1：三臂均有道路转弯相关视角变化；coarse车身色彩不同，但prompt未指定颜色，不把颜色偏离BF16当错误。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0269_r1.png)
- 大象r0：三臂均可见象鼻抬起/弯曲和水雾；global较正面且主体尺度不同，BF16/coarse较侧面。构图接近BF16不能单独证明coarse质量更好。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0316_r0.png)
- 大象r1：三臂均为水边的大象，若干时点可见局部鼻部姿态及水雾变化；全局RAFT静态判定不能解读为没有局部动作。[图](/data1/models/svdquant-wjq/research/20261003/E038/contacts/vbench0316_r1.png)

全部媒体与图的对应摘要见 results/research/E038/contact_sheets.json；所有例子保留，未事后挑选输出。
