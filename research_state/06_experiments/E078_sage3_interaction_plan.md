# E078：官方SageAttention3与正式SVDQuant基线的组合

用户明确建议的现象探索，不将两种现成方法组合称创新。问题：SVDQuant的完整主干量化改变传入attention的特征后，官方SageAttention3是否产生比独立误差叠加更大的损伤？此前E006只有量化QKV投影＋FlashInfer FP4，不能回答官方Sage3＋完整主干的组合。

先获取官方thu-ml/SageAttention，pin d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5，原样build_ext；native现有Python3.12（README建议>=3.13，setup允许>=3.8），记录实际能否编译/运行，不为此升级共享环境。新源位于独立DATA1目录，PYTHONPATH导入，0原环境依赖改写。编译上限600秒。

四角：B0原BF16＋原SDPA；BA原BF16＋Sage3；S0用户冻结E009 legacy SVDQuant＋原SDPA；SA相同SVDQuant＋Sage3。保持50主block线性层200个、rank32、smooth不变；仅主block有效长段用Sage3，两个refiner与独立padding短段保持原SDPA。保持QK norm/RoPE、head_dim128、原scale、非因果。Sage3官方preprocess会原地减K均值，wrapper必须独立clone K，以免污染输入/后续重放。原始N不得裁尾；官方自行padding并传入真实KL。

固定两原完整teacher状态p30/s05、p36/s14，实际输入来自E065b保存、BF16及S0 raw参考来自原已校验输出。四臂各2次共8DiT；B0/S0逐byte复现是必要有效性检查。逐模态保存raw完整输出并独立CPU复算。

定义参考误差eS=S0−B0、eA=BA−B0、eSA=SA−B0；交互I=SA−S0−BA+B0。主读出R=||eSA||²/||eS+eA||²。R>1才代表相对实际独立误差向量和的净放大；交互能量大本身不能说明有害。另报SA/S0、||SA−S0||²/||BA−B0||²及cos(eS,eA)。同时看video/audio，不把attention单独坏误认组合协同损伤。两状态只探索，prompt/时间混杂，非显著性或画质结论。

若两点video净放大均>10%，进一步定位输入分布/量化中心或P×V的竞争机制，先在同输入做实际干预；如果只有一点阳性则报告不一致，不直接宣布机制；都≤1则本设置不支持放大，不开展任意组合网格。无论何种结果均回答用户的比较请求。若官方backend不可用，报告具体阻碍，不用FlashInfer冒充。

最低smoke：BF16非因果head128在不整除128的N=257、与实际H3长序列运行finite；官方入口实际调用fp4attn_cuda.fwd，禁静默fallback。模型GPU预算600秒、<60GiB、<500MiB新输出，编译另600秒。0新校准/训练/视频。需要时另立完整视频实验，当前误差不等于视觉损失。

接入修订（GPU模型实验尚未执行）：官方setup的CUTLASS git clone约4分钟无进展，停止本任务下载/构建进程，保留未完成目录和日志。改用本机FlashInfer包附带CUTLASS4.5.0头文件（只读symlink），原Sage源码/setup不改；追加最多300秒编译，不声称原600秒总预算未变，先前下载时间与重试分列。实验源码未执行前改记录dependency来源而非缺失的git metadata。

接入修订2（模型0调用）：终止git后其自身清理了未完成目录，故仅日志保留，之前mv未成功，symlink成功。缓存依赖构建发现系统nvcc13.0与torch cu128冲突，未进入编译。下载NVIDIA官方12.8.1 redistrib的nvcc/cudart/cccl三个压缩包共约81MB，SHA256全部验证，在DATA1独立toolchains目录解包。最终匹配工具链构建预算300秒；前述失败/下载另记成本，不覆盖日志。Sage源码未修改，不绕过版本检查，不改系统CUDA/PyTorch。

接入修订3：最小工具链不含cuBLAS/cuSPARSE/cuSOLVER头文件，torch公共头依赖这些；初次cu128编译在缺cusparse.h处终止。通过CPATH引用现有同一cu128 torch环境的nvidia包头文件，剩余240秒重新编译，Sage源码无改动。
