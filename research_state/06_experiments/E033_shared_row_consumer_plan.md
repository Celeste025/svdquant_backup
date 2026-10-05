# E033：封顶一个共享校正行消费者原型，检验完整路径

2026-10-03，GPU前固定。E032的K16 codebook准备约8.49ms，与freeblock8.43ms接近，准备allocated峰值下降约25.7%；全部168head优于global，三层保留global→block张量误差优势66.06%/55.97%/71.74%。它精度低于range1，但每Qtile只需选择一行T。尚不能据准备结果宣称完整收益，因此只实现一个最小consumer原型，不扩cluster/rank网格。

私有代码与唯一JIT namespace/module，原env/site-packages/已执行源保持不变。官方SM120源29headers约332KiB复制到DATA1并保存来源manifest；仅binding、params、launcher、mainloop四处语义修改。新API codebook_fwd(q4,k4,v4t,qsf,ksf,vsft,table,ids,*,sm_scale,unpadded_k_len,out=None)，仅D128/BF16输出/noncausal/return_lse=False。table=[B,H,K,Npad] FP32，ids=[B,H,G] int32；DS描述符逻辑M设128*K沿原zero-stride布局，Q真实长度/调度不变。producer把原DS行m_block改为ids[b,h,m_block]。K可1/16/G，TMA类型、512B事务、shared、barrier、QK/P/PV和consumer加回顺序全部保留。官方原模块仍负责量化。

数据仍只用三层E018 p36/s14原QKV及E032实际中心/id，无DiT、新视频或更长序列外推。正确性先用同一实际pack/table比较：三层codebook新consumer对旧gather完整corr各一次，共6native；block0的identity映射（G行、id=arangeG、block Q中心）和zero映射（1行、id=0、global Q中心）各新旧2次，共4native；总10。ID值域/shape及CPU标量指标在计时外。保存全部输出和数值差。

正确性合同是仅改变数据寻址而保持公式：finite、valid shape必须满足。新旧输出允许至多一个相邻BF16可表示步长的差异（±0合并）；报告maxULP、count>1、maxabs、pooled/perhead误差，不要求逐位相同。若>1步，则保存已执行输出并在benchmark前停止，检查寻址或编译数值变化；这不是方法的科学阴性，也不自动调宽门槛。任何修改另立明确版本。比较只对同pack/table原生新旧，绝不要求它等同高精度模型。

正确性通过后完整resident QKV→valid NHD BF16 output测量，三层各五臂：global、freeblock、exact Qchunk4096、exact Qchunk8192、codebook consumer。每臂3warm+10repeat，无CUDAgraph/compile wrapper。每repeat包含官方BF16预处理、Q/K/V各一次pack、必要KᵀFP32复制/校正GEMM或真正重做K16/6Lloyd+T16、原生attention、必须的contiguous copies及输出拼装。codebook不能复用旧C；chunk不得预先物化完整correction，也不得重复KV pack。SF切片沿原已验64-row独立slab，4096/8192边界均对齐128；尾分别2176和6272padded rows。输入上传/layout转换、CPU验证/保存不计时。

每臂最终输出保存，计时外验证指标；记录原数组CUDAevent和同步wall、resident/warm allocated/reserved基线、每repeat resetpeak/增量。三层13repeat×(1+1+6+3+1)=468 benchmark attention，加10 correctness=478；0DiT。baseline只计算自己所需均值，三臂自由block中心的完整输出应与原block数值一致，codebook与E032数值对齐情况完整报告。比较实际全流程时延/峰值/精度，不拼接独立计时预测收益，不把T容量当GPU总峰值。

GPU0单一root tmux/日志/900秒硬预算（包括首次私有JIT，较此前600秒增加仅为新构建），CPU source/interface/结构检查和独立审查在先。失败、旧检查及源均保留，不覆盖旧GPU结果。若最终被global及精确分块实际前沿支配，停止该实现路线而非扩basis/cluster变体。即使得到新取舍，也只是三层单状态的工程证据；新颖性、跨模型/长度/质量仍须另行证明，经典kmeans和centroid共享不构成论文贡献。
