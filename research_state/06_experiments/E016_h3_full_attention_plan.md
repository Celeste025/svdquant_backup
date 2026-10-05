# E016：现成 NVFP4 attention 的完整模型成本—保真对照

2026-10-02，GPU前预注册。探索/成熟强基线，**不是新算法、融合或组合贡献**。E014普通linear路径的BF16 attention占kernel时间58.7%；E006局部投影×attention交互未获支持，不等于现成低位attention没有部署价值。本轮固定SVD主层配方，量清两种官方attention实现的完整收益和代价，不扩大linear×attention全因子。

## 固定配置与执行语义

复用E014三个真实共同状态、原Comfy H3权重、200个SVD native linears、rank32/group16及原linear舍入/fastpack。非目标原dtype不变。四个evaluate臂：`bf16_original`、`svd_bf16`、`svd_block_mean`、`svd_global_mean`；仅后三臂benchmark。使用已验证E006 nvfp4-native环境，Torch仍同一个2.11.0+cu128路径，FlashInfer0.7.0.post1；不安装新环境或自写kernel。

用原50个main attention的bound forward作用域路由原Comfy helper，**不重写QKV拆分、QK norm、RoPE或out_proj**。main cu依次为[0,22384,22400]、[0,22227,22272]、[0,22539,22592]。只将首个有效长段N送入官方FP4 API，形状contiguous [1,56,N,128]，原scale=128^(-1/2)、causal=False、unpadded_k_len=N；官方内部padding后，裁Q回N并回填原位置。原padding段调用原helper的等价局部边界；两refiner完全走原函数。不可用长度猜层身份、不可silent fallback、不可将原padding当作有效key。

所有main模式（包括BF16原helper）统一使用预绑定CPU cu元数据，并在完整调用结束后以相同方式核验真实传入cu。只缓存既定边界，不缓存QKV/均值/scales/correction；这样避免原BF16每层GPU.tolist而低位臂免除此同步的不公平优势。refiner双方继续原函数。两级旧输出精确重放仍必须通过；这是普通metadata缓存强基线，不是新优化贡献。

低位臂每个DiT应有50次官方quantize/fwd、52次真实BF16 SDPA和200次FP4 linear GEMM；svd_bf16为102次BF16 SDPA/200FP4 linear；bf16_original为102/0。均零磁盘加载。Profiler应见实际SM120 E2M1 linear以及低位臂的50个FP4 attention kernel，不能只靠Python包装计数。

`block_mean`原样调用官方per_block_mean=True；`global_mean`为False。**False仍centering：K先按未padding序列求mean，Q按padding后全序列求mean**，不是no-centering。每臂每层必须用自己实际QKV重算mean/scales/correction，不能借用teacher/另一臂统计。两种模式代表两套官方完整数值配方与成本，不将差异独立归因统计粒度。固定TF32=False、原BF16 backend与所有模型设置。

## 两级重放和评价

同一新环境先完整BF16三状态raw/velocity逐byte等E014；随后svd_bf16三状态也逐byte等E014 SVD。第二项排除原生linear环境漂移。两级任一失败即停止；它们已经计入12次evaluate，不额外探针。

三state×四臂各一次完整forward，保存完整video/audio raw、适用时velocity、真实输入签名及执行审计。独立CPU FP64逐模态累计误差energy/reference energy/NMSE/cos等，相对同轮BF16；另报两FP4 attention臂相对svd_bf16的差异。既有E014 BF16/SVD结果双重绑定。全部输入/双输出保留，不能选有利状态或混合不同输入比较。E014这些输入均已看过，不称未见泛化质量集。

bench固定p1原shape、GPU5独立进程串行，三个SVD臂各1warmup+3repeat+1独立profile。所有warmup/repeat/profile输出须与自身evaluate SHA一致。计时包含原模型全部在线QKV布局转换、padding、centering、quantization、FP32 correction、native attention和回填，以及主层pack/smooth/LR。不得离线准备QKV/统计量。采用E014同步计时边界：记录CUDA与host forward时间，fastpack flags最后统一验证并单列/另报含检查时间；输出CPU copy/hash在计时外，runtime诊断hook仅evaluate/profile。若adapter加finite/源数据检查，须避免逐层CPU同步污染正式计时，并明确检查开销位置。

正式加速比只比较本轮新测的三个SVD臂。完整BF16本轮只作数值重放，不能借旧E014的BF16延迟拼接同环境加速比，也不追加该计时。

报告所有repeat与中位数、unique model storage、稳态peak allocated/reserved、启动峰值，profile单独列出。两种correction实际shape/bytes可报告，但必须同时测整模峰值；不能把单张量0.818GiB→4.79MiB称整模显存收益，也不外推长视频OOM。CPU mock路由仅验证切片/mask/调用合同，不证明GPU算术正确；False首次GPU使用是其完整evaluate。

## 决策与边界

若成熟低位attention已给出可接受的固定状态成本—误差取舍，更新部署基线，后续质量主张仍需另行实际生成评价。若有取舍，保留完整数据再判断是否有具体可研究的残余机制；不能因单纯更快/更慢、缺适配或默认模式多占显存而立新贡献。若ABI/边界/精确重放/finite或资源约束失败，停止记录，先排查实现，不把失败当算法阴性。

这是单DiT固定状态的数值与成本比较，不是视频质量、同步、长序列容量或端到端生成加速。SVD目前仍非充分融合的最优实现；本轮也不声称最强全部低位attention上限。

## 资源与产物

同一空闲GPU5串行：12 evaluate+15bench=27DiT，最多30；首次GPU起30分钟共享deadline、不重置；当前与历史allocated≤60GiB。长任务tmux、日志results/logs，大产物/data1/models/svdquant-wjq/research/20261002/E016，报告results/research/E016。先真实CPU输入/作用域路由检查，再冻结新源、官方API/JIT/相关CUDA headers及输入，root统一GPU启动。

[可行性核查](../00_state/E016_native_attention_feasibility.md)。E014/E015已完成与独立审查，是上一目标轮的实际progress；本轮继续原顶会研究目标，不把完成强基线替代论文贡献。
