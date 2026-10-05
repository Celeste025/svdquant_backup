# 原版 Wan native W4A4 成对 baseline 入口

2026-10-03。只读检查；没有 GPU、重新校准、模型大文件哈希或修改执行源。**现有 `wan2.1-1.3b-real-nvfp4-s16` 有充分证据属于原版非 rCM PTQ；最短路径是保留 E043 的 Diffusers 0.40 pipeline，只在其 transformer 上加载这份 PTQ 状态，再调用 E007 native 转换与既有 fastpacker。尚未验证这份 checkpoint 在 0.40 下完整加载/运行，不能直接把旧 rCM 的执行结果算作验证。**

## 已核实的来源与合同

- checkpoint 根为 `/data1/models/svdquant-wjq/ckpts/wan2.1-1.3b-real-nvfp4-s16`。`model.pt/scale.pt/wgts.pt` 实文件与 `smooth.pt/branch.pt` 相对软链均存在，后两项解析到 `runs/wan_s16_real_nvfp4`。原成功运行 `run-260826.234259` 的 config/log 明确记录基础 `Wan2.1-T2V-1.3B-Diffusers`、BF16、`shift_activations: false`、UniPC50/CFG6/33帧，校准路径 `datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16`、64 cache records；不是 rCM 四步来源。该 checkpoint 没有现代 manifest，不能补称当年全权重来源已逐字节绑定。
- CPU mmap 读取状态元数据，并小片核对三个不属于 300 主量化层的权重：patch embedding 首行64、time embedder 首行256、proj_out 首行1024元素，cast BF16 后与已验原版权重全部相等；对应 rCM 分别不同33/242/937元素。它与日志共同支持原版来源，不是用全模型重建误差反推来源。
- `wgts.pt` 恰为 **30 blocks × 10 Linear = 300**：self/cross attention 各 q/k/v/out，加 FFN 两层。CPU meta 构造的 Diffusers 0.40 原版模型中，全部名称存在且仍为标准 `nn.Linear`。这300个 saved residual W 均BF16；完整825项 model state 另有95项FP32，加载时须保留当前原版模型的 dtype 规则，不能粗暴整个图 `.to(bfloat16)`。
- `scale.pt` 900项：每层 outer-global、group scale、zero；保存容器均FP32。300个 outer-global 都有限正值（范围 `9.536743e-6`–`5.987259e-4`），zero全0。局部group16的SF虽以FP32存储，其合同是E4M3值；抽查 block0 q/k及block15 fc2可无误差转E4M3，**不是已经验证全300层roundtrip**。后续现成转换器本身会逐层检查，不要重算weight scales。
- 210组BF16低秩状态，rank32；例如 self-QKV 共享 `A[32,1536]`，保存的拼接 `B[4608,32]`；out-proj `B[1536,32]`。211项FP32 smooth。共享A对象不等于已融合一次down GEMM，维持已有hook语义即可。原recipe为W/A E2M1、FP32 tensor-global + E4M3 group16，A动态、W静态，既有Wan codebook ties规则；不要另改为H3舍入或笼统声称标准RNE。

## 可复用接口与实际适配点

1. 继续 E043 同一 base、TE embedding、CPU initial noise、FP32 latent/update、UniPC50/CFG6/shift8与FP32VAE。只调用 [load_quantized_transformer](/home/wjq/workspace/svdquant-exp/scripts/infer_rcm_wan_4step.py:24)(pipe, **原版checkpoint**, BASE)，不要调用其 rCM main/sampler。该函数名/文件虽含 rCM，实际默认 recipe 即 `real_nvfp4.yaml + wan_s16.yaml`，已支持 packaged smooth/branch。显式设置 `SVDQUANT_DATA_ROOT` / `RCM_RUNS_ROOT` 到新 DATA1 工作目录，避免默认 scratch 写入旧路径。
2. 接 [convert_wan_transformer_to_native](/home/wjq/workspace/svdquant-exp/scripts/research/wan_native_nvfp4.py:324)，使用它既有300权重roundtrip、saved residual一致性、LR-before-QDQ hook顺序检查；然后对300个quantizer调用 [validate_quantizer_contract](/home/wjq/workspace/svdquant-exp/scripts/research/wan_nvfp4_fastpack.py:165)，赋 `pack_activation_fast`。保留 `collect_fastpack_checks()` 检查实际输入domain。只 `.to(device)`，不得在转换后再整体cast dtype而把FP32 global/E4M3 buffer改成BF16。原版81帧产生32760 video tokens，K维1536/8960均满足现有native packing要求；没有固定31200 token的软件限制。
3. **0.40 的可解兼容点已实测到 CPU meta 层级**：直接 `DiffusionModelStruct.construct` 因新具体 `WanAttention` 类型未注册而 KeyError；按现 loader 的同样 `register_factory(type(attn1), _default_construct)` 注册后，30block struct成功。不要启用0.40 QKV fusion（它创建 `to_qkv/to_kv`，会绕开这300个已安装模块），也不要为了旧runner强行更换0.40 attention processor。E007/E008 runner硬锁Diffusers0.33.1，不能直接执行；可复用其库函数。
4. **当前 shell 的一次 CPU import 实际停在环境接线**：DeepCompressor config 导入会加载 `deepcompressor_C`，默认PATH找不到ninja。没有GPU/模型前向，未继续构建。现成 ninja 在 `/home/wjq/.conda/envs/convrot-wan/bin/ninja` 和历史PTQ env，已有 `.so` 在 `/data1/models/svdquant-wjq/research/cache/torch_extensions/deepcompressor_C/deepcompressor_C.so`；E043 launcher已包含前者PATH和后者cache。因此无需先安装，但新worker预检需在其**实际launcher环境**确认完整 config/loader import；本次没有把“文件存在”冒充完整loader已成功。

CFG仍由原pipeline按cond/uncond分别前向，故每次动态A global保持自己的统计域；不要合并两支batch。量化仅接收BF16投影输入/输出，不触及FP32 scheduler state。50步每video预计100DiT、30,000 native residual GEMM、50 scheduler和一次公开VAE decode；计数用实际结果核对，LR另为BF16。

作为同时成对的 **plain** 强基线，已有 [wan_mainweight_qad.py](/home/wjq/workspace/svdquant-exp/scripts/research/wan_mainweight_qad.py:151) 可直接复用：对新加载的原版 transformer 执行 `install_qad(model)` → `export_packed(model)` → `install_packed(model, artifact)`，不训练/不创建 optimizer；这只是从原BF16 W生成FP32 master后按既有legacy Wan配方打包，部署为300个 `PlainPackedWanLinear`，无smooth/LR。释放第一步返回的模块字典等master引用；最终安装已包括同一fast activation packer。**不要复用E020/E022的plain/QAD artifact**，它们仍由rCM权重导出。此构建路径的CPU签名/实现已核，尚未对当前0.40原版实际GPU执行。

**剩余真正不确定性只有两类**：旧PTQ校准为33帧，现81帧且shift8；原校准flow shift未从保存config可靠恢复，这是校准分布外推而非loader错误。以及0.40的新图尚未完成该checkpoint的旧QDQ/native完整数值执行，saved hooks、300roundtrip和有限首次完整前向应先落地；不要求native与BF16/QDQ逐位相同，也不引任意NMSE科学门槛。确认可运行后沿E043全部8例成对生成即可。这是旧SVDQuant/native部署baseline，不是新方法。
