# Current State

**最新执行状态：active。** E047匹配校准基线准备中，固定原版81帧/CFG6/UniPC50/shift8、rank32/g10/64records。独立选择审查确认新64条与旧1600文件按原Random0抽样相同；14提示、35时间位置、35cond/29uncond，不含最后两步，明确保留fast政策。采集worker与完整PTQ入口实现中，尚未启动GPU；PTQ6小时上限，旧成功参考约2小时。E046已有末步部分改善且新增2.65GiB，未恢复原版质量；仍无合格paper claim。

2026-10-03。E025不支持TAEHV是当前语义/低动态问题的主因：RAFT16条判定不变，MJ.322401→.332559、文本3升5降，主要语义偏差保留；结束VAE排错。官方采样发布路由同三步UniPC，合同核查收尾。E026保持原μ不额外量化而复用K4，attention NMSE .00241969→.00331458（+36.98%），仍比global .00356130低6.93%；52/56head比block差、53/56比global好。这是单层固定μ数值取舍，不是全部融合方案的数学下界或视频质量。**目标active，无阻塞；仍无已成立的新方法、等质量加速或可投稿核心贡献。**

## 最新证据

- E046 complete：summary SHA e1129e856ada4029df64e9b64fc0e8145e97869cb989dfee2167f30a3cf31a72，CUDAfalse；16媒体/终态、400steps/816DiT收据、同state/CFG已核，八nativefinal对E044零漂移。生成1126.01秒/评价72.58秒。root全部八对九帧观察完成且先于评分；报告042/D072。

- E045 complete：summary_v2 SHA703370b55e02cd96d5e9ac2684f1fc2f15b61da5b19b7bbd096b00c9688b3002，CPU10.68秒/CUDAfalse。r0/r1 postclamp full后帧MSE .0051593/.0063932；first-only .0000373/.0000137，rest-only .0051282/.0063707。CFG后latent误差1.1957/1.4820已高于首片.4948/.5694。root八张九帧观察完成，非盲评/实时。恢复launcher191.497秒，PIDs3605018/3605024已退出；GPU0/1释放。

- E044 complete：16新增+8继承媒体全覆盖，800步标量链、16FP32终态、1600逐DiT记录与实际共享输入已核；summary SHA a924257b1e38335e3e9b0b967bdabd6083c11ae573102d7d2dc5262ef51db1d5。SVD−plain MJ+.35378、7正1负；SVD−BF16−.06625、2正6负，fineness全8下降、DINO7下降。AMT .96686/.96500/.97034，不能据高平滑度认定质量恢复。报告040，D070。

- E043 complete：官方基础Transformer SHA已匹配revision0fad780a534b6463e45facd96134c9f345acfa5b；原版Diffusers50步/CFG6/shift8/81帧，八条全部生成和评分。独立核八实际噪声、16初终FP32 tensor、400 scalar step及8媒体SHA；中间tensor未存，不称重放。AMT.96686/DINO.92700、MJ逐例见报告039。所有生成/评估进程退出。

- E042 complete：六条PCM/六NPZ/四原图，CPU2.540秒，CUDA未初始化；root已看两full与两zoom_v2并审源。v2仅修排版、0信号重算。包络非音频事件标注，基线自身同步仍unknown；报告038，D067停止当前AV同步归因与新方法派生。

- E041 complete：两agent各一seed、每seed三臂全部124帧，非实时播放/盲评。BF16可辨投影开合r0/r1为22/20；global为至少5/unknown，coarse至少5/至少10，不能直接比较下界推频率。所有物理接触不确定。报告037，D066停止当前统一变慢表述。

- E040 complete：2native/2decode/2新down/6up，0通信/attention/DiT；launcher7.378秒、独立CPU6.672秒，原/side重放与global均零漂移。decoded-X输出valid NMSE1.83369e−6，相比side5.9324e−8仍属小绝对差；不作质量必要性主张。worker1369659已退出，GPU0释放。报告036，C006主线关闭。

- E039 complete：BF16/FP8/row/side pairmax-wall中位6.348/11.090/8.762/4.897ms；side相对原native valid NMSE5.9324e−8，row与side down实际一致。allocated峰值.87775/.87775/.82168/.64503GiB，非整模。launcher9.23秒、独立CPU4.25秒；报告035。

- E038 complete：四公开动作×两独立seed×三attention臂，同native SVD权重，24媒体/480DiT/960scheduler；生成与质量独立核验complete。coarse−global MJ均值−.012914；拍手双seed正，叠衣/汽车双seed负，大象异号。RAFT逐case同6/8、DINO .92275/.90650/.91539。root已看全部8张固定8帧三臂图，非完整视频盲评。首次0DiT检查失败保留、v2最小修复；正式launcher1146.45s、评价216.08s。报告034。

- E037 complete：launcher7.18秒、52native/0DiT，rank worker68067/68068退出。独立CPU8.17秒，CUDAfalse，两臂各自复现E034输出；global全部56heads误差较高。A2A远端324.488/287.971MiB（均含逆输出），allocated峰1.4342/1.3525GiB。报告033；E038准备中。

- E036 complete：launcher7.38秒、78native/0DiT，rankworker4053101/4053102退出，GPU0/1释放。两rank配对max wall三臂28.633/23.214/21.553ms，十轮范围互不重叠；三臂输出均等于E034 coarse16，BF16 NMSE .0028563913。CPU独立复核9.39秒complete；双向A2A含逆输出619.5/396.9766/324.4883MiB，FP4另Ksum逻辑.05469MiB。无整模/质量claim，报告032。

- E035 complete：capture39.72秒、1BF16 DiT/102SDPA/3native projection；CPUrouter18.14秒/0GPU、独立六小artifact复核1.42秒。真实H3 3D128分组、prefix11/video200、CDF>=.9/min4。mask变化行63.7%–80.5%，但平均每video query仅净增.305–.672块；未测consumer成本或质量。worker3585485/3710380均退出；GPU0–5最近空闲、6–7其他任务。报告031。

- E034 complete：44.38秒/156native/0DiT，worker3237350退出、GPU0释放。coarse三层NMSE .00285639/.00815243/.00757479，全部168head优于global，vscodebook32改善136更差；同容量table下完整时延低5.71%–5.92%。12末输出/实际C/id与120条实测独立复核complete；9个既有臂输出对E033差异为零，报告030。

- E033 complete：110.37秒/478native/0DiT，worker3062111退出，GPU0释放。5控制maxULP0，15末输出与全150计时/内存数组独立复算complete；codebook对E032输出一致，164/168head劣于freeblock。全路径比fullblock峰值低21.7%，仅快0.21–0.23ms，非稳定加速claim。报告029；E034 prepared/implementing，四臂156native，无新kernel。

- E032 complete：43.69秒/3native/0DiT、独立三输出及中心/id复算complete。K16全部168head优于global，三层NMSE .00280712/.00804822/.00749995。中心构造1.77–1.81ms、raw准备8.49ms；精度低于range1，未测完整consumer收益。E033 prepared：三层五臂、10正确性+468计时attention，900秒GPU0含冷JIT；不扩聚类网格。

- E031 complete：97.86秒/9native/0DiT、独立9输出/24×10timing/Ω及小basis复核complete。fullSVD/range0/range1三层保留率92.16–96.65%/76.71–91.69%/90.66–94.55%；range1全部168head优于global，range0一head差。raw准备产物与baseline不同，不推完整速度；报告028。

- E030 complete：78.28秒、1DiT+6probe，capture内50FP4attention/52BF16SDPA/200GEMM；独立六输出/basis复核complete。固定basis相对same-sampleEuclidean NMSE+10.63%/11.54%/11.82%，三层53/56、56/56、56/56heads退化。166/168head仍优于global，block0 head5/16更差；保留全部。factorableFP32比BF16center pooled小幅降低0.023%–0.038%，逐head正负都有，不是质量等价。donor重放一致，模型释放allocated.00891GiB。

- E029 complete：15native/0DiT、65.57秒，15输出独立CPU复算通过，6现场端点对历史一致；504候选head全部优于global，但相对block32改善/472退化。rank16保留91.66%–96.66%误差差优势，只有same-sample oracle表示信号，无实际低秩consumer。

- E028 complete：CPU13.104秒、0 GPU，三层×56head×三geometry共504项谱。r16 Euclidean/K-score/K4-error尾能量分别为block0 2.27%/0.26%/0.89%、block24 11.90%/2.88%/8.07%、block48 9.10%/0.74%/3.63%。同样本oracle、未含重新Q量化，不证明固定basis或真实consumer收益。
- E027 complete：完整QKV→output full/chunk4096为31.780/33.975ms，allocated峰值3.518/2.601GiB；四输出与原block一致。独立BF16 M16 stripe14.815ms不作融合成本下界；独立CPU汇总complete，报告026。

- E026原输出独立CPU重算：block/fixed μ×K4/global NMSE .002419689/.003314582/.003561302，oracle比block高36.98%、仍比global低6.93%；52/56head vsblock差、53/56 vs global好。2次native、0DiT、18.53秒含I/O/统计、4.05GiB；不是性能结果。独立K4与μ还原及六packet绑定通过，correction未存完整tensor，不称其已独立逐位重算。该成本实验已由E027完成，见最新条目。
- E025全部16同latent/16VAE decode完成320.16秒、13.22GiB；temporal130.16秒/MJ12.33秒含launcher，均退出。MJ .322401→.332559、3/8prompt提高；RAFT16条判定不变，DINO16条下降；root已看全部8r0五帧，主要语义失败保留。独立汇总complete。

- E024独立CPU汇总evaluation_summary.json complete（SHA58cba1d2…684c4f44e）：正式16视频/48DiT/14400NVFP4GEMM/1440FP4 self-attention/1440dense cross-attention，全部81帧480×83216fps。MJ total/alignment/fineness/coherence=.322401/.307388/.097620/.570618；AMT.994420、动态2/16仅shuttle、DINO.979017。相对rCM plain 6/8prompt MJ提高，相对BF16仅4/8，不能当量化因果或普遍质量收益。
- E024固定FastVideo8444c089/HF621c6aeb/TAEHV011dfc211；完整模型与原参考TE/VAE/tokenizer均公开SHA匹配。独立env torch2.12/cu130/kernel0.3.5/FlashInfer0.7.0.post1已就绪；runtime_env及所有JIT在DATA1。首次smoke含JIT286.58秒，正式suite112.69秒含build15.35；pipeline中位4.916秒、TAEHV.230秒均diagnostic，不是compile稳态性能。root已看全部8r0固定五帧，无旧大块条纹但多题语义偏离。

- E022最终CPU汇总finalsummary.json及video_summary.json均complete：四臂64视频/256DiT/57600FP4/15360SDPA；全部实际16noise、条件、77帧媒体及评价原始数组核实。selected生成190.97秒、MJ12.52秒、时序125.85秒，所有worker已退出。
- MJ总分BF16/plain/SVD/QAD=.180770/.071057/.101717/.071926；QAD−plain+.000868为相抵，2/8prompt、6/16seed提高，4/8prompt双seed异号，火星r0主导正向贡献。alignment/fineness/coherence均未改善。不用近零均值证明质量等价。
- RAFT动态通过9/6/10/11条（各16），不复现E021“QAD运动减少”；QAD flow均值39.30对plain12.70，增量91%来自科学馆/庭院/球场失真例，不叫正确运动。DINO .88815对plain .92650，11/16轨迹下降。root已看全8组replica0五帧四臂sheet，仅抽帧观察。

- E022独立快照snapshot_step0068：四dev prompt及四timestep汇总均0>32>64；32→64两路径各32/32条记录改善，不声称所有row全程单调。只有4文本/8轨迹/32相关state，非视频质量。固定128终点及native候选选择规则不变。
- E022三baseline48视频完成192DiT/38400FP4/11520SDPA，16实际初态各不相同；原始媒体完整核验，MJ与全帧AMT/RAFT/DINO全部完成。MJ均值BF16/plain/SVD .18077/.07106/.10172；失败样例全保留，未作质量排序。root已看全部8张replica0五帧sheet。
- E022四次VAE诊断52.82秒/13.20GiB/0DiT：球场坏例与阅读对照的BF16 MP4精确复现；FP32球场仍同样竖条，像素MAE .00484/.00210。只排除“简单换VAE精度即可修复”的解释；不批量重解码、不宣称根因已定位。GPU诊断/评价进程均退出。
- E020：rCM-Wan完整31,200 tokens，300主矩阵共1,391,984,640个FP32 master，64实际Adam更新。非目标bias/权重冻结，导出无在线LR、无训练master。全部24目标由当前BF16现场重算并复现旧缓存。
- 训练QDQ pooled NMSE .141154→.092964（−34.14%）；开发两个prompt/8相关状态的native .137708→.148410（+7.77%）。QDQ第32步较好是探索性，不回改固定64终点。开发输出已独立CPU重算；训练scalar无保存输出，明确reported-only。
- 同卡独立部署：BF16/SVD/plain/QAD中位1.789/1.990/1.659/1.658秒，模型storage2.649/.859/.785/.785GiB；前向峰值allocated4.099/3.015/2.331/2.331GiB。低位reserved6.379–7.014GiB高于BF165.768，不将packed尺寸当总显存。
- E021：四prompt×四臂16视频、64DiT、14400真实FP4GEMM、3840BF16SDPA；全77帧/480×832/16fps。shared输入/final/media均核验；四prompt共一个实际噪声序列，不能当四噪声重复。
- QAD相对plain MJ总分两升两降，均值+.2334由campus异常贡献+.2542主导。campus的BF16本身严重绿条纹失真，未事后排除。8抽帧奖励与全帧AMT/RAFT/DINO分别报告，无人工完整运动观看结论。
- QAD AMT较高而RAFT运动阈值仅1/4（BF164/4，plain2/4，SVD3/4）；二值贴阈值、campus失真flow大，不把高运动或平滑度当质量。蓝球/背景变化只是抽帧观察，当前数据不能证明记忆某训练样例。

## 决策与下一步

最新D072：保留BF16末步为强控制、部分改善但未恢复；旧校准真实shape/schedule mismatch已核，优先匹配原版81帧shift8校准，暂停patch机制实验。既有D071：E045单末步足以产生局部碎片，但不证明高精度末步能修复native轨迹；先做该强控制，不扩decoder/CFG新loss。既有D070：E044已建立原版native真实质量缺口，保留SVD作为强于plain的基线；不把plain/SVD整配方差分归因LR，也不把较高AMT当质量。下一先设计少量同teacher状态实际前向/采样介入，检查时间位置差异和固定decoder响应；自由分叉终态δ及九帧周期推断无效。近邻已覆盖一般decoder抗扰动/causal不均衡，不直接加loss或训练；当前无新GPU实验。P006 parked/C006关闭保持。

配置排查收尾：不改为无权重绑定的DMD，不扩VAE dtype/分块网格；E023普通W-global训练仍只保留计划。普通QAD的MSE/质量错位已有强近邻，不能据此发明loss或重启运动收缩故事。

E026固定原μ复用K4损失本例block相对global大部分精度优势；E027已知chunking提供实际显存/时间基线，停止朴素在线M16调度。E028转向受限中心，rank16谱集中但K-error几何比K-score更宽。E029同步重中心化已完成且有压缩信号，三geometry无稳定胜者。E030已固定p30 Euclidean basis迁移p36，准确性明显回落但多数仍好于global；可分解FP32合同未造成同等量级损失。RoPE/模态核查已完成，不能从相位抵消推出有效rank16 selector；现有源码含32未旋转维度以及模态反例。E031/E032已完成自适应构造成本；当前以E033最小共享行consumer验证实际取舍，不先扩复杂多行consumer。此前近邻Sage/MpFA及HeadQ公开构造已记录，低秩结合律本身不是贡献。

## 保留的历史边界

- H3原生FP4投影与FP4 attention均已真实部署。E009 BF16/native DiT8.260/6.787秒；E014 BF16/SVD/plain8.291/6.804/5.823秒。local/终点NMSE存在取舍，无等质量普遍收益。
- E016现成BF16/block/global attention6.791/5.124/4.968秒；E017真实视频p36 block更低tensor误差却四个奖励回答更差，global与SVD回答相同。既有实现组合不是新方法，停止扩基础网格。
- E018有可干预的Q分组相位误差结构，伪边界也交替、向量不整体翻号，不立闪烁claim。E019返回LSE使39/1.616亿元素极微变化；原分片实验没执行，不能当机制阴性。相关方向因强近邻与机会成本park。
- E012/13过严teacher行为gate仅说明其具体诊断没就绪，不泛化否定teacher/研究；E011因无已验证SM120 consumer未执行，不当科学阴性。
- C001/C002当前路线rejected；C003parked；C004/C005stopped。全部旧实验及文献碰撞见实验/决策索引；不再以任意10%/byte equality作为全部科研准入。

## 工程与资源

已完成用户指定14个community skills安装、git更新、8×RTX PRO5000 72GB审计、旧H3 LR hook修复；修复没有被测质量收益。E005–E021已执行源/结果保留，后续变化新文件或显式版本；不推送远端，不覆盖用户原dirty/untracked文件。

E020–E034全部GPU/实验CPU/下载/安装进程已退出，包括E024 smoke3578802/3580314、suite3665782、temporal3719754、MJ3719755；新任务前重新查GPU。最近资源检查：GPU0/1/5空闲，GPU2–4/6–7被其他任务占用；E031/E032 worker2483370/2715035均已退出，实际启动前再次检查。所有大模型/env/cache在DATA1；FlashInfer不遵守XDG，首次19.5MB JIT曾落root缓存，之后完整复制DATA1并显式FLASHINFER_WORKSPACE_BASE，原共享cache未删。MJ Python独立prefix保留旧torch2.11/cu128；新FastWan env torch2.12/cu130。公开下载前两轮0B失败因tmux未继承代理，第三轮仅修环境后完成；小kernel smoke报告profiler字段错误保留，已存实际前向成功证据，未重跑GPU掩盖。正式模型生成/评价均无失败。

最近报告：[038 音画事件读出](../reports/038_20261003_audio_event_readout.md)、[037 拍手可辨性](../reports/037_20261003_clapping_readability.md)、[036 低秩信息与范围收尾](../reports/036_20261003_projection_information_control.md)、[035 输出通信与投影](../reports/035_20261003_output_projection_boundary.md)、[034 完整视频中心对照](../reports/034_20261003_center_video_results.md)、[033 global强对照](../reports/033_20261003_global_center_control.md)、[032 两卡校正通信](../reports/032_20261003_sequence_parallel_correction.md)、[031 投影与稀疏预算](../reports/031_20261003_projection_sparse_routing.md)、[030 粗中心强对照](../reports/030_20261003_coarse_center_control.md)、[029 共享行消费者](../reports/029_20261003_shared_row_consumer.md)、[028 自适应成本](../reports/028_20261003_adaptive_center_cost.md)、[027 受限中心](../reports/027_20261003_restricted_query_centers.md)、[026 校正成本](../reports/026_20261003_correction_cost.md)、[025 校正项干预](../reports/025_20261003_qmean_k4_intervention.md)、[024 完整解码器控制](../reports/024_20261003_decoder_control.md)、[023 官方FastWan产品基线](../reports/023_20261003_official_fastwan_baseline.md)、[022 扩大QAD的完整结果](../reports/022_20261003_expanded_qad_final.md)、[020 小数据QAD完整视频](../reports/020_20261003_qad_video_results.md)。完整记录：[README](../README.md)、[experiment_log](../06_experiments/experiment_log.md)、[decision_log](decision_log.md)。

独立视频测试已在训练结果出现前固定：[E022视频补充协议](../06_experiments/E022_video_evaluation_plan.md)。8prompt×2不同seed，排除新train/dev及本项目已查历史生成文本，四臂64视频；本轮随机恰为4imaging/aesthetic与4scene/background题，不能代表全部VBench或专门运动基准。不改样例/不加固定128次要臂，后续绑定预定native开发选中的checkpoint。

E022训练/生成/评价已complete，finalsummary.json与video_summary.json独立核验（后者SHA78bd795c…0e984f31）；原worker1890988/2981814/3059070/3059071均退出。既有源、媒体、checkpoint及失败记录保留。新增纯CPU配对评分图，原汇总文件未改。

ModelOpt成熟配方已固定commit e68eb44并保留小源码快照；默认固定校准outer global，当前E022为W/A每次动态global，二者都动态group16 SF。E023仅W固定初始global单因素计划prepared_not_started，无GPU/新训练器；这不是新方法，也不是完整ModelOpt复现。先完成E022，不能用配方缺项替代质量验证。

共同生成异常只读核查已完成：本地官方rCM同77帧/sigma80/四步schedule/更新公式；UMT5同attention_mask后零pad512，原DiT也无零padding屏蔽。关键转换映射未见明确偏差，但旧等价报告仅随机全长非零text、小latent、t500单forward，不能证明真实条件完整rollout等价。根因仍未定位；如继续，直接做实际条件的官方原版端到端对照，避免新的小shape检查循环。

原架构真实条件对照已完成，见rcm_rollout_reference.json与独立validation；8DiT/480SDPA/4decode、worker95.99s/13.21GiB，两例final相对converted NMSE .112151/.209927。root已看两张固定5帧，球场共同条纹仍在；这是BF16主干+FP32源码要求阶段+FLASH SDPA的受控原架构，不是原发布二进制或原UMT5完整复现。停止当前排错网格，不把共同异常当量化机制。
