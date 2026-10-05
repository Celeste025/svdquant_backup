# E035：先隔离真实投影误差是否改变累计概率路由预算

2026-10-03，GPU前固定。上一goal turn为progress：E033/E034完成并独立复核；聚类路线停止扩网格。N2本地入口核查新发现cuDNN已有SM120可变块数/尾长/块内有效前缀consumer，不需要新kernel。现成H3 VSA固定top-k，不能直接检验top-p预算变化；已有量化×稀疏/teacher attention/动态预算校准强近邻，当前无新claim。

本轮先只捕获同输入投影配对及做路由诊断，不预热或运行稀疏consumer。数据固定E014 manifest的e010_p036_s14（原E010 BF16 teacher轨迹、已有文本与噪声），blocks0/24/48。一完整BF16 DiT，捕获各attn真正raw x、原rope/cu kwargs及该次helper处post-norm/RoPE Q/K/V；预计102 SDPA。随后三层各只将qkv_proj临时替换成已验证E009 legacy-export SVD NativeH3Linear，原fast activation packer，接同raw x沿原attn.forward运行；helper捕获QKV后局部sentinel立即返回，不做额外attention/outproj/全DiT。1完整BF16 DiT＋3native projection；当前只此配方，不扩plain或新训练。

p36原projection输入[22592,5376]；先完整处理22592再对已捕获QKV取cu定义的有效N22539，绝不能先裁projection输入改变tensor-global scale。原Comfy实现view[T,3,56,128]和norm/RoPE直接复用，不复制猜布局。真实有效布局text0:813、audio813:1227、video1227:22539；53模型padding只在projection期间保留，路由不当有效key。原调用、实际x及BF16/native QKV存DATA1；模型历史输出只报告数值漂移，不用任意byte科学gate。旧模型/执行源不改。

GPU前v2几何修正（没有读取本轮结果）：cuDNN源确认支持每KV物理块有效前缀，不再需要简化成raster contiguous。复用本地FastVideo H3 VSA的实际三维128-token分组，token segments=(813,414,(37,18,32))，tile_shape=(4,4,8)。网格来自该teacher video latent[1,24,37,36,64]和patch(1,2,2)。原官方纯CPU几何函数可按AST精确抽取执行，保留来源，避开GPU模块的import-time autotune；不重写散射/尾块算法。text与audio分别纯块：7＋4=11 prefix tiles（尾45/30）；video为10×5×4=200 tiles，含真实时空尾；总211物理块，padded长度27008，valid仍22539，所有pad仅位于各块有效前缀后。保存partition/untile/variable_sizes，确保每validtoken唯一映射，sum sizes=N。

新适配只替换预算规则：按原H3 pooling作每块FP32有效token mean，FP32 meanQ×meanKᵀ/sqrt128；200个video query块仅对200个video key块做FP32softmax/CDF，按概率排序取超过p=.9的最小前缀、至少4块（BSA现有CDF规则），再union全部11个prefix keys；prefix query全部211keys。按块概率的CDF是proxy，不声称等于真实token注意力质量；variable-size块的有效token成本另外报告。分组来自现成H3 VSA，但CDF替换仍是显式诊断adapter，不是原fixed-topk产品，也不是BSA完整实现。仅这一种几何，不加contiguous臂；原v1计划保留为未执行。

CPU-only router脚本分别读取三层BF16和native同输入Q/K。保存6个score/mask/counts小tensor和源绑定，报告每层/head/row物理块预算、真实有效key预算、前缀常量成本、mask相交/变更、归一化CDF阈值裕量、所有partial块被选情况；QKV差异仅局部数值指标，不解释为视频质量。用相同BF16 mask按其自身native输入评分只作诊断，teacher mask不是部署oracle。不测稀疏耗时，也不将mask/Jaccard或边数当成真实时延。

CPU检查只核真实调用/导出可用、geometry不丢token/尾长、selector覆盖threshold及最小4的基本合同，不引入精确float并列门槛。完整capture与router分别有明确日志、源、失败保存；GPU0命名tmux、900秒含load/3局部投影；CPUrouter180秒。capture结束立即释放模型/GPU。若预算差异微小或只来自常规数量/阈值作用，停止这一具体系统故事；若有值得解释的实际图变化，再固定同native QKV只换两个mask及一次同预算控制，使用现成cuDNN实测真实成本。不设置任意10%研究准入，不把尚未测的成本当效应。
