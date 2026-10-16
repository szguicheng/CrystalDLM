# 确定目标条件和交错训练样本形式

Labels: wayfinder:prototype
Type: prototype
Mode: HITL
Status: open
Assignee: none
Parent: ../map.md
Blocked by: 03, 11

## Question

一条首轮样本应怎样表达目标条件、简短可核对文字和 PLaID 晶体段？用少量可检查的样本示意确认：结构化目标与语言要求如何对应，哪些字段是输入条件、哪些是预测目标，文字位于晶体前后哪个位置，怎样体现交错与真实结构依赖？

原型仅为静态样本/说明，不构建训练数据集或运行模型。语言只表达已有数据能支持的事实，不补写未经验证的长因果推理；让用户对具体样本作出确认。

## Comments

### 用户指出输入条件偏离 — 2026-10-16

用户重申：架构图中的 user prompt 应包含结构性提示、结构特征或有标签支持的性能要求，ehull 只是选定的附加性质条件，并非唯一输入。2026-10-11 的图示记录已经明确这一点。当前 `prepare.py` 却仅将源材料的精确 ehull 放入 prefix，结构语言描述全部作为输出监督；这是实现偏离原方案，不能解释为用户新增加了需求。

两轮历史结果保留并归类为 ehull-only 简化试验。主线下一步应先补全用户请求字段及其与独立性质条件、输出描述、晶体目标的绑定，并同步定义条件遵循评价；不能直接在旧配方上切换到 diffusion。该讨论未授权新一轮训练。

当前小样本生成评估为：已固定的 900 条验证记录按排序后的 material_id 用 `random.Random(42).sample(..., 32)` 无放回抽取32条。没有按 ehull 或结构分层，也没有按性质值去重；32条含7个零目标和25个互不相同的正目标，实际只有26种不同的ehull-only prompt。该抽样只是一轮小规模生成检查，不等于完整的结构/性质条件评估。

### User sketch and data proposal — 2026-10-11

用户提供新版架构图：用户结构要求经 AR 形成候选设计文字，原请求与显式 ehull 目标共同进入同模型的 diffusion 模式；晶体 token 先还原 Wyckoff 文本，再由 PLaID parser 展开结构。用户提出将材料结构、已有 ehull 与结构/性质语言描述绑定，并希望用 LLM 从 CIF 或 PLaID 文本形成描述。

主智能体的待讨论建议：将第三项称为“结构语言描述”，区别于已有可解码的 PLaID 晶体表示。先从统一结构版本提取可核对事实，再用模板或现成 LLM 组织文字；已知性质来自标签，不由 CIF 或 LLM 直接猜测。标注 LLM 属于离线数据构建，不自动成为推理时新增模块。Robocrystallographer 是可调查的既有事实/描述工具，尚未运行或采用。

三元组是基础资产，训练时还需区分用户请求、AR 监督文字和 diffusion 结构目标。描述需说明对应原始结构还是 PLaID 解码后的规范结构，避免原始/精修晶胞中的原子数、空间群或晶格数值混用。尚未确定字段集合、描述模板、LLM 选择、最终段落顺序或批量标注方案；本票仍为 open，不视为原型已获批准。

### User confirmation — 2026-10-12

用户确认不专门训练 CIF→描述模型，使用现成 LLM 做离线标注。当前要求先核查 CIF 可提取的性质/结构信息，不能直接提取的性质用额外标签，再据字段设计模板或提示词。字段依据由 [核查 CIF 可提取字段与离线描述提示词](11-description-fields.md) 保存；字段与具体提示词仍待讨论。

### 静态字段与提示词草案 — 2026-10-12

主智能体在本地现有编解码环境处理前三条材料记录，提取原始及 PLaID 解码结构的组成、空间群、晶系、晶格、体积、理论密度和独立轨道。组成保持及轨道多重性总和检查通过。样例见 [实际字段样例](../../../research/wayfinder_crystal_20261011/description-field-examples.json)，单一提示词与人工文字示例见 [描述模板草案](../../../research/wayfinder_crystal_20261011/description-template-draft.md)。未调用 LLM，未批量标注；局部配位/连接方法尚未选定，模板尚未获批准。

### User approval — 2026-10-12

用户同意采用字段设计与提示词，要求提示词和标注输出改成英文。主智能体已更新 [英文提示词与示例](../../../research/wayfinder_crystal_20261011/description-template-draft.md)。这个确认不等于局部近邻方法、标注 LLM、训练样本段落顺序、评价容差或完整样本契约已经决定；本票仍开放。

### User-selected annotation model — 2026-10-12

用户指定 SciencePal 网关的 deepseek-v4-pro，并要求最大思考强度。配置按官方参数设 thinking enabled、reasoning_effort max，密钥通过现有 .env 引用，未写入提示词、研究文档或配置产物。用户指令授权本轮小范围接口验证，地图对其他训练/批量执行的限制保持。

[非敏感调用配置](../../../research/wayfinder_crystal_20261011/annotation-model-config.json)、[样例核查回执](../../../research/wayfinder_crystal_20261011/deepseek-smoke-review.json)。同一材料两次 Chat Completions 返回英文描述并通过基础事实检查；都没有思考内容，reasoning_tokens=0，最大思考未验证。Responses API 请求同一模型返回 HTTP400，上游指出 deepseek-flash 服务ID不存在；/model/info 的元数据读取被虚拟密钥权限拒绝。保存各次回执，不以 HTTP200 宣称最大思考已生效。暂不扩批量标注，待网关思考透传/接口适配问题明确。

### User acceptance and preannotation audit — 2026-10-12

用户接受客户端按标准发送 max；上游没有返回思考字段或统计不再作为推进阻碍。此确认取代上一条评论中等待网关思考透传的要求。继续采用已成功的 Chat Completions 路径；原始回执保留。用户要求先报告待标注数量并复核整体设置，再考虑全量。

逐条数据与四文件哈希核查通过：总数 45,629，train/val/test 为 36,491/4,517/4,621，材料 ID 和现有结构键均唯一。已有三条基础字段样例、一个材料的两次成功 API 描述；正式批量未启动。详细设置和缺口见 [批量前核查](../../../research/wayfinder_crystal_20261011/preannotation-settings-audit.md)。下一步需要全量 PLaID/字段提取、局部近邻方法与小批次质量验证；不因客户端 max 已确认而跳过这些准备。

### User-authorized preannotation execution — 2026-10-12

用户明确同意进行全量前工作：PLaID 编解码和结构字段提取、固定局部结构分析方法、30 个训练材料英文试标、质量及用量核查。此授权允许准备阶段的代码实现、CPU 结构处理和 30 个材料 API 标注；不扩展为全量 LLM 标注或生成模型训练。执行资产位于 research/crystal_preannotation_20261012/。

### Preparation execution evidence — 2026-10-12

已实现 PLaID 编解码/结构字段提取与独立标签绑定，几何模式 CrystalNN 配位、多面体 motif 和 Larsen 连接维度采用固定参数；角/边/面共享未提供。30 个训练材料覆盖全部七个晶系和28个空间群。实际试标发现逐位点冗余输入导致数量误读、cp 被扩写为特定堆积类型、ehull 被舍入，现改为局部环境汇总、明确 motif 名称、LLM 结构文字加程序原值标签句；最终30段经事实/格式核查通过。执行资产位于 research/crystal_preannotation_20261012/，最终样本为 pilot30-rendered-final.jsonl。

Burgundy CPU 作业624103已完成45629个材料；发现1425条被空间群别名解析错误拒绝，已修复并通过回归检查，作业624223正进行全量v3验证。全量LLM标注和生成模型训练未启动；本票的训练段落契约仍开放。

### Preparation complete — 2026-10-12

全量v3已完成，作业624223 COMPLETED/0:0。45629条来源绑定全部核对通过；最终45465个可标注（train36362、val4501、test4602），164条隔离。最终队列使用 research/crystal_preannotation_20261012/prepared_002/，明细及方法见 [准备结果](../../../research/crystal_preannotation_20261012/README.md)。30个训练材料最终英文段落核查30/30通过，与全量同ID的PLaID文本及语言输入事实一致。两次CPU验证累计17:06:12 CPU时间、38:27墙钟、GPU0，两作业资源均释放。全量LLM标注和生成模型训练未执行。

### Shared-polyhedron fields and Flash trial — 2026-10-12

用户要求补齐角/边/面共享字段，并试用DeepSeek V4 Flash、max强度测速度。已在45465个有效材料完成增补：45128可判定、207不适用、130几何不明确、0计算错误；旧164条隔离记录保留。所有父字段/标签完整核对未变。共享类型以周期顶点身份和凸包真实边/面及分离条件核实。

同批30材料Flash新文本含共享信息；另30次Flash对照与保存Pro输入仅模型名不同。Pro平均2.128秒，Flash同输入2.131秒，Flash共享版2.296秒，没有明显提速。最终30个段落核查通过；初稿1个越界形状赋名已修正，原始回执留存。详细数据和样例见 [共享字段与Flash试标](../../../research/crystal_sharing_20261012/README.md)。这是试标结果，不代表生产标注模型已最终选择或共享生成模型已训练。

### Readiness review — 2026-10-13

用户询问是否可启动全量实验及启动前瑕疵。已复核当前资产：结构队列准备完成；试标客户端仍限定30条，自动生成质检不等同于最终人工修订后的30/30；缺失共享字段类别尚未进入语言试标。建议先补批量入口与质检、训练集300条自动批次，再扩批。训练样本/模型协议、GRACE评分校准与48GPUh实测仍未冻结。详细审查见 [全量实验启动前核查](../../../research/crystal_sharing_20261012/experiment-readiness-20261013.md)。这是建议，不记录为用户已经批准新的全量标注或训练。

### User-scoped subset execution and resume requirements — 2026-10-13

用户明确当前优先补齐数据，健壮性限定为批量标注出错后有限重试、保存成功结果及断点续跑；暂不扩大到评分或 A100 训练断点讨论。用户随后缩小执行范围：现有可标注 train/val 分别固定同一随机种子抽取 1/5，先转换并开展前期小实验，通过后再全量转换及后续训练。采用 seed=42、分别抽样、向下取整，train 7,272、val 900，共 8,172；test 保留。ID 清单和来源哈希永久保存，原始划分/事实/标签不变。标注指定 Flash/max，并发 8，无 1 秒成功等待。新执行资产位于 research/crystal_bulk_annotation_20261013/；目前构建子集和持久化批量客户端，尚未声称标注或小实验完成。

### Subset annotation started — 2026-10-13

seed42 子集已生成，重复抽样得到相同 ID 与压缩文件哈希。批量客户端通过7项伪网络测试和真实HTTP/SIGKILL恢复测试；既有6项提示词/渲染测试通过。真实8并发16条全部提交，再跑同样16条未增加调用次数。随后启动仅8172条子集的本地tmux任务 crystal-annotation-subset-20261013，Flash/max、并发8、成功间等待0、逐条SQLite持久化。成功/失败回执和恢复命令见 [子集执行记录](../../../research/crystal_bulk_annotation_20261013/README.md)。标注全部完成、科学核查与小实验训练尚未完成；全量未启动。

### Subset annotation complete and training input audit — 2026-10-13

1/5 子集已完成全部8172个描述，train7272/val900，最终失败0、未完成0；3次临时错误经重试恢复。导出ID完全匹配固定seed42清单，无遗漏/重复。正式训练样本段落/对照契约尚未关闭；用户已授权Burgundy小测试。当前按不可变facts_sha256核对并合并描述—PLaID—原始ehull，资产位于research/crystal_training_preflight_20261013/。
