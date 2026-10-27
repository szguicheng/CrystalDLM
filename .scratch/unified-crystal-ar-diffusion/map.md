# 固定 PLaID 表示的统一晶体生成模型：首轮实验决策地图

Labels: wayfinder:map
Status: open
Tracker: local-markdown
Created: 2026-10-11

## Destination

形成可交给实现阶段的首轮实验协议：确定单一性质目标、结构—性质—文字数据、共享 AR/diffusion 模型与匹配纯 AR 对照、评价方法、成功和停止标准，并给出单卡预算内的运行安排。
地图完成表示主要决策已经明确，不表示模型已实现、已训练或达到目标。

## Notes

- 2026-10-26 完整Design数据已就绪且本地/远程哈希一致。AR-design32条诊断在120epoch后全部复现并严格有效+SG正确32/32；正式训练已从原始Qwen3-8B开始，三卡/global12、固定3epoch，后接10,770次/需评价checkpoint的生成及GRACE共享评价。见[Phase3运行](../../experiments/crystal_tasks_v1/04_design_phase3_run.md)。当前尚无正式Design成绩，Hybrid未启动。

- 2026-10-26 AR-understanding 正式基线完成：固定3epoch/9093updates，validation4501、test4602，流程exit0/GPU释放。按validation NLL选出的step7200与final均完成独立评价；final test四字段全对4600/4602（99.9565%），best4597/4602（99.8914%）。18,206条评价回执重算一致；详见[正式结果](../../experiments/crystal_tasks_v1/phase3_understanding_20261026/RESULTS.md)。这是四字段读取基线，AR-design和Hybrid对照尚未完成。

- 2026-10-25 PDT / 10-26 UTC，负责人通过交接闸门 2 并恢复 Design 补标。AR-understanding 32 条过拟合检查通过（全四字段及完整复现32/32，NLL4.78e-6），已从原始Qwen3-8B重新初始化开始三卡/global12全量3epoch；后台后接validation/test。Design按原模型/提示词/8并发在本机新活动目录续跑，冻结数据未变。运行记录见[Phase3](../../experiments/crystal_tasks_v1/03_phase3_run.md)；正式结果尚未产生。

- 2026-10-25 用户授权准备两个独立任务的全量 JSONL：Design 保留空间群＋Ehull→英文描述＋PLaID；Understanding 改为仅 PLaID→四字段回答。原划分36,362/4,501/4,602；Understanding已在服务器完成全量校验，Design另37,293条描述正在补齐。只准备数据，不训练。见[数据准备票](issues/24-full-design-understanding-data.md)。

- 2026-10-23 完整700组循环评价复核完成：直接重生成/反馈策略结构成功337/224，80meV最终达标55/28，唯一新颖达标49/26；700组和两臂评分均完成，exit0、GPU释放。相同修正模型下反馈策略较差；见[正式结果](../../experiments/repair_feedback_full_20261022/RESULTS.md)。

- 2026-10-22 用户授权修正模型的700组循环生成及完整ehull评价：七空间群各100组，同新模型比较直接重生成与结构反馈，最多3次、首轮共享；GRACE后置评分，不进入反馈。见[执行票](issues/23-repair-full-feedback-evaluation.md)与[协议](../../experiments/repair_feedback_full_20261022/PROTOCOL.md)。9项测试和7组完整smoke通过，正式700组已启动三卡生成，后接两卡评分；尚无正式成绩。

- 2026-10-21 修正训练完成：1212更新覆盖7272材料，900验证完成，32条合成错误修正配对生成旧模型8/32、新模型27/32有效且满足空间群；自然START均32/32，照抄错误输入12→0。exit0、GPU释放；见[结果](../../experiments/crystal_repair_training_20261021/evaluation_001/RESULTS.md)。尚未验证真实生成错误的多轮循环收益或原生成能力保留。

- 2026-10-21 用户授权新模型训练，空闲时用三卡。新增“错误晶体＋程序反馈→修正说明＋正确晶体”任务，保留生成/读取；固定7272/900划分，三张H20（1/2/3）、1212updates。数据v2及三卡断点恢复/验证试跑已通过，正式任务已提交，后接900验证和32条新旧模型自由生成对比；见[修正训练票](issues/22-crystal-repair-training.md)。GRACE不参与，本轮结果尚未产生。

- 2026-10-20 用户授权诊断和修复反馈切换。14例回放中纯AR/FLARE均0/14切换；新请求中以失败结构作为输入数据恢复14/14自然START。只修复上下文组织，未训练或强制切换。完整70组复验已结束：122/122次自然切换，但新增有效晶体0，反馈仍9/70（直接重试19/70），67次完全重复上一轮晶体；见[修复票](issues/21-feedback-switch-fix.md)。

- 2026-10-20 用户授权结构反馈循环试验：冻结现有FLARE权重，7空间群×10组配对种子，每组最多3次生成，对照独立重采样与结构反馈重试。GRACE不进入反馈，本轮不训练、不做性质评分。正式70组已于20:27完成：独立重采样19/70成功，反馈9/70；122次反馈重试中117次未进入晶体段，其余5次结构无效，GPU已释放。后续诊断未启动；见[新票](issues/20-structure-feedback-loop.md)。

- 2026-10-20完整FLARE式对照复核完成：AR169/700有效、25/700最终80meV新颖达标；FLARE121/700、24/700。900验证完成，exit0/GPU释放。生成后读取由共享评分器`experiments/evaluation_common/readback.py`统一核算；历史回执和正式汇总已重算，化学式正确为169/169与121/121、四字段全对为162/169与115/121。结构有效和性质结果不变。见[结果](../../experiments/flare_crystal_full_20261019/RESULTS.md)。未显示优于匹配AR；后续授权范围见结构反馈循环试验票。

- 2026-10-19 用户授权完整固定7272/900子集训练验证：匹配AR/FLARE各909updates，900验证＋各700端到端设计请求＋GRACE评分，详见[新票](issues/19-flare-full-training.md)。全量原始四万条不在本轮范围。

- 2026-10-19 FLARE式小样本迁移完成：8train/8val、64updates；diffusion有效train8/8、val4/8，源模型对应均0/8；三组保留样本AR均5/8。仅固定参考lead机制检查，80输出、exit0、GPU释放，未跑全子集或性质评分。见[结果](../../experiments/flare_crystal_qwen3_8b_20261019/RESULTS.md)。
- 2026-10-19 用户授权采用FLARE配方基于Qwen预训练能力。准备以现有晶体纯AR检查点为起点，干净AR＋晶体互补去噪，先固定8train/8val、64步和同预算AR控制；见[新票](issues/18-flare-crystal-pilot.md)。原论文与当前官方代码存在shift/mask细节差异，采用显式无泄漏目标契约并记录迁移差异。
- 2026-10-18 小规模控制验证完成：五组各28次均严格有效0，允许重新预测和高遮盖训练只在边界通过数上出现改善；两分支各128步完成，GPU释放。见 [验证结果](../../experiments/structure_conditioned_hybrid8b_20261018/validation_001/RESULTS.md)。未扩展到700或性质评分。
- 2026-10-18 用户已授权根据根因报告开展小规模验证：固定28次、源检查点三种采样、两个各128updates的匹配继续训练分支；不自动跑700或GRACE。见 [验证票](issues/17-hybrid-generation-validation.md) 和 [冻结协议](../../experiments/structure_conditioned_hybrid8b_20261018/validation_001/PROTOCOL.md)。
- 2026-10-18 17:05复核：共享AR＋diffusion首轮已结束，700次严格有效0（边界失败292、格式/一致性失败408），无候选进入GRACE计算；运行exit 0且GPU释放。详见 [混合结果](../../experiments/structure_conditioned_hybrid8b_20261018/RESULTS.md)。目前需诊断结构序列生成失败，未启动新训练。
- 2026-10-18 最新执行授权：用户要求开展共享AR＋diffusion训练，保持新的结构条件＋ehull输入。见 [混合8B实验](issues/16-structure-conditioned-hybrid8b.md)。沿用同基座初始化、固定数据和微调设置，晶体改为掩码扩散，训练后用同700次与GRACE口径对照；当前为代码/预检准备阶段。
- 2026-10-18 复核：结构条件Qwen3-8B纯AR重训及700次评分已完成，exit 0，GPU释放。最终新颖达标12/700（旧模型1/700），但严格有效95/700（旧140/700）；见 [结果](../../experiments/structure_conditioned_ar8b_20261017/RESULTS.md)。下一步需分析结构无效原因，尚未授权后续训练或diffusion。
- 2026-10-17 最新执行授权：用户要求使用“结构要求＋ehull”新输入重新训练Qwen3-8B并评价，仍先纯AR。见 [结构条件8B训练](issues/15-structure-conditioned-ar8b.md)。先以空间群＋真实上限条件修正训练输入，保留冻结子集和旧结果；训练后按相同700请求及GRACE口径评价，不启动diffusion或外部对照。
- 2026-10-17 评价方式已重新设计，当前依据为 [生成评价v1](../../research/wayfinder_crystal_20261011/generation-evaluation-design-v1-20261017.md)：独立固定请求、每请求100次、80meV主门槛、统一结构与性质评分；先7空间群共同基准，再组合条件扩展。来源结构不必从双方未见材料中抽取，新颖性针对实际生成结果。设计阶段不启动模型。
- 2026-10-17 当前研究优先级：确定结构请求来源及与 PLaID++ v4 的共同评价协议。用户确认逐项评价和固定低 ehull 方向，提出80 meV/atom、每请求100次作为具体候选；先核对阈值、分母、训练重合及公开权重，不启动新训练。见 [验收契约](issues/06-acceptance-contract.md) 和 [对照建议](../../research/wayfinder_crystal_20261011/prompt-source-and-plaid-comparison-20261017.md)。
- 2026-10-15 当前执行优先级：用户已授权在 LiuGangGPU 的现有 `/home/liugang/src/crystalDLM` 上启动 Qwen3-8B、BF16＋LoRA 纯 AR 基线，先预检后训练；沿用固定 7,272/900 数据和 32 个验证 ID。默认两张 H20、全部项目任务同时最多三张；旧单 A100 的规划由此次服务器和时间安排更新。见 [8B 纯 AR 实验](issues/14-qwen3-8b-pure-ar.md)。本次不启动新的联合训练。
- 当前执行优先级见 [先运行匹配纯 AR 基线](issues/13-pure-ar-baseline.md)：用户纯AR作业627890已完成首轮学习试验，严格有效率7/32；性质评分及结构依赖未验证，先分析该基线再决定联合训练。
- 2026-10-13 用户纠正实验顺序：性质/数据准备 → 同数据与基座的匹配纯 AR 基线 → 分析基线 → 共享 AR＋diffusion 联合训练。当前新 ehull 子集尚未运行匹配纯 AR 基线。既有20步 hybrid 作业仅保留为提前执行的工程诊断，不视为基线或正式联合实验，也不作为跳过纯 AR 的依据。当前优先准备纯 AR 配置，完整运行规模尚待确认。
- 2026-10-13 用户已要求在 Burgundy 按既定共享架构开展小测试。当前允许新数据接入、实际 tokenizer 长度核查与单 A100 技术预跑；记录见 [验证子集数据与共享模型训练链路](issues/12-subset-training-preflight.md)。完整对照训练与性质评分仍由原决策票管理。
- 2026-10-13 最新执行范围：用户要求先对可标注 train/val 分别用相同固定随机种子抽取 1/5，保存 ID 清单；先完成该子集英文标注并验证训练可用性、开展前期小实验，再决定全量转换与后续训练。test 暂不转换。并发 8，无成功请求间等待，必须支持逐条持久化、有限重试和断点续跑。此记录取代下文历史“仅 30 条试标”的执行边界，未启动全量 45,465 条标注。
- 本地图以规划为主，当前执行例外包括已完成的 CPU 字段提取、1/5 子集英文标注及获准的 Burgundy 技术预跑。技术预跑使用新实验目录，既有训练代码和快照保留。
- 用户已确认：首轮一个性质；文字为简短、可核对的设计说明；单张 A100，累计上限 48 GPU 小时。预检查、训练、对照和模型评估均计入预算；不能容纳时应调整方案并由用户确认。
- 已按用户要求补充角/边/面共享字段并完成 Flash max 试标，见 [结果](../../research/crystal_sharing_20261012/README.md)。
- 离线描述使用英文提示词和英文正文，字段与提示词采用用户已确认的设计；确认记录见 [确定目标条件和交错训练样本形式](issues/04-sample-contract.md)。
- 用户已指定标注模型与最大思考设置，并授权小范围接口验证；状态与非敏感回执见 [确定目标条件和交错训练样本形式](issues/04-sample-contract.md)。用户接受标准 max 请求设置，不以缺少思考遥测作为阻碍；1/5 子集 8,172 条英文标注已完成，最终失败 0；全量标注未启动。源数据池 45,629 个；已完成全量前准备，最终可标注 45,465 个，30个训练材料试标核查通过，结果见 [准备结果](../../research/crystal_preannotation_20261012/README.md)。原批量前核查见 [设置核查](../../research/wayfinder_crystal_20261011/preannotation-settings-audit.md)。
- 主线固定 PLaID 类表示，文字 AR 与晶体 diffusion 共享核心网络。新 tokenizer、relational_cnl_v1 的继续训练不属于本地图。
- replace/swap 是老师的举例，不是已确定的编辑链任务。首轮围绕目标条件生成，不把结构修改自动称为性质优化。
- 词汇使用 [项目词汇表](../../CONTEXT.md)。每轮先使用 wayfinder；人工决策使用 grilling 和 domain-modeling；研究票使用 research，论文分析可使用 academic-research-suite。
- 已有 MP—ehull 数据与当前优先级见 [确定首轮性质目标与数据口径](issues/03-property-contract.md)；该票的部分选择已有用户回复，评价方法和完整数据契约仍未冻结。
- 研究区分：老师明确要求、论文直接实现、迁移设计建议。历史 PLaID 与自然语言编码实验分别保存，不用旧编辑闭环成绩替代新性质任务的证据。
- 使用本地 Markdown tracker：[操作约定](../../docs/issue-tracker.md)。人工票只通过用户实际回答解决；研究建议不能代替人工选择。
- 2026-10-13 已按用户要求在项目根目录初始化 Git，分支 main；当前尚无提交或远程仓库。研究资产继续由决策票链接，既有实验快照保持原样。
- 外部 MLIP 已由用户选定，选择记录见 [确定首轮性质目标与数据口径](issues/03-property-contract.md)；后续仅核查固定模型接入，不继续做 MLIP 选型比较。
- 子智能体只能修改其负责的研究资产和研究票；地图由主智能体维护，既有实验、数据与凭据保持不动。

## Decisions so far

- [Qwen3-8B 纯 AR 实验](issues/14-qwen3-8b-pure-ar.md)：一轮 909 次更新和 32 条生成评估完成，严格有效 21/32；有效输出中 20 个集中于同一化学式/空间群，零目标七个请求全部无效。性质未评分，先分析输出集中和计数错误，不自动启动新的联合训练。

- [先运行匹配纯 AR 基线](issues/13-pure-ar-baseline.md)：一轮训练完成，32/32切换/边界通过，7/32结构严格有效；输出集中及计数自洽仍待解决，目标性质未评分。

- [验证子集数据与共享模型训练链路](issues/12-subset-training-preflight.md)：A100 技术预跑完成，数据容量、训练梯度和权重恢复通过；3次生成均未触发晶体段；该工程诊断不能替代尚未执行的匹配纯 AR 基线。

- [核查首轮单性质数据与评价依据](issues/01-data-evidence.md)：已核对现有配对数据；新生成晶体的性质评价及标签口径仍有待补齐的证据。
- [核查共享 AR 与 diffusion 的论文配方和实现差距](issues/02-method-evidence.md)：已区分论文配方、旧原型和迁移候选；新任务的预算可行性仍需要预检查证据。

- [核查 ehull 评价路线与能量口径](issues/08-ehull-evaluation-evidence.md)：已明确候选路线与资产/能量兼容性缺口；必须在专用校准集上实测后才能采用评价分数。

- [核查 GRACE-3L-OAM-L 接入与 ehull 标尺](issues/09-grace-evaluation-evidence.md)：已明确固定模型接口及 benchmark 对参考能量/相图的依赖；运行校准和资产准备仍未执行。

- [核查 CIF 可提取字段与离线描述提示词](issues/11-description-fields.md)：已区分直接字段、程序派生描述符与外部性质标签，并提取三条实际结构的基础字段。

## Not yet specified


当前能精确表述的开放问题均已建立子票。数据和方法研究可能揭示新的约束或需要追加的调查；收到证据后再扩展地图，不预设实现改造或后续大规模实验。

## Out of scope

- 新晶体 tokenizer/自然语言编码研究：用户已搁置，先隔离表示变量。
- replace/swap 编辑链、逆合成与材料工艺规划：未被确定为首轮任务。
- 多性质目标、长篇科学推理：用户已将首轮限定为一个性质和简短可核对文字。
- 全量模型训练、发布与超预算计算：技术预跑之外的完整实验仍待协议冻结。
- 再次比较或替换用户已选的 MLIP：模型选型已由用户完成。
- 将模型自述、预测器分数或单步结构变化直接当作实验验证的材料性能结论。
