# Phase 1–2 验收报告

日期：2026-10-25 PDT / 2026-10-26 UTC。状态：Phase 1 配置与环境、Phase 2 共享评价工具完成，停在闸门 2。未启动 Phase 3 训练，Design 补标保持暂停。

负责人本轮确认：design 比较适用于 AR-design、Hybrid-design、PLaID++；Understanding 单独评价。固定 LLaMA-Factory 0.9.5 / Accelerate 1.11.0；三卡训练。三卡与原 global batch 8 的冲突已单独确认改为 **global batch 12 = 3 ranks × 1 example × 4 accumulation steps**。

## 已完成内容

| 内容 | 实现与证据 |
| --- | --- |
| 共享配置 | [configs/shared.json](configs/shared.json)：原始 Qwen3-8B、全量固定划分、3 epoch、LoRA rank32/alpha64、lr2e-4、BF16、seed42、三卡/global12；AR-design 与 Hybrid-design 除目标定义外共享设置 |
| 框架安装 | LLaMA-Factory v0.9.5 commit `7af909522a951e3ad9f022ea6f88b6755257eaa5`，Accelerate 1.11.0；[版本回执](setup_20261026/versions-after.json)与安装前后包清单齐全，`pip check` 通过 |
| 可训练参数接入 | [项目内框架补丁](setup_20261026/llamafactory-control-tokens.patch)加入 PEFT 的四行 `trainable_token_indices`，避免扩词表后训练整块 embedding/head；微型 Qwen 验证参数集合及反向传播通过 |
| 原生数据处理 | [framework-preflight.json](framework-preflight.json)：真实 LLaMA-Factory processor/collator 审计全部 Understanding；只对答案及 EOS 计 loss，prompt/padding 为 -100 |
| Understanding | [evaluation/understanding.py](evaluation/understanding.py)：四字段、all-four、约化组成比较，错误字段独立扣分；逐材料回执 |
| NLL / 去噪 | [evaluation/losses.py](evaluation/losses.py)与[nll.py](evaluation/nll.py)：正确 shift、token 加权汇总、分段指标、固定互补 mask 与两种去噪归一化；去噪不命名为 NLL |
| Design | [evaluation/design.py](evaluation/design.py)：复用严格审计，实际 SG、按请求上限判性质、Stable/Unique/Novel/S.U.N.、SM-COV、多种子统计；缺少/失败的性质评价明确标状态及 null rate |
| 共享入口与追踪 | [evaluation/cli.py](evaluation/cli.py)、[plan.py](evaluation/plan.py)、[grace.py](evaluation/grace.py)：完整尝试计划、材料/尝试 ID 校验、输出/请求哈希绑定、run_meta、逐条 GRACE 持久化、禁止覆盖旧结果 |
| 完整参考库 | [reference-acceptance.json](reference-acceptance.json)：36,362 个训练结构、4,602 个 test 结构全部成功解码，没有通过丢弃失败行缩小参考库 |

训练环境仍为 Python 3.11.17、Torch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.18.0、pymatgen 2025.10.7、PyXtal 1.0.8。框架的其他传递依赖变更见 [dependency-plan.json](setup_20261026/dependency-plan.json)；80 个下载文件均校验 SHA-256 后安装。GRACE 独立环境未变更。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 单元测试 | **32 项通过**，见 [unit-final.log](unit-final.log)；覆盖分段 loss、解析负例、计划/回执关联、未评分状态、GRACE 失败与持久化 |
| 未定义变量/局部变量静态检查 | `ruff --select F821,F823` 通过 |
| 全量 Understanding 参考答案回放 | train **36,362/36,362**，validation **4,501/4,501**，test **4,602/4,602**；四字段及 all-four 全部 100%，见 [understanding-gold.json](understanding-gold.json) |
| 原生模板最长序列 | train **612**、validation **604**、test **605** tokens；控制 token 注册后长度短于 Phase 0 的原始 tokenizer 统计，全部低于 cutoff1024 |
| 原生标签解码 | 例如 `The formula is K1 Eu1 Cu2 Te4. The space group number is 99. The unit cell contains 8 atoms and 6 independent orbits.<\|im_end\|>`；输入晶体未进入 labels |
| 真实结构指标验收 | 32 个训练库内 fixture 的新颖性 **0/32**；train/test 各 32 个 fixture 自比较 COV-R/P 均 **32/32**；不符 SG 正确扣分 |
| 命令行验收 | Understanding 全部 test gold **4,602/4,602**；design 的三种子合成 fixture 正确去重、判非新颖、保存计数和元数据，见两份 [Understanding](acceptance_understanding_cli/summary.json) / [Design](acceptance_design_cli/summary.json)回执 |
| 数据保护 | 六个正式 JSONL/metadata 及 manifest 哈希保持原值，见 [frozen-data-after.json](frozen-data-after.json) |
| 交付一致性 | 30 个源文件/配置/补丁的本地—远程 SHA 一致；项目内缓存路径验证通过，见 [delivery-verification.json](delivery-verification.json) |

真实结构验收使用**合成性质回执**隔离指标逻辑，没有运行 GRACE，也不是任何模型的生成成绩。32 个 test fixture 中有 4 个按批准的 StructureMatcher 容差与全量训练库匹配；固定划分保持不变，新颖性指标会识别这类结构。该选取不是随机抽样，不据此估计全量匹配比例。

## 实际处理的问题

1. LLaMA-Factory 0.9.5 的默认扩词表逻辑会把完整 embedding/head 加入训练，与批准的四输入行＋LoRA 不同。采用项目内小补丁并通过真实框架/小模型测试，没有改共享 base 环境。
2. 服务器环境激活会覆盖外层 `TMPDIR/HF_HOME/PIP_CACHE_DIR` 等设置。首次串行下载因此短暂使用了 `/data1/users/liugang/tmp` 和该用户缓存；已停止本轮对应安装进程，保留失败记录，后续通过 [python.sh](python.sh)在激活后强制项目内路径。没有清理共享缓存或修改激活脚本。最终边界核查通过。
3. 旧实验归档后的相对路径及旧 7,272 条新颖性库不能直接沿用。新入口绑定归档审计/GRACE 资产，并读取当前完整训练库。
4. 防止没有 GRACE 回执时显示“测得 S.U.N.=0”：现在显示 `not_run/partial/failed`，对应 rate 为 null；实际结构指标仍可报告。
5. 修正了测试中的 FP32 容差、测试发现方式和 metadata fixture 布局。所有修正已回归测试；既有模型、归档源文件和正式数据未改。

## 下一阶段边界

请审阅一页[指标说明](evaluation/METRICS.md)。闸门 2 通过后，先进行 **AR-understanding 的 32 条过拟合检查**，通过后全量训练 3 epoch。Phase 3 训练入口需按共享协议接入每 epoch 保存、最佳/最终 checkpoint 保留和分段曲线记录；本轮没有启动该训练入口。

Design 仍缺 35,981 条描述，正式 JSONL 尚未发布；其 cutoff2048 只是待审计上限，不允许截断或用已有描述子集替代。补标未恢复。Hybrid 的 END 修复/原生代码对照仍属后续 Phase 4–5，受闸门 4 约束；PLaID++ 加载与实际生成属于 Phase 7。

尚未验证真实 8B 三卡训练、检查点 NLL 实测、新生成结构 GRACE 评分及完整 Design 长度分布。Design 自由描述事实一致性尚无自动评分契约，已在指标说明中单独标为未评价，不以四字段 Understanding 或描述 CE 替代。
