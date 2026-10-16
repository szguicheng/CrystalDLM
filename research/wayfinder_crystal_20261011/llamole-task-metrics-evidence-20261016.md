# Llamole v1：条件分子生成与逆合成的任务契约和评价证据

研究日期：2026-10-16。本文以 [arXiv:2410.04223v1](https://arxiv.org/html/2410.04223v1) 为论文证据版本；另检查作者 [liugangcode/Llamole 官方仓库](https://github.com/liugangcode/Llamole) 当前 `main` 代码。当前仓库代码只用于核对输入接口，不把它当成论文 v1 的冻结实现或结果复现。

## 结论

Llamole 的任务不是“改一句提示词让 LLM 生成分子”。它把**自然语言结构/用途要求、独立提供的性质条件、图结构生成、逆合成规划**连成一个任务：LLM 解释问题并选择模块，Graph DiT 生成分子图，GNN 产生逆反应候选，A* 搜索把目标分子连到可购买原料。属性达成用生成后性质 oracle 评分，逆合成成功率另行统计。

对 CrystalDLM 可借鉴的是明确的输入/输出契约：结构设计要求和目标性质需同时给出，性质值进入明确的条件字段；模型输出可解析的晶体结构及与该结构绑定的文字；性质必须对生成结构独立评价。分子逆合成路线没有直接的晶体任务对应物，除非另行定义材料合成规划问题。

## 版本与证据边界

- 论文版本是 2024-10-05 的 arXiv v1。下表论文数字与方法描述只指这个版本。
- 仓库 `main` 于本次查阅时提供了例子、配置和数据接口。它的 `qwen_material.yaml` 使用 `<rollback_start>/<rollback_end>`，而论文 v1 附录 A.1 写的是 callback token；仓库评测调用 A* 时设 `iterations=100`，论文方法段写最多 300 次。论文 v1 文字描述类别 one-hot/连续聚类编码，当前 `main` 的 `ConditionEmbedder` 则对每一列标量走单独 MLP。下文将仓库代码明确标为当前 `main`，不据此替换论文 v1 的方法或结果。[官方配置](https://github.com/liugangcode/Llamole/blob/main/config/generate/qwen_material.yaml) · [官方条件嵌入实现](https://github.com/liugangcode/Llamole/blob/main/src/model/graph_decoder/conditions.py#L540-L604)
- 论文 §5.1 的测试集数量标签与 §4、附录 D.1 相反。本文采用 §4 和 D.1 的一致口径：药物/小分子 9,986 条，聚合物材料 750 条；保留 §5.1 的文字错误说明，避免把两类任务数量写反。

## 输入：问题文本与独立属性向量

MolQA 的问题由一条自然语言指令构成，要求设计（通常还要求合成）同时满足结构和性质条件的 molecule/polymer。问题构造模板将 `{property_info}` 和 `{structure_info}` 放入问题生成提示词，问句可以要求功能、组成特征、环/链或其他结构信息。论文例句形式包括“设计具有 X、Y、Z 性质的分子/聚合物”，并明确允许同时写性质和结构要求。[论文 §4、附录 B.2.4](https://arxiv.org/html/2410.04223v1)

这条文本之外，Graph DiT 另收目标性质条件。论文 v1 描述类别条件采用 one-hot，连续条件使用基于聚类的编码 `Linear(Softmax(Linear(c)))`；每个条件都有 learned drop embedding，缺值时使用该 embedding。各条件嵌入与文字嵌入求和。Graph DiT 预训练时，文字通过冻结 SciBERT 逐 token 表征平均后再线性映射；进入 Llamole 后，文本条件改由 LLM 在 `<design_body>` 等 8 个 query token 上的末层隐状态平均，再线性投影到图模块。两者是不同阶段的文字编码，不能混写成推理时一定使用 SciBERT。[论文 §3.2、附录 C.1、C.3](https://arxiv.org/html/2410.04223v1)

作者当前 `main` 的评测代码进一步确认了数据接口：`MolQADataset` 从每条样本的 `property` 字典按固定顺序抽值，并将缺项补为 NaN；`run_eval` 将该张量作为 `molecule_properties` 独立传给 `model.generate`。这说明该评测实现不靠 LLM 从问句自行恢复精确数值。[官方 `eval/dataset.py`](https://github.com/liugangcode/Llamole/blob/main/src/eval/dataset.py#L383-L474) · [官方 `eval/workflow.py`](https://github.com/liugangcode/Llamole/blob/main/src/eval/workflow.py#L793-L827) · [官方 `modeling_llamole.py`](https://github.com/liugangcode/Llamole/blob/main/src/model/modeling_llamole.py#L3159-L3292)

当前 `main` 的条件向量顺序是 `[BBBP, HIV, BACE, CO2, N2, O2, FFV, TC, SC, SA]`。前 8 项是主实验性质；`SC`、`SA` 是合成复杂度/可及性相关的附加条件。`data/property_ranges.json` 给出当前例子数据的各性质 min/max；训练与评测样本缺少的性质以 NaN 表示。[官方 `eval/dataset.py`](https://github.com/liugangcode/Llamole/blob/main/src/eval/dataset.py#L401-L434) · [官方 `property_ranges.json`](https://github.com/liugangcode/Llamole/blob/main/data/property_ranges.json)

仓库里的材料样例把自然语言“高 CO₂ 渗透率、低 N₂/O₂ 渗透率、含芳香环”等要求，与 sidecar `CO2=0.94, N2=0.0, O2=0.0, FFV=0.381, SC=2.28, SA=4.21` 放在同一条样本里；药物样例则以“同时抑制 HIV 与 BACE”的问句配 `HIV=1, BACE=1`。这些只是仓库的少量演示样本，不能当作整套 benchmark 的标签统计。[官方材料样例](https://github.com/liugangcode/Llamole/blob/main/data/molqa_material_examples.json#L362-L383) · [官方药物样例](https://github.com/liugangcode/Llamole/blob/main/data/molqa_drug_examples.json)

标签并非全部来自实验测量：论文先汇集已有标签，再用监督 GNN 生成伪标签；多任务 GIN 对不同 rationale/environment 子图的预测方差倒数形成置信度，只保留超过中位置信阈值的预测供标注。随后 Llama-3-70B 根据属性、结构和合成复杂度写描述，另一轮生成问句。论文说不同目标并非都具备所有性质，因此缺失条件是数据契约的一部分。[论文附录 B.2.2–B.2.4](https://arxiv.org/html/2410.04223v1)

## 输出：分子图与路线

推理先由 LLM 生成设计说明并预测 `<design_start>`，8 个 query token 把之前的文本压成条件向量；Graph DiT 随后生成目标分子图。生成图通过 graph encoder 的 `<molecule>` 位置返回 LLM 上下文，之后 LLM 可继续生成文本。训练/数据文件将结构段写成 `<design_start>...<mol_start>SMILES<mol_end>`；这只是数据/接口标记，结构生成本身由 Graph DiT 图模块完成，并非纯文本 LLM 自回归拼写 SMILES。[论文 §3.1–3.3、附录 A.1](https://arxiv.org/html/2410.04223v1) · [官方 `modeling_llamole.py`](https://github.com/liugangcode/Llamole/blob/main/src/model/modeling_llamole.py#L3159-L3292)

逆合成阶段以已生成分子为根节点，在 AND-OR 树上用 A* 搜索：LLM 给出剩余合成步数启发式，GNN 对每个中间产物预测反应模板并提出反应物。路线中的每步包含模板/反应条件、反应式和反应物。论文的 LLM 启发式是五分类，得分依次为 `0, 1, 2.5, 4.5, 7`；反应预测负对数似然形成当前路径成本，可购买分子的启发式成本为 0。[论文 §2.4、§3.2、附录 A.2](https://arxiv.org/html/2410.04223v1)

## 数据与评价口径

MolQA 从带反应路线的 USPTO 产物构造任务。论文称约 139K 个目标分子有 1–10 步路线；其中 2,196 个适合作为聚合物单体目标。按路线长度分别抽取约一半且每种长度最多 3,000 个进入测试，得到约 11K 条测试任务：9,986 药物小分子、750 聚合物；其余约 126K 个目标用于指令微调。测试目标从用于图模型预训练的 MolPair 目标中排除。[论文 §4、附录 B.2.1、B.3](https://arxiv.org/html/2410.04223v1)

主性质任务共 8 项：药物类别标签 HIV 抑制、BBBP（血脑屏障渗透）和 BACE 抑制；聚合物连续性质 CO₂、N₂、O₂ 气体渗透率、自由体积分数 FFV 和热导率 TC。训练文字还包括 SC/SA 合成复杂度，但主表不是 10 项性质 benchmark。论文没有给出 0/1 类别编码的完整语义映射表；类别任务按数据集标签评分，不能自行补写阈值。[论文 §4、附录 B.2.2](https://arxiv.org/html/2410.04223v1)

对生成结构用随机森林 oracle 预测性质，类别任务报告 balanced accuracy `BA=(TPR+TNR)/2`，连续材料性质报告目标值与预测值的 MAE。其他指标是化学有效率、基于 Morgan fingerprint 的结构相似度、对参考文字的 BLEU-4 与 ROUGE-L；聚合物有效性还要求至少两个 `*` 聚合点。[论文 §5、附录 D.1](https://arxiv.org/html/2410.04223v1)

这套结果**没有报告**每条请求达到预先定义数值阈值的命中率，也没有报告所有指定属性同时命中的 joint-success 比例；主表是逐属性 BA/MAE 汇总。随机森林分数是计算 oracle，不是实验测量。文中 BACE/HIV 双性质及结构约束案例属于单例展示，不替代整套联合成功率。[论文 §5.1–5.2、附录 D.1](https://arxiv.org/html/2410.04223v1)

性质优化基线是 GraphGA；它是图遗传算法性质优化方法，表中没有文字指标。另比较 14 个 LLM：ICL 使用按目标属性差异检索的 5 个最近问答示例；文本 SFT 对照用 LoRA 并把分子写成 SMILES。Llamole 测试的三个基座为 Mistral-7B、Qwen2-7B、Llama-3.1-8B。[论文 §5、附录 D.1](https://arxiv.org/html/2410.04223v1)

### 表 1：主控性质与分子生成结果

下表复写论文 Table 1 的 GraphGA、三个 Llamole 行；“最佳 LLM 基线（逐列）”是从 ICL/SFT 行按各指标取最优值，列间可能来自不同模型，不代表单个基线模型。类别 BA 与有效率/相似度/文字分数越高越好；材料性质 MAE 越低越好。

| 方法 | 有效率 | Morgan 相似度 | BLEU-4 | ROUGE-L | HIV BA | BBBP BA | BACE BA | CO₂ MAE | N₂ MAE | O₂ MAE | FFV MAE | TC MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GraphGA | 0.885 | 0.112 | — | — | 0.536 | 0.515 | 0.560 | 0.847 | 1.556 | 0.747 | 0.020 | 0.042 |
| 最佳 LLM 基线（逐列） | 0.797 | 0.136 | 0.226 | 0.388 | 0.471 | 0.483 | 0.515 | 2.222 | 2.322 | 2.119 | 0.110 | 0.086 |
| Llamole + Mistral-7B | 0.900 | 0.139 | 0.262 | 0.434 | 0.596 | 0.617 | 0.740 | 0.593 | 1.409 | 0.565 | 0.021 | 0.028 |
| Llamole + Qwen2-7B | 0.888 | 0.135 | 0.261 | 0.432 | 0.600 | 0.639 | 0.746 | 0.645 | 1.452 | 0.581 | 0.021 | 0.026 |
| Llamole + Llama-3.1-8B | 0.913 | 0.142 | 0.254 | 0.427 | 0.623 | 0.629 | 0.713 | 0.653 | 1.344 | 0.549 | 0.021 | 0.030 |

来源：[arXiv v1 Table 1](https://arxiv.org/html/2410.04223v1)。例如，Llamole 的 BACE BA 达 0.740–0.746，而 GraphGA 为 0.560；在聚合物 CO₂/ N₂/ O₂ MAE 上三个 Llamole 行均优于 GraphGA，但 FFV 上 GraphGA 的 0.020 略低于 Llamole 的 0.021。因而不能把结果概括成每项性质均胜过性质优化基线。

### 逆合成路线是独立指标

路线成功率表示 A* 是否在时间/迭代上限内找到从生成目标到 `G_avail` 可采购分子的逆合成路线，不表示性质达成。可采购池使用 Enamine Building Block 2024-06 版并补充常见离子和起始物，约 1.3M 项。论文方法段的停止条件是 30 秒或 300 次迭代，反应 GNN 每次提供 top-50 模板。[论文 §3.3、附录 D.1](https://arxiv.org/html/2410.04223v1)

| 任务 | 最佳 ICL 路线率 | 最佳文本 SFT 路线率 | Llamole Mistral | Llamole Qwen2 | Llamole Llama-3.1 |
|---|---:|---:|---:|---:|---:|
| 药物小分子 | 5.5% | 1.5% | 29.9% | 33.7% | 35.1% |
| 聚合物 | 4.8% | 0.8% | 14.3% | 17.9% | 17.6% |

来源：[arXiv v1 Table 2](https://arxiv.org/html/2410.04223v1)。论文另称 ICL 共有 777 个成功案例、SFT 324 个，Llamole 超过 10K，其中 40.48% 的成功案例需要至少两步。路线结果应单列，不能混入性质命中率。

## 消融能支持什么，不能支持什么

Figure 6 对 Graph DiT 依次去掉文字条件、再去掉文字和性质条件，并用 learned null embedding 代表被移除条件。论文的定性结论是：文本说明帮助 Graph DiT 理解结构要求；性质侧通道支持生成达到所需性质的分子。[论文 §5.2.1](https://arxiv.org/html/2410.04223v1)

主表也显示全系统比文本 LLM/SMILES 对照在多项指标上更强，但这不是单因素的 graph-versus-SMILES 消融：比较还同时涉及 Graph DiT、图预训练、模块调用、图编码器、数据和微调流程。可据此支持“多模态模块化系统在该 benchmark 表现更好”，不能把提升全部归因于图表示本身。论文没有给出 generated-molecule novelty/uniqueness 指标，也没有用 Morgan 相似度替代 novelty。作者称的“首个”是方法创新主张（文本与图交错生成），结构相似度是与参考分子的接近程度，二者含义不同。[论文摘要、§1、§5、附录 D.1](https://arxiv.org/html/2410.04223v1)

LLM 在此有两种清楚的作用：设计时解释并压缩之前的文字来条件化图生成；逆合成时给 A* 一个剩余步骤成本启发式。附录 A.2 的五档分值为 `0, 1, 2.5, 4.5, 7`。Table 3 的材料路线率（默认 LLM heuristic / domain heuristic / 不限时）分别是 Llama-3.1 `17.6/17.6/31.2%`、Mistral `14.3/14.7/27.3%`、Qwen2 `17.9/18.1/27.3%`。去掉 30 秒限制会显著增加成功率，说明这里的“可合成”也受规划预算影响。[论文 §2.4、§5.3.2、Table 3、附录 A.2](https://arxiv.org/html/2410.04223v1)

## 对 CrystalDLM 任务定义的迁移

论文证据支持把首轮问题定义为**带目标性质的结构条件生成**：输入同时给出精确的性质名称/单位/目标值或范围，以及文本表达的组成或对称性等结构要求；目标性质另作结构化条件，而非让模型从“更稳定”一类自然语言中自行猜数值。输出应包含可解析的 PLaID 晶体，文字描述必须对应实际生成结构。评价分层报告严格结构有效、结构约束是否满足、文字—结构事实一致性、生成结构经外部 evaluator 计算的目标性质差距/命中情况；文本自述和原样本性质标签都不能充当生成结构的性质分数。

这是从 Llamole 输入/输出和 oracle 评价得到的迁移建议，不是 Llamole 对晶体的实验结论。材料逆合成结果不证明晶体性质优化；若当前目标是 ehull 条件晶体生成，先定义合适的结构描述输入和生成后 ehull 评价即可，无需为了复刻分子论文而添加逆合成模块。

