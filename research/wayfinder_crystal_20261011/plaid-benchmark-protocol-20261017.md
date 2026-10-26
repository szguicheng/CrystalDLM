# PLaID++ 最新版基准口径核对（2026-10-17）

## 结论

截至核对日，arXiv:2509.07150 的当前版本是 **v4，2026-06-11 修订**；原 `plaid-task-metrics-evidence-20261016.md` 记录的是 v1。v4 的 unconditional Table 1 关键数值没有改变：旗舰 PLaID++ 的稳定率为 22.27%，S.U.N. 为 7.74%；最强此前基线 ADiT 的 S.U.N. 为 5.3%，相对增幅为约 46%。v4 增加了 bulk-modulus、band-gap 和 LeMat-GenBench 结果，因此 v1 笔记不能再被当成最新版全文的结果摘要。[当前 arXiv 版本记录](https://arxiv.org/abs/2509.07150)；[v4 正文](https://arxiv.org/html/2509.07150v4)。

对 CrystalDLM 当前固定任务，PLaID++ 提供的是相邻证据：它证明了以性质标签做偏好优化，以及空间群条件生成；它没有报告“结构性要求 + `ehull ≤ 0.08 eV/atom` 请求 → 生成候选达到该请求”的评估。论文中的 0.08 eV/atom 是 eqV2 的偏好分桶线，不能解释为经过验证的用户输入范围控制。

## 最新版结果、任务和阈值

| 口径 | v4 中的定义或结果 | 对比较的限制 |
|---|---|---|
| Unconditional S.U.N. | Table 1：旗舰 PLaID++ 稳定率 **22.27%**、S.U.N. **7.74%**；相较 ADiT **5.3%**，高约 **46%**。作者另称稳定样本中约 72% novel、55% unique。稳定指 eSEN 松弛后 `ehull ≤ 0 eV/atom`；unique 是生成批次内去重，novel 是相对于训练集结构匹配。 | 表内百分点是在 10,000 个可解析输出的设置下报告；不能和带条件的 S.S.U.N.、CrystalDLM 的请求级成功率直接互换。Novel/unique 均是结构匹配口径，不是实验可合成性。 |
| Space-group S.S.U.N. | PyXtal 判定实际空间群与目标相符，eSEN 松弛后 `ehull ≤ 0.1 eV/atom`，并且 unique、novel。七个目标群各采样 1,000 个：P1 (1)、C2/c (15)、Amm2 (38)、I-4m2 (119)、P3 (143)、P6₃/mmc (194)、F-43m (216)。正文报告：只加空间群偏好后相对 Wyckoff base 平均提升约 **22%**；再合并 unconditional 偏好后总提升约 **47%**。v4 另报告未直接优化的 8 个空间群平均改善约 **20%**。 | 正文没有给 S.S.U.N. 的逐群数值表；相对增幅来自 Figure 4 的聚合描述。不要把 22% 或 47% 当成绝对百分点，也不要从图上读近似柱高称为精确值。 |
| DPO 稳定性偏好分桶 | eqV2 预测：stable `ehull ≤ 0`；metastable `0 < ehull ≤ 0.08`；unstable `ehull > 0.08 eV/atom`。空间群偏好数据中，`ehull ≤ 0.08` 的结构算可接受。 | 这是偏好数据构造阈值；主结果的稳定/亚稳评估主要用另一套 eSEN 代理模型。它不是模型输入 `ehull ≤ 0.08` 的控制结果。 |
| v4 新增性质结果 | Bulk-modulus 任务表按 1,000 个样本统计：S.U.N. 且高 BM 的计数为 BM-only **28**、Joint **40**；高 BM 总数为 **32**、**54**。band-gap 表按 1,000 个样本统计 S.U.N. 且超过阈值的计数：Wyckoff base / PLaID++ It 1 分别为 `>1 eV: 4/15`、`>2 eV: 1/9`、`>3 eV: 0/3`。LeMat-GenBench 外部基准中，PLaID++ Stable **12.4%**、S.U.N. **1.0%**，参考集约 500 万结构并用多 MLIP 评估。 | 这些是 v4 新增的性质偏好/外部基准证据；LeMat 的 1.0% 与 MP-20 / eSEN Table 1 的 7.74% 使用不同参考集及评估协议，不能当作同一指标的回退。BM 表题称 conditioned generation，但不能据此推断任意性质条件的通用控制已验证。 |

正文定义 S.U.N. 为稳定、批内唯一、相对训练集新颖结构的占比。稳定率用 eSEN 松弛结构并以 Materials Project 2023-02 作为 hull 参照，采用 MP2020 energy corrections；DPO 偏好标签则来自 eqV2。条件生成通过 PyXtal 判定空间群。论文另以 1,000 个随机 unconditional 结构做 DFT：稳定率约 19.1%、S.U.N. 13%。作者解释 DFT 子集较小、unique 比率因此更高；这组 DFT 数不应与 eSEN 全量 10,000 样本数并列成等分母结果。[v4 §5.2–5.3、Appendix 7.1–7.4](https://arxiv.org/html/2509.07150v4)。

## 预算、解析重采样与分母

- v4 §5.1 对每个 unconditional fine-tuned model 采 **10,000 个结构**；无法从生成字符串解析 CIF 时重采。正文说明这样保证输出可解析成晶体对象，但不保证物理/结构有效。主指标的报告分母是目标数量的已解析样本；原始解码尝试数、解析失败数和重采次数没有报告。
- 空间群条件评估为 **7 群 × 1,000 个**。按论文口径比较时使用每群 1,000 个可解析样本，S.S.U.N. 的自然分母是每个目标群 1,000；失败解码没有作为已报告的独立失败率。
- 当前官方采样实现 [`llm_sample.py`](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/llm_sample.py) 会丢弃无法解析的输出并继续生成；条件采样会把失败的 prompt index 放回队列。评估实现 [`novelty.py`](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/evals/novelty.py) 取输入 CSV 前 10,000 行，再筛掉缺失 ehull，并在稳定筛选集内算批内唯一和相对 train split 的新颖性。采样代码与发布结果的运行提交未由论文逐项绑定，故这里只把它当作当前公开实现行为，不能证明论文每份 CSV 的实际行数或解码尝试数。

公平对齐本项目的 **每条结构请求最多 100 次原始生成尝试** 时，应把无效、解析失败、违反结构要求、重复和属性不达标都保留在 100 次预算中，单独报告每条请求的 useful-candidate yield / request success。PLaID++ 的 10,000 个解析后结构和 1,000 个解析后条件结构评测没有公布原始尝试总数，且任务输入不同；不可将其 7.74% S.U.N. 直接当作相同预算下的基线成功率。

## 支持的输入条件与证据边界

- v4 的 SFT 任务包含 infilling；其余 de novo 样本可在 prompt 中随机附加 1–5 项信息：化学式、ehull、每原子形成能、band gap、空间群。作者明确说研究重点是空间群条件生成。训练时出现某项字段不等于对该条件的生成控制已经评估。
- 当前公开 [`llm_sample.py`](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/llm_sample.py) 的条件模板是化学式、**标量** e_above_hull、空间群编号、band gap。它没有 `formation_energy_per_atom` 或 bulk-modulus 条件模板。论文的主 S.S.U.N. 实验是指定空间群编号；v4 还加了 BM 与 band-gap 的偏好目标实验。
- 对项目请求 `ehull ≤ 0.08 eV/atom`：论文确实用 0.08 的 eqV2 阈值来构造偏好对，也在 SFT prompt 配方中包含 ehull 字段；但报告没有用这一不等式请求测试生成结构的达标率。既不能把“偏好优化更稳定”当作请求条件服从，也不能将输入源材料标签当成生成结构的 ehull。
- PLaID++ 的主条件实验使用空间群模板，尚未验证“含层状骨架/指定配位”等更丰富的自然语言结构要求。故它不能覆盖本项目要求的完整输入契约：结构要求与另行给定的 ehull 范围同时作为请求，且生成结构由独立 evaluator 评分。

## 已发布模型身份

- 论文当前记录指向 [Hugging Face `HOPE-Lab-HMC/PLaID`](https://huggingface.co/HOPE-Lab-HMC/PLaID)。核对时 HF `main` commit 为 `91f3acacc43bbcc5256f50f1f92badfdcc323094`；其文件树有 `reference/` 与 `train_dpo/` 两个 adapter 目录。`train_dpo/adapter_config.json` 标注 `base_model_name_or_path=qwen-local/Qwen2.5-7b`、LoRA rank 8、target modules `q_proj` 和 `v_proj`、alpha 32、dropout 0.05。HF 文件名为 `adapter_model.safetensors`，不是一份独立的 Qwen 全量权重；模型卡将 `train_dpo/` 称为旗舰模型。
- 该身份信息可见于 [固定 HF snapshot](https://huggingface.co/HOPE-Lab-HMC/PLaID/tree/91f3acacc43bbcc5256f50f1f92badfdcc323094)、[DPO adapter config](https://huggingface.co/HOPE-Lab-HMC/PLaID/blob/91f3acacc43bbcc5256f50f1f92badfdcc323094/train_dpo/adapter_config.json)、[reference adapter config](https://huggingface.co/HOPE-Lab-HMC/PLaID/blob/91f3acacc43bbcc5256f50f1f92badfdcc323094/reference/adapter_config.json)。官方 GitHub `main` 在核对时为 `9ac91f07a263594bbaef1574efbf483a98152f0c`，与本仓库 `work/PLaID` 子模块 HEAD 相同。
- 当前论文 v4 未列出可与 HF adapter 文件对应的 checkpoint SHA / 训练运行名。公开文件树和 config 也未证明复现时所需的 `qwen-local/Qwen2.5-7b` 具体快照、reference/SFT 与 DPO adapter 的加载链、或 v4 新增结果是否由 HF 当前 adapter 产生。因此可称它为作者公开的旗舰 DPO adapter，但**不能保证该发布快照就是 v4 每项结果使用的精确 checkpoint**；仅靠该 adapter config 也不足以声称已验证推理可复现。未下载权重或运行推理。

## 与 2026-10-16 v1 笔记的变化

1. 版本从 v1 更新到 v4（2026-06-11）。Table 1 的 22.27% 稳定率、7.74% S.U.N. 及 ADiT 5.3% 基线未变；v4 摘要把 S.U.N. 表述为超过 50% 的相对提升，直接按 Table 1 计算仍是约 46%。
2. v4 增加 BM、band-gap 和 LeMat-GenBench 外部基准。新增结果扩展“性质偏好”证据，但没有补出 CrystalDLM 所需的 ehull 上限 prompt-following 评估。
3. 条件 S.S.U.N. 口径仍是 `ehull ≤ 0.1`、PyXtal 目标群正确、批内唯一及相对训练集新颖；0.08 eqV2 是 DPO 分桶，不是同一评测阈值。v1 笔记对主条件任务的界定保持成立。
4. 预算描述需补上公开采样代码的失败解析重采，以及“报告可解析输出数、不报告原始尝试数”的边界。对本项目 100 次尝试预算，应按尝试数作分母。
5. HF 发布身份补为 commit 和 adapter 配置，并将论文-v4-checkpoint 对应关系标为未验证。

## 一手来源

- 当前论文与版本历史：[arXiv abstract/version history](https://arxiv.org/abs/2509.07150)；[current v4 HTML](https://arxiv.org/html/2509.07150v4)；[v1 HTML](https://arxiv.org/html/2509.07150v1)。重点定位：v4 §5.1–5.3、Table 1–2、Appendix 7.1–7.4；v1 Table 1 和空间群实验。
- 作者官方实现：[andaero/PLaID](https://github.com/andaero/PLaID)；[sampler at pinned commit](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/llm_sample.py)；[novelty evaluator at pinned commit](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/evals/novelty.py)。
- 作者发布模型：[HF model card](https://huggingface.co/HOPE-Lab-HMC/PLaID)；[pinned model snapshot](https://huggingface.co/HOPE-Lab-HMC/PLaID/tree/91f3acacc43bbcc5256f50f1f92badfdcc323094)；[train_dpo adapter config](https://huggingface.co/HOPE-Lab-HMC/PLaID/blob/91f3acacc43bbcc5256f50f1f92badfdcc323094/train_dpo/adapter_config.json)。
