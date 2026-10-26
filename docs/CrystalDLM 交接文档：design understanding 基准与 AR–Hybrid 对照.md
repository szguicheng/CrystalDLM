# CrystalDLM 交接文档：design / understanding 基准与 AR–Hybrid 对照

面向执行 AI · 2026-10-25 · 项目负责人与 Claude 整理

## 0. 阅读方式

你是接手执行的 AI。按第 9 节的阶段顺序推进，每个阶段有「验收」；标有「闸门」的地方必须停下汇报，等负责人确认后再继续。

- 标「已定」的内容不要改；标「待确认」的内容先问负责人，不要自行决定。
- 文中没写的细节，选标准、保守的做法，并记入日志。
- 不确定时停下来问。不要为了跑通而改数据、改指标或悄悄换设置。

## 1. 目标

研究方向：让模型把自然语言与晶体结构联系起来，能按要求设计晶体，也能围绕晶体做描述与问答。技术假设：文字用自回归（AR），晶体用 diffusion（对 PLaID 离散 token 做掩码去噪），二者共享模型核心参数和同一上下文。这个假设是否优于纯 AR，目前没有结论。

本轮目的：先建立可复现、可比较的基准，再回答「hybrid 对 design 是否有价值」。具体做三件事：

1. 把 design（原 G）和 understanding（原 Q）作为两个独立任务，各有 train / validation / test 和明确的 metric。
2. 训练三个模型：AR-design、AR-understanding、Hybrid-design。
3. 在同一评价流程下比较，并在之后在同一任务上运行开源的 PLaID++。

此前「hybrid 未优于纯 AR」的结果不能当结论：起点是嵌套的 checkpoint，G 与 Q 混训，diffusion 采样在 END 附近存在训练/推理不一致，且 24/700 与 25/700 的差距在噪声内。旧结果只作背景，不要引用为发现。

## 2. 已定决策

| 项目 | 决定 |
| --- | --- |
| 任务命名 | G 改称 design，Q 改称 understanding |
| 任务关系 | 两个任务分开训练、分开评价，不混训 |
| 要训练的模型 | AR-design、AR-understanding、Hybrid-design 三个。Hybrid-understanding 取消，understanding 全部用 AR |
| 基座 | Qwen/Qwen3-8B（instruct / 后训练版本，不是 Base）。须核实实际名称，见 Phase 0 |
| 起点 | 每个模型都从原始基座出发，不得从任何已训练的 checkpoint 继续 |
| 训练轮数 | 每个模型固定 3 epoch |
| 数据 | 全量，不再抽样（不要 1/5 采样）。划分已完成，不得重新划分 |
| AR 训练框架 | LLaMA-Factory |
| Hybrid 训练 | 复用 HybridDiffusion（FLARE 思路）原生代码并适配晶体任务 |
| loss 记录 | AR loss 与去噪 loss 分开记录，不合并成一个数 |
| 验证 | 全程记录 train / validation 曲线 |

## 3. 背景要点

- PLaID 类表示：用文本记录空间群、晶格参数、Wyckoff 位点等，作为晶体的文本表达。
- design：输入条件请求（空间群、ehull 上限等），输出英文结构描述 + PLaID 晶体。
- understanding：输入真实晶体（及问题），输出四项回答：化学式、空间群、晶胞原子数、独立轨道数。
- Hybrid 的 diffusion 作用于 PLaID 离散 token（掩码去噪），不是对连续坐标加高斯噪声。参考 FLARE：同时训练干净的 AR 预测与带噪晶体 token 的恢复。
- 性质评分使用固定 GRACE 弛豫 + 参考凸包得到的代理 ehull，不是 DFT。
- 已知问题：现有 understanding 的答案常已出现在输入里，更像事实读取（见 5.3）。

## 5. 数据

### 5.2 格式

数据使用 LLaMA-Factory 的 JSONL 格式（模板参考 https://github.com/hiyouga/LlamaFactory/tree/main/data），`dataset_info.json` 已存在。先读 `manifest.json`、`dataset_info.json`、`metadata/`、`examples/`，据此确定字段与列映射，不要猜。

## 6. 训练指标定义

记号：x 为条件请求，y = (y\_1..y\_T) 为 design 的参考输出（英文描述 + PLaID 晶体）。

### 6.1 design，AR

L\_design,AR = -(1/T) · Σ\_t log p(y\_t | x, y\_\<t)

它衡量模型能否在条件和正确前文下预测参考输出的下一个 token。分别记录并报告：描述部分 CE、晶体部分 CE（各除以自己的 token 数），总体为按 token 数加权的结果。

### 6.2 design，Hybrid

- c：正确晶体 token 序列；M：被遮住、需要预测的位置集合。
- 使用两个互补的遮蔽版本：第一版遮住一部分位置，第二版遮住其余位置，因此每个晶体 token 恰好被作为一次去噪目标（FLARE 的 clean AR + 互补去噪）。
- S\_denoise = 两个版本上所有被遮位置的交叉熵之和。
- **L\_crystal,denoise = S\_denoise / T\_crystal**：每个晶体 token 的平均恢复误差。这是可解释的晶体去噪指标。
- **L\_denoise,current = S\_denoise / (T\_text + T\_crystal)**：当前旧代码实际优化的项，分母是整个 design 输出的 token 数。
- 两者含义不同，**必须分别命名、分别记录，不得混用**。
- 总目标：L\_design,hybrid = L\_design,AR + L\_denoise,current，两项按代码里的权重组合（若权重为 1 即直接相加；请记录实际权重）。AR 项覆盖描述和晶体，所以晶体 token 同时受到干净的 AR 监督和去噪监督。

### 6.3 understanding

z 为输入晶体及已给的上下文，q 为问题，a = (a\_1..a\_K) 为参考回答：

L\_understanding = -(1/K) · Σ\_t log p(a\_t | z, q, a\_\<t)

只对回答部分计 loss；输入的晶体、描述和问题不计 loss。请打印一条样本，把参与 loss 的 label 位置解码出来，亲自确认这一点。

## 7. 评价指标

### 7.1 Test NLL（design 与 understanding 都报告）

在 test 集的参考答案上计算每 token 的平均负对数似然。注意：

- AR 模型的 Test NLL 是精确的 AR NLL。
- Hybrid 的 AR 分支同样可算 AR NLL，但推理时晶体是用 diffusion 生成的，AR 分支的 NLL 并不反映实际生成过程。
- Hybrid 的去噪交叉熵是掩码恢复误差，在当前的掩码方案和权重下，**不是严格的似然，也不一定是似然的上界**。因此它不能与 AR 的 NLL 直接比大小。
- 所以对 Hybrid 要把 AR loss 与去噪 loss 分开报告，对比表里注明各数字的含义。去噪 loss 在 test / validation 上必须使用固定的掩码方案和随机种子，否则数值不可比。

### 7.2 design 的生成评价

采样协议：

- 条件取自 design test 集中的请求。三个模型使用同一份条件列表和同一组随机种子。
- 每个条件的生成次数 N 和条件数量见「待确认」；旧实验是 7 个空间群 × 每个 100 次，但样本量偏小。至少用 3 个随机种子，报告均值 ± 标准差，同时保留 k/N 计数。
- AR 与 Hybrid 的解码设置（温度、top-p、diffusion 步数等）各自固定并记录。两者调参预算要对等，不得只为一方调参。

| 指标 | 说明 |
| --- | --- |
| 格式合法率 | 输出能被解析为 PLaID，并还原成 pymatgen Structure |
| 结构有效率（strict valid） | 沿用旧实验的严格有效判据（在 deprecated 或 data\_preparation 中找到并复用；若改动须说明） |
| 空间群命中率 | 用 spglib / pymatgen 对生成结构实际分析的空间群，与请求一致 |
| ehull 达标率 | 固定 GRACE 弛豫 + 参考凸包得到的代理 ehull ≤ 请求上限 |
| 唯一性、新颖性 | 对满足条件的结构去重；新颖性相对训练库（有条件时也含验证集）判定 |
| S.U.N. | 同时 Stable（条件与性质达标）、Unique、Novel 的比例，分母为生成尝试总数 |
| COV-R / COV-P | 与 test 中一批真实晶体对比：COV-R 为真实晶体中能在生成集合里找到相似结构的比例（覆盖面）；COV-P 为生成晶体中能在真实集合里找到相似结构的比例（贴近真实）。越高越好 |

COV 的匹配方式与阈值（建议采用 CDVAE 式的组成 + 结构指纹距离，或 StructureMatcher）需由你选定、写清楚，并对所有模型一致。

目的说明：加入 validity 与 COV 等指标，是为了确保结论不只依赖 S.U.N. 一个数。可另外加一些在参考集上的分布指标（如密度、元素数分布的 Wasserstein 距离），这是建议项，不是负责人的要求，加入前先说明。

### 7.3 understanding 的评价

- 解码：贪心（temperature 0），固定 max\_new\_tokens。
- 解析并逐项比较四个字段，分别报告正确率：化学式、空间群、晶胞原子数、独立轨道数，再报告「全部正确率」。
- 比较规则要事先写明并测试：化学式用 pymatgen Composition 比较（避免顺序、约化形式带来的假错误）；空间群按数据里的表示（编号或符号）；原子数、轨道数按整数。
- 输入是 test 集里的真实晶体，不是模型自己生成的晶体。

## 8. 训练监控：怎么看曲线、要不要一直盯着

**常规做法不是人一直盯着随时停**，而是：

1. 训练前设置好自动评估：每隔固定步数在 validation 上算一次 loss（LLaMA-Factory 里用 eval\_strategy=steps 与 eval\_steps，并指定验证集），同时按固定间隔记录训练 loss，写入 TensorBoard 或 wandb。
2. 每个 epoch 结束（至少）保存一次 checkpoint。
3. 训练结束后统一看曲线，再做判断。过程中只需偶尔查看，出现 NaN、loss 突然爆炸、完全不下降时才中止。
4. 有些项目会设早停（val loss 连续多次不再下降就自动停）。本项目已定固定 3 epoch，因此**不做早停**。

**本项目的具体规则**：

- 评估频率：每个 epoch 至少评估 5 次。验证集用全量；若太慢，用一个固定不变的随机子集，并记录。
- 按片段拆分记录：design 记录描述 CE、晶体 CE；Hybrid 额外记录 clean AR 与两种去噪归一化（见 6.2）；understanding 只记答案 CE。合并平均会掩盖各部分的变化。
- 若验证 loss 在后期回升，把验证 loss 最低的 checkpoint 与最终 checkpoint 都跑完整评价并同时报告，交由负责人决定。**不要自行改动 epoch 数重训。**
- loss 看起来不降时的排查顺序：(1) 小批量过拟合检查，见 Phase 3；(2) 解码一条样本的 label 掩码，确认哪些 token 在计 loss；(3) 检查学习率、warmup、梯度累积下的 loss 平均方式；(4) 按片段拆开看；(5) 确认起点不是已训练过同任务的 checkpoint（那样 loss 本来就在平台期）。

## 9. 阶段任务

### Phase 0：只读勘察

- 读 `manifest.json`、`dataset_info.json`、`metadata/`、`examples/`。记录各任务各 split 的样本数、字段格式，用 Qwen3 tokenizer 统计 token 长度分布。
- 检查 design 数据是否已完整；若缺失，记录缺口，先做 understanding 线。
- 核实基座：找到 Qwen3-8B 的路径或 HF 名称，读 `config.json` 与 tokenizer 的 chat template，记录准确名称；确认是带 thinking 开关的后训练版 `Qwen/Qwen3-8B`，而不是 `Qwen3-8B-Base`。确定 LLaMA-Factory 里对应的 template，以及 thinking 模式的处理（训练与推理必须一致）。
- 记录环境：GPU 型号/数量/显存、CUDA、PyTorch、LLaMA-Factory、DeepSpeed / FSDP 的可用性。未安装则安装；无法安装就汇报。
- 定位并记录路径：旧 hybrid 实现、HybridDiffusion / FLARE 原生代码库、PLaID++ 代码与权重、旧的严格有效判据与 GRACE 弛豫评分代码。找不到的向负责人要。

验收：`00_recon_report.md` 写完，含上述全部信息与未解决问题。**闸门 1**：汇报勘察结果。

### Phase 1：固定训练方案

写一份共享基础配置：基座、3 epoch、固定随机种子、精度、优化器与学习率调度。微调方式沿用旧实验（LoRA 等），具体超参见「待确认」。AR-understanding 与 AR-design 可因序列长度不同调整 cutoff\_len / batch size，要记录。**Hybrid-design 与 AR-design 除目标函数及其必需的 attention mask 外，其余设置（数据、起点、LoRA 配置、学习率、batch、epoch、种子）必须一致。**

### Phase 2：评价工具

在 `evaluation/` 实现第 7 节的全部指标，任何模型训练之前就做好，并写单元测试：

- understanding：把参考答案当作模型输出，四项和全部正确率应都为 100%；故意改错一项，应只扣对应字段。
- design：把训练集晶体当作「生成结果」，新颖性应接近 0；把 test 真实晶体当作生成结果，COV-R 应接近 1；构造空间群不符的结构，应被空间群命中率正确扣掉。
- 同一份评价代码对三个模型和 PLaID++ 都适用，不允许每个模型各写一套。

验收：测试全部通过，附一页说明各指标的输入、输出、解析方式和打分规则。**闸门 2**：负责人审阅指标说明。

### Phase 3：AR 基线（LLaMA-Factory）

按 AR-understanding → AR-design 的顺序（design 数据就绪后再做第二个），每个模型：

1. 小批量过拟合检查：取 32 条训练样本训练多个 epoch，loss 应降到接近 0，贪心生成应能复现训练样本且格式正确。做不到先排查（见第 8 节），不要直接上全量。
2. 全量训练 3 epoch，按第 8 节记录曲线。
3. 在 validation / test 上评价：understanding 用 7.3，design 用 7.1 与 7.2。

验收：曲线图、最终与最低验证 loss 的 checkpoint 信息、评价结果表、`run_meta.json`。**闸门 3**：汇报 AR 基线。

### Phase 4：修复 END 边界问题

已知现象：Hybrid 的 diffusion 采样在晶体结束符（END）附近，训练与推理的处理不一致。该问题未修好之前不得训练 Hybrid-design，否则 Hybrid 的劣势可能只是 bug 造成的。可与 Phase 3 并行。

步骤：阅读训练样本构造与采样两段代码；写最小复现，对同一批晶体分别按训练时与采样时的方式构造 END 附近序列，比较 END 位置的 token、END 之后位置的处理（mask / padding / 忽略）、长度确定方式、attention 可见范围；修复使二者一致；写单元测试；用小数据短训练验证采样能在 END 处正确终止、END 之后没有垃圾 token。

验收：测试通过，附改动说明。**闸门 4**：负责人审阅改动后再训练 Hybrid-design。

### Phase 5：Hybrid-design

1. 以 HybridDiffusion / FLARE 原生代码库为参照，在同一小数据上比较 loss 曲线和采样输出，确认移植后的行为与参考一致。
2. 适配晶体任务（PLaID token 的遮蔽、attention mask、采样），按第 6.2 节分别记录 clean AR、L\_crystal,denoise 与 L\_denoise,current。
3. 先做 32 条过拟合检查，再全量训练 3 epoch。
4. 用 7.2 的同一采样协议评价。

### Phase 6：综合评价

对三个模型汇总全部指标：S.U.N.、validity、条件满足率、COV-R/COV-P、Test NLL（分开报告 AR 与去噪）、understanding 各字段准确率。多种子结果给出均值 ± 标准差。

### Phase 7：PLaID++

在同一组 design 条件、同一评价流程下运行 PLaID++，输出同样的指标。说明它的输入条件如何与我们的条件（空间群、ehull）对齐；对齐不了的部分写明，不要硬套论文里的 S.U.N. 数字。

### Phase 8：报告

放入 `report/`：设置、曲线、结果表、局限。结论措辞规则：差异落在多种子标准差以内时，不得写「更好」；明确哪些结论还有待验证；记录所有与计划不一致的地方。

## 10. 记录与复现规范

每次训练或评价都写 `run_meta.json`：命令行、配置文件、git commit、基座准确名称、数据 `manifest.json` 的哈希、随机种子、环境版本、起止时间、checkpoint 路径。日志保留，不覆盖旧结果。代码是否正确，靠单元测试和对照参考实现来证明，不要拿 LLM 的审阅结论当证据。

## 11. 待确认事项（先问负责人）

1. 微调方式：沿用 LoRA 还是全参；学习率、LoRA rank、batch size 等超参由谁定。
2. design 的 test 条件列表、每个条件的生成次数 N、随机种子数。
3. COV 的相似度判据与阈值，ehull 上限的取值。
4. understanding 的答案设置是否要改（只输入 PLaID，或加入需要推导的问题）。在得到答复前，沿用现有数据。
5. 是否加入额外的参考集分布指标。
6. PLaID++ 与我们任务条件的对齐方式。
7. 硬件限制与可用 GPU 时长。

## 12. 禁止事项

- 修改 `datasets/`，或重新划分数据。
- 从任何已训练的 checkpoint 继续训练，或把 G 与 Q 混训。
- 只为 AR 或只为 Hybrid 单方面调参。
- 把 L\_denoise,current 称为「每晶体 token 的误差」，或把去噪交叉熵当作 NLL 与 AR 对比。
- 在差异小于噪声时声称 Hybrid 优于或不优于 AR。
- 在 END 问题修复并经审阅之前训练 Hybrid-design。
- 在 design 数据缺失时自行补造描述。
