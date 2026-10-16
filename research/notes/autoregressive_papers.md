# 晶体自回归生成模型比较（全文阅读笔记）

本文档基于本地逐页 PDF/TXT 全文和与表示、条件生成、采样、附录实验直接相关的附录整理。三篇论文的官方 arXiv 当前元数据也在 2026-09-07 核对：WyFormer 为 v4（2025-06-05），CrystalFormer 为 v3（2025-09-12），Matra-Genoa 为 v2（2025-01-29）。除非特别标注，页码均指所读 PDF 页码。

## 一眼比较

| 工作 | 结构表示 / 基本 token | 生成范式与顺序 | 实际条件生成（论文做过） | 输出边界 | 对拟议“新 tokenizer + 非 AR 扩散 + conditional”方案的关键差别 |
|---|---|---|---|---|---|
| WyFormer（Kazeev et al., arXiv:2503.02407v4） | 空间群 token；每个 Wyckoff 位点是一个**复合 token** `(element, site symmetry, enumeration)`；离散表示，坐标不在主生成 token 中 | 无 positional encoding 的 Transformer encoder；训练时随机打乱 WP 集合，生成时按 element→site symmetry→enumeration 级联、自回归加 STOP | **空间群条件**实际用于 de novo；另有 property prediction（不是条件生成）；化学计量作为条件只是未来工作/附录讨论，不应写成已完成 | 先生成 Wyckoff representation（空间群+元素+位点），再由 PyXtal 随机实例化并用 CrySPR/CHGNet，或接 DiffCSP++ 得到完整坐标结构 | 新方案若保留离散对称先验和集合式 WP 结构，再加入新 tokenizer、set-BPE 或非 AR 采样，差异应定位在 tokenizer/生成过程/条件接口；论文本身的公式与性质条件范围仍有限 |
| CrystalFormer（Cao et al., arXiv:2403.15734v3） | 空间群、按字母顺序的 Wyckoff letter、元素为离散输入；自由 fractional coordinates 是连续变量（von Mises mixture），晶格是连续 Gaussian mixture；不是 BPE | decoder-only masked Transformer；按 Wyckoff letter 的字母序，从高 site symmetry/低 multiplicity 到一般位点，逐个生成 `w→element→x,y,z`，最后 lattice | 空间群条件生成；给定元素列表的 logit mask；结构补全/MCMC（双钙钛矿）；性质引导 MCMC（MEGNet band gap、formation energy）；不是化学式条件训练，也没有 SG 微调 | 端到端输出对称不等价原子坐标和晶格，采样时按 SG/Wyckoff 投影并对晶格对称化，因此是 full structure | 新 tokenizer+非 AR 扩散会替换其显式因果顺序与 MCMC likelihood；但 CrystalFormer 已有“性质后验”控制和 in-filling，不能笼统归为无条件 AR |
| Matra-Genoa（De Breuck et al., arXiv:2501.16051v2） | 可逆混合序列；元素、化学计量、稳定性、空间群、Wyckoff 位点、自由坐标、晶格均为 token 字段；离散字段 learned embedding，连续字段 Gaussian embedding；共享 990 个 unique WP 词表 | causal Transformer；前缀到后缀的 greedy/Boltzmann 自回归，示例 `TERNARY EHULL LOW → elements → stoichiometry → Sg → WP → coordinates → lattice` | **稳定性** low/high（阈值 0.075 eV/atom）实际条件；化学空间（元素集合）、指定空间群、指定 Wyckoff 位点实际使用；性质扩展只是原则上可做 | token 序列可逆，直接还原含自由坐标和晶格的 full structure；后续 uMLIP relaxation 不是表示补全 | 新方案的非 AR 扩散若针对连续坐标/全结构，将不同于其混合离散-连续因果动作空间；但其稳定性/元素/SG/WP 前缀条件已是 conditional AR 实例 |

## 1. Wyckoff Transformer（WyFormer）

### 版本与阅读完整性

- 官方 arXiv 当前页：提交 2025-03-04，当前 v4 为 2025-06-05；题名、作者和 ICML 2025 发表信息与本地正文一致。
- 本地逐页 TXT 为 32 页；WyFormer 原始 PDF 实际文件名是 `maskgxt.pdf`，而 `wyformer-2503.02407-preflight.json` 正确指向该文件（32 页）。这是工作目录中的旧文件名，内容和 SHA 已由根任务核对；因此按该 PDF/TXT 全文阅读，不把文件名差异当作论文缺失。

### 表示与 tokenizer

正文第 2.1 节（PDF pp.4–5）定义表示 `R=(G,E,W)`：第一个 token 是 space group；之后每个 WP 用三部分表示：元素、site symmetry、enumeration。论文把三者嵌入后拼接，再经线性层混合，故生成时的基本单位是一个把“元素+WP 身份”绑定在一起的复合 WP token，而不是原子坐标 token。附录 R（PDF p.29）明确把 token 写成 `(element, site symmetry, enumeration)`，并报告 MP-20 中 10,904 个 unique tokens（考虑 SG Euclidean normalizer 的选择）。

site symmetry 是跨空间群可复用的类别；enumeration 在同一空间群和 site symmetry 下区分不同 WP。空间群不是单一类别 embedding，而是由 pyXtal 生成的对称操作 one-hot 矩阵，经筛选和线性层得到 embedding（§2.2，pp.4–5）。球谐描述符（§2.1.1，p.4；附录 O/P，pp.28）只用于 property prediction/实验性采样尝试，主生成模型仍使用 site-symmetry+enumeration。

这不是 BPE：全文没有 subword merge、BPE 训练或可变长度字符串分词；token 是人工定义的离散字段/字段复合。坐标未进入主生成 token。附录 A（p.14）说明 Wyckoff representation 加上自由坐标和晶格参数才是完整无损表示，并给出一个结构只有 4 个连续参数；但 WyFormer 主文明确把坐标留给后续结构生成器。

### AR 范式、顺序、置换/对称性

§2.3（pp.5–6）给出 cascade：先预测下一 WP 的 element，再在已有 element 条件下预测 site symmetry，最后在已有 element+site symmetry 下预测 enumeration；训练时随机取已知序列长度、随机选择级联部分，MASK 未知部分，交叉熵训练。生成算法 1（附录 C，pp.15–16）从 space group 初始化，重复 element→site symmetry→enumeration，遇到 STOP 结束。它因此是自回归的，但训练输入是集合式、无 positional encoding。

模型对输入 token 集合是 permutation-invariant（§2.2，p.5），训练每轮随机打乱 WP 顺序以逼近生成时的排列不变性（§2.3，p.5）。作者也明确承认“自回归生成本身不是 permutation-invariant”，靠小 WP 数量和 shuffle augmentation 缓解。等价 Wyckoff representation 通过每个 epoch 随机挑选一个表示进行训练（§2.3，p.5）；这属于训练增强，不是严格的生成时群商。

### 已实做条件生成与输出

实际 de novo 实验（§2.4、§3.1，pp.5–9；附录 C、算法 1，pp.14–16）是：从 MP-20 train+validation 的 SG 分布采样/指定空间群编号，然后 AR 生成 WP representation。完整结构的两条下游路径是：

1. PyXtal 按 WP representation 随机实例化，六个随机初始化中选最低能量，再用 CrySPR+CHGNet relaxation；
2. PyXtal 初始结构交给 DiffCSP++（WyFormer-DiffCSP++）。

因此 WyFormer 主体的直接输出是 protostructure/Wyckoff representation，不是完整 fractional coordinates+lattice；full structure 来自后处理生成器。论文报告 1000/105/10000 样本的 validity、novelty、S.U.N./S.S.U.N.，并将 WyFormer 与 CrystalFormer 等比较（Tables 1–2，pp.7–9）。

“性质条件”在本文做的是独立 property prediction：formation energy、band gap（MP-20）以及 AFLOW 的 thermal conductivity、Debye temperature、bulk/shear modulus（§3.2，pp.8–9，Tables 3–4），不是把目标性质喂给生成器做 inverse design。论文声称对称性+组成携带性质信息，但实验模型是 separate property-prediction model（§2.3，p.5）。

“化学式/化学计量条件”没有完成为主实验。结论（p.9）说 stoichiometry **可以**像 space group 一样加入条件，且可用 rejection sampling 做 CSP；这是 future/possibility，不是已报告的 formula-conditioned benchmark。也没有报告微调一个通用语言模型来实现 SG 控制；附录 R/N 的 WyLLM 实验是另一个比较/消融（fine-tuned LLM 与 vanilla LLM），不是 WyFormer 的训练方式。

空间群对称性通过 WP 本身保证；WyFormer 生成的是“对称模板”，PyXtal/DiffCSP++负责坐标和晶格细化。WP 等价性通过 normalizer-aware representation、随机等价表示和等价模板指标处理（§2.1，pp.4；表 1 说明）。

### 与拟议方案的真实差别

若拟议方案是“新 tokenizer + 非 AR 扩散 + 某种 conditional 任务”，WyFormer 的可核对基线是：离散 WP 集合、对称先验和 AR cascade；完整坐标由外部 PyXtal/DiffCSP++完成。若新方案保留这些先验而替换 tokenizer（例如 set-BPE）或采样过程，差异应按实际实现与任务边界陈述；不能把 WyFormer 的 separate property prediction 或“将来加 stoichiometry”算作已经完成的 conditional generation。

## 2. CrystalFormer

### 版本与阅读完整性

- 官方 arXiv 当前页：提交 2024-03-23，当前 v3 为 2025-09-12；当前页显示 29 页、Science Bulletin 期刊信息。
- 本地 `crystalformer-2403.15734.pdf/.txt` 及 preflight 均为 29 页、PASS；逐页阅读正文 pp.1–16 和附录 A–E pp.17–29。

### 表示与 tokenizer

方法 §II.A（PDF pp.2–4）将晶体写成 SG `g`、Wyckoff letters `W`、元素 `A`、对称不等价原子的 fractional coordinates `X`、晶格 `L`。输入顺序按 Wyckoff letter 字母序排列；每步字段是离散 `w_i`、离散 `a_i`、连续 `x_i,y_i,z_i`，最后连续 lattice。自由坐标用 periodic Fourier features输入，概率用 von Mises mixture；晶格用 Gaussian mixture（p.3–4，附录 A pp.17–20）。

这不是 BPE。论文没有任何 BPE/subword vocabulary 学习；词表是化学元素（119，包括 padding）和 Wyckoff letters（28，包括 padding）的人工类别，见附录 A Table S1（p.18）。每个 Wyckoff letter 的 multiplicity 和可用坐标由 SG/WP 表决定。

### AR 范式、顺序、约束与输出

式 (1)（p.3）明确分解 `p(C|g)`：`w1→a1→x1→y1→z1→...→wn→an→xn→yn→zn→L`。采样从高 site symmetry、低 multiplicity 的特殊位点到低 site symmetry、较大 multiplicity 的一般位点（§II.A，pp.2–3；算法 2，p.19）。训练用 masked decoder-only Transformer 和 causal NLL（p.4）。

生成约束包括：WP 对指定 SG 有效；letters 必须字母序；无自由坐标的特殊位点最多占用一次；这些由 logit masking 实现（p.4，算法 1 p.17）。同一 letter 下多个连续位点按 fractional coordinate lexicographic order 排序。作者用 SG origin 选择固定连续平移自由度，并用 SG/WP 处理对称等价原子；坐标视为 periodic variables（p.4）。采样时将坐标投影到相应 WP、将 lattice 按 SG 对称化（附录 A 算法 2，p.19）。所以 CrystalFormer 是直接生成 full structure 参数的模型，输出不需另一个坐标 diffusion 才完整。

### 已实做条件生成、补全与性质控制

1. **空间群条件随机初始化**：指定七个代表性 SG，各采样 100 结构，与 PyXtal 同组成初始化比较，并做 DFT relaxation（§III.A，pp.7–9，Table 2；附录 C pp.24–26）。这是真实 SG-conditioned generation。
2. **结构补全/元素替换**：双钙钛矿 `Fm-3m` 的字符串留空元素和 e 位 x 坐标（Fig.8，p.10），先用 MCMC sweep 更新元素和坐标，再 AR 采样 lattice；与 pymatgen ionic substitution 对比（§III.B，pp.9–10；附录 D p.27）。这是真实 structure-conditioned in-filling，不是无条件采样。
3. **性质引导**：训练好的 CrystalFormer 作为 `p(C|g)` prior，外接预训练 MEGNet band-gap 和 formation-energy predictor，按 Bayes posterior `p(C|g,y)∝p(y|C)p(C|g)` 做 MCMC；目标示例 `E_g=2 eV`、`E_form=-3 eV/atom`（§III.C pp.10–11；附录 E pp.28–29）。性质模型没有在 MP-20 上重新训练/微调，而是 MatGL 提供的预训练模型（附录 E p.28）。因此这是实际的 property-guided conditional sampling，但条件通过外接 likelihood+MCMC 进入，不是 Transformer 内部额外 condition token。

元素列表也可以通过 mask 化学元素 logits（§II.A，p.5；算法 2 p.19）；论文没有把“给定化学式”作为训练/评测任务。对实验观测/衍射谱条件的讨论是“同样可行”的框架扩展（§III.C，p.10），不是本文已做实验。没有通用模型微调作为 SG control 的主方法。

### 与拟议方案的真实差别

CrystalFormer 已经同时覆盖 SG-conditioned full-structure generation、结构/原型补全和外接 property-guided MCMC；因此把所有 AR 模型概括成“无条件”会错误。拟议非 AR 扩散的新增点，应具体表述为并行/去因果生成、或新 tokenization/更直接的 formula/性质输入，而不能声称“首次加入条件生成”。它的离散顺序仍是显式字母序 AR，性质控制需要额外 MCMC，且化学式条件并未作为已验证的端到端任务。

## 3. Matra-Genoa

### 版本与阅读完整性

- 官方 arXiv 当前页：提交 2025-01-27，当前 v2 为 2025-01-29；当前页显示 14 页、作者与摘要信息。
- 本地 `matrageno-2501.16051.pdf/.txt` 及 preflight 为 14 页、PASS；逐页阅读正文 pp.1–10 和 Supplemental Information pp.11–14。

### 表示与 tokenizer

§II.A（PDF pp.2–3）定义可逆序列 `S_N={w_i}`，字段覆盖 composition、stoichiometry、structure、stability：元素、计量数、稳定性 token、space group、Wyckoff position、必要的自由坐标、晶格六参数。230 个 SG 的 1731 个 WP 经共享 common orbits 合并为 990 个 unique Wyckoff positions。图 1(c) 给出完整序列示例：`TERNARY EHULL LOW ... elements ... S139 ... WP ... x=... a=...`。

离散类别（元素、stoichiometry、SG、WP、稳定性等）用 learned embeddings；连续坐标/晶格字段用 Gaussian embeddings（§II.A p.2；§IV.B p.8）。经典 sinusoidal positional encoding 加到序列（§II.A p.2；§IV.B p.8）。全文没有 BPE、subword merge 或必须的 BPE tokenizer；它使用作者定义的共享晶体字段词表和连续编码。

### AR 范式、顺序、对称性与输出

§II.B（p.2）用 causal attention、离散 cross-entropy + 连续 MSE 的 mixed loss 预测下一个 token；推理按 greedy/Boltzmann 分布 `p(x_{n+1})∝exp(f/T)`。序列前缀先给条件（如 `TERNARY EHULL LOW`），再生成元素/计量、空间群、WP、自由坐标和晶格（p.2–3）。

Wyckoff orbit 将对称等价原子压成一个位点，并保留自由参数；表示被设计为 invertible。SG/WP algebraic orbit 约束在 token 序列转回结构时强制对称，因此论文报告的直接模型输出是包含坐标和晶格的 full structure，而非只到 protostructure；M3GNet/ORBITAL 后处理用于弛豫与稳定性筛选，不是补齐缺失坐标。

### 已实做条件生成

1. **稳定性条件**：以离 hull 距离 0.075 eV/atom 为阈值，将稳定性离散为 LOW/HIGH token，放在序列开头；这是本文实际训练和采样的主要 conditional mechanism（§II.B p.2）。摘要与结果声称得到比 PyXtal+charge compensation 高 8 倍的稳定概率。
2. **化学空间条件**：实验中可指定元素/化学空间（§II.B p.2–3）；Al–Ca–Cu ternary 实验明确“不告知公式或空间群编号”，从 2,000 个生成结构中恢复 WBM 测试集 15 个 hull 结构中的 11 个（§II.C p.4，Fig.3）。这里是元素集合/化学空间约束，不等同于把完整化学式 token 作为目标输入。
3. **空间群与 WP 条件**：论文将 required symmetry（SG）和 specific Wyckoff positions 列为框架支持的条件（§II.B p.2），并说明可用于探索 structural prototypes；正文没有为 SG/WP 条件单列与稳定性同样完整的 benchmark，因此应记为支持的条件接口/声明，而不是已完成的专门评测。
4. **性质条件的边界**：作者说条件化“原则上可扩展到任何性质”，但本工作限制为 stability、chemical space、required symmetry、specific WP。没有 band gap/formation energy 等连续性质目标控制实验。3M 结构实验按温度改变稳定性/多样性，稳定性评估用 ALIGNN 预测 hull distance（§II.D pp.4–6；§IV.D p.8），这不是额外性质条件。

### 规模、消融与限制

MP 与 Alexandria 共 2,668,720 训练结构（§IV.A p.7）；16 heads、12 blocks、d=512、约 41M 参数，7 张 H100 训练约 36 小时（§IV.C p.8）。3M 结构约 48 小时生成，约 1000/min（§IV.F p.9）。作者承认 greedy sampling 可能不是条件采样最优，未来可用 MCTS、beam search、GFlowNets（§III p.6），这不改变其当前 AR 身份。

### 与拟议方案的真实差别

Matra-Genoa 已经是“新字段 tokenizer + 条件 AR + 离散/连续混合动作空间”的强基线：它实际评测 stability prefix 与元素化学空间，并把 SG/WP 作为框架支持的条件字段；但 SG/WP 没有单列专门 benchmark。它端到端输出完整结构。拟议非 AR 扩散只有在并行/迭代去噪、条件融合方式、坐标/晶格建模或可验证的 formula/property task 上有清晰差异；不能把 Matra-Genoa 的“未来可扩展到任意性质”写成已经完成性质条件实验。

## 对比结论（避免误标）

- 三者都不是 BPE 模型；区别在人工晶体字段词表、WP 复合 token、连续坐标/晶格的处理。
- “AR = 无条件”不成立：WyFormer 做 SG-conditioned representation generation；CrystalFormer 做 SG 条件、结构 in-filling 和外接性质后验 MCMC；Matra-Genoa 做稳定性前缀、化学空间以及 SG/WP 条件。
- 输出层级不同：WyFormer 主模型止于 WP/protostructure，完整结构依赖 PyXtal/CrySPR/CHGNet 或 DiffCSP++；CrystalFormer、Matra-Genoa 的表示都直接包含坐标与晶格，可生成 full structure（前者是 SG/WP 投影+连续分布，后者是可逆混合序列）。
- 若新工作提出“新 tokenizer + 非 AR diffusion + conditional formula/property”，最可信的 novelty 对照是：相对 WyFormer 的端到端 full-structure/条件范围扩展、相对 CrystalFormer 的非因果生成和无需 MCMC 的条件注入、相对 Matra-Genoa 的并行去噪与连续/离散联合条件化；而不是笼统声称“首次使用条件生成”。

## 来源与核对记录

- 本地全文：`/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/work/papers/wyformer-2503.02407.txt`、`crystalformer-2403.15734.pdf/.txt`、`matrageno-2501.16051.pdf/.txt`。
- 本地 preflight：三篇均有逐页材料；CrystalFormer 与 Matra-Genoa 的 preflight PASS 且页数与 PDF 一致，WyFormer sidecar 正确指向实际 PDF `maskgxt.pdf`，页数 32。
- 官方元数据：<https://arxiv.org/abs/2503.02407>、<https://arxiv.org/abs/2403.15734>、<https://arxiv.org/abs/2501.16051>。

本文不依赖长段英文引文；英文原文逐字引用合计为 0 词，符合每篇不超过 25 词的限制。
