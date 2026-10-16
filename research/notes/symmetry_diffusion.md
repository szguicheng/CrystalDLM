# 对称晶体生成与拟议 tokenizer + 非 AR diffusion + conditional generation：逐篇核对

## 阅读与版本核验

- 本地材料：`wyckoffdiff-2502.06485.txt`（其对应 PDF 文件名是同目录的 `dynacrys.pdf`，属于旧文件名）、`symmcd-2502.03638.pdf/.txt`、`wyckoffdiff-adaptor-2601.08115.pdf/.txt`。
- 三份 TXT 均逐页通读至正文和附录（分别 1543、1276、706 行）；SymmCD 24 页、WyckoffDiff 对应 PDF 18 页、Adaptor 11 页。三份对应 PDF 的本地 preflight 均 `PASS`，且 WyckoffDiff 的 18 页提取完整。
- 官方 arXiv 当前页（2026-09-07 访问）：WyckoffDiff 是 **v4，2025-10-10 修订**，标题 *WyckoffDiff -- A Generative Diffusion Model for Crystal Symmetry*；SymmCD 是 **v3，2025-05-23 修订**，标题 *SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models*；2601.08115 是 **v1，2026-01-13**，标题 *Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models*（即本地文件并非标题“WyckoffDiff-adaptor”，后者是方法名）。

## 三篇逐项对照

| 工作（当前版本） | 表示 / tokenizer 基本单元 | 生成范式 | 条件生成：SG / 化学式 / 性质 / 部分结构 | 输出粒度与对称、置换处理 | 与拟议方向的关系 |
|---|---|---|---|---|---|
| **WyckoffDiff** (2502.06485v4) | 不是文本 BPE。先从经验分布采样空间群 `s∈{1,…,230}`，再在该 SG 的全部 Wyckoff 位上建离散 protostructure：受限位 `z⁰` 每位为“空/一个元素”（类别变量），非受限位 `z∞` 为“每种元素的计数 0…P”（矩阵中的离散计数变量）；元组 `(s,z⁰,z∞)`（§2.1、§3.1，pp.2–4）。 | **非 AR**；D3PM 离散去噪扩散（§3.3，p.4），各离散变量并行加噪/反向预测，WyckoffGNN 在所有位点的全连接图上预测每个类别的 clean 分布（§3.4，pp.4–5）。 | §3.2 的 Eq. (4) 明确学习 `pθ(z⁰,z∞|s)`，所以**给定 SG 的结构化去噪条件**存在；但主实验中的 `s` 仍由经验先验 `p̂data(s)` 先采样，没有独立的外部 SG 控制 benchmark。化学组成由扩散变量联合生成，未做性质或部分结构条件实验。§5 只是**后处理发现流程**：生成 20,000 个新 protostructure，PyXtal 半随机补自由度、MACE 约束松弛，再用 DFT 筛出低形成能/凸包候选（§5，pp.8–9；Figure 3），不能写成性质条件生成。 | 直接输出 **protostructure（SG+元素占据的 Wyckoff 描述）**，不输出一般位置的内部坐标和完整晶胞；后续 PyXtal/MACE 才实例化几何（§5，pp.8–9）。SG/Wyckoff 结构约束使受限位最多一个原子，等价原子由位点对称自动展开；没有原子序列的自回归排列问题。代价是一般位置内部参数丢失、几何依赖半随机后处理（§5，p.8；结论 p.9）。 | 与拟议方向的重叠：都是离散、并行/非 AR 扩散，并把晶体结构压缩成对称感知单元；因此不能声称“首次用非 AR diffusion 生成对称晶体”。拟议 tokenizer 的具体粒度、是否覆盖完整几何，以及需要哪些条件仍待确定，不能预先把它们写成既定差异。 |
| **SymmCD** (2502.03638v3) | 不是文本 BPE。先取 SG `G`、轨道代表数 `M`，再用非冗余不对称单元：`C′=(G,k,X′,S,A′)`，其中 `k∈R⁶` 是受 SG 约束的晶格参数，`X′` 是每个 orbit representative 的分数坐标，`A′` 是元素类别，`S` 是位点对称。`S` 用 15×13 二值轴-操作矩阵；SG 另用 `(26×15)+7=397` 维二值表示（§4.1–4.2，pp.4–6；Appendix E.1 附近）。平均 MP-20 仅 4.7 个 orbit 对 18.9 个原子，约四倍压缩（§4.1，p.5）。 | **非 AR** joint diffusion：元素 `A′` 与位点对称 `S` 用 D3PM 类别扩散；坐标 `X′` 用 wrapped-normal 连续扩散；晶格 `k` 用高斯连续扩散；同一 denoiser 联合预测（§4.3，pp.6–7；Algorithm 1–2）。 | 训练/采样接口把 SG `G` 和代表数 `M` 作为输入（Algorithm 2，p.7），因此有**给定 SG** 的生成/SG 分布控制；但论文没有做给定化学式、性质或部分结构的 conditional generation benchmark。模型反而明确指出 composition-conditioned CSP 较难，因为流程先采样 SG、再按 SG 采样组成（Conclusion，p.10）。形成能只作为 proxy/property evaluation（Table 3，p.10；Appendix F.3 的预测器消融），不是生成时条件。 | 输出不对称单元及其对称元数据，然后把代表原子按 `G,S` 展开成完整晶胞；采样末端将 `S` 投影到最近合法点群、`X′` 投影到兼容 Wyckoff 位，再复制 orbit（Algorithm 2，p.7）。轨道化表示消除等价原子置换；完全连接代表图和坐标差分用于消息传递（§4.3，p.7），denoiser 本身非 equivariant，但固定晶胞轴提供 canonical reference。 | 重叠更强：同样是对称感知、离散+连续联合、非 AR diffusion，且同样以 Wyckoff/orbit 作为基本单元；不能声称拟议方向首次提出“用对称压缩单元做 diffusion”。拟议 tokenizer、条件类型和输出粒度均待定，因此这里只保留方法重叠，不预设具体差异。 |
| **Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models / WyckoffDiff-Adaptor** (2601.08115v1) | 沿用 WyckoffDiff 的 protostructure，不是文本 BPE：SG `s`、一般位 `z∞` 的元素×计数矩阵、受限位 `z⁰` 的元素/空向量（§2.1，pp.4–5）。在 WyckoffGNN 各层插入 MatterGen 风格 Adaptor，把条件 embedding 注入输出 Wyckoff element matrix；后处理再由 protostructure 转 AFLOW/Pymatgen label、PyXtal CIF（§2.1，pp.4–5）。 | **非 AR 离散 diffusion**，本质是 WyckoffDiff 的 D3PM 反向生成加条件 adaptor；不是连续坐标 diffusion。 | 这是三篇中唯一明确做 conditional 实验的扩展：同时条件 **SG、化学系统/元素组合、energy above hull=0**（形成能稳定性 proxy）；在 MP-20 上训练，评估 Li–O、Ti–O formation-energy phase diagrams，并补做 ternary Al–Mg–O、quaternary Ba–Ta–In–O（§2.2，p.5；Figure 3，pp.7–8）。SG 条件经 Wyckoff 输入在优化后仍较好保持（Figure 2，pp.6–7）；化学式是 chemical system 条件，不是逐一指定完整化学计量 token 的保证。没有部分结构条件实验，也没有独立晶格/坐标 inpainting。 | 输出仍是 protostructure/CIF 两阶段：模型输出元素占据，PyXtal 补一般 Wyckoff 位内部参数并转换完整结构。优点是 SG 对称由表示显式保证；限制是一般位置内部参数未被模型生成，可能随机化，同一 protostructure 可得到不同几何；论文展示 Ba–Ta–In–O 中化学上不合理近邻，且 MatterSim 优化可能破坏输入 SG（§3，pp.7–8；Conclusion pp.8–9）。运行速度表 1：V100 16 样本 MatterGen 366 s，WyckoffDiff prototype 34 s + CIF 26 s ≈60 s（p.8）。 | 这是拟议 conditional 方向的最直接先例：不能声称“首次将 Wyckoff 对称扩散与 SG+元素+稳定性条件结合”。拟议 tokenizer、待选条件类型和输出粒度都尚未由用户确定，因此只将其定位为最直接 conditional 先例，不预先承诺具体创新差异。 |

## 条件能力的边界（避免过度声称）

1. **SG 条件**：SymmCD 的 Algorithm 2 明确输入 target SG；WyckoffDiff 原文主模型只按经验 SG 分布采样；Adaptor 才把 SG 与其他条件并列输入并做实验。
2. **化学式/元素条件**：三篇中只有 Adaptor 实验了 chemical system（Li–O、Ti–O，以及更大元素组合）。SymmCD 的结论明确把“给定 composition 的 CSP”列为限制；WyckoffDiff 的组成是无条件联合采样。
3. **性质条件**：Adaptor 的 `energy above hull=0` 是稳定性条件并生成形成能相图；SymmCD/WyckoffDiff 的形成能均是生成后评估或 proxy，不是采样时性质控制。
4. **部分结构条件**：三篇未报告 mask/inpainting、给定局部坐标或给定部分 Wyckoff occupancy 的实验。将“按 SG/化学系统条件”扩写成“部分结构条件”是不成立的。

## 与拟议新 tokenizer + 非 AR diffusion + conditional generation 的可用结论

- **不可声称首次**：非 AR 离散扩散（WyckoffDiff）、离散+连续联合非 AR 扩散（SymmCD）、对称感知的 SG/元素/稳定性条件扩散（WyckoffDiff-Adaptor）均已存在。若拟议 tokenizer 是结构化、对称感知、按位置/轨道生成，方法重叠尤其直接。
- **待定的比较维度**（需先确定拟议任务再实证）：tokenizer 是否统一且可逆、条件类型是否包含 SG/化学式/性质/部分结构 mask、输出是否覆盖完整晶胞几何或内部参数、是否显式处理等价原子置换，以及是否采用统一的条件满足率/几何有效性/对称保持率/novelty/uniqueness 评测。当前不能把这些待选项写成已确定的创新差异。
- **关键风险**：若仍把一般 Wyckoff 位内部参数交给 PyXtal 随机补全，就继承 WyckoffDiff/Adaptor 的细节丢失和化学不合理键问题；若只报告 unconditional novelty/stability，不能证明 conditional generation；若 tokenizer 只把离散类别串成序列却采用自回归 next-token 采样，也不能称非 AR。

## 原文短引（每篇合计不超过 25 个英文词）

- WyckoffDiff（p.1 摘要）：“**the discrete nature of our model enables fast generation**”（10 words）。
- SymmCD（p.1 摘要）：“**the asymmetric unit, the smallest subset of the crystal**”（10 words）。
- WyckoffDiff-Adaptor（p.3 摘要）：“**embeds conditional generation within a WyckoffDiff architecture**”（8 words）。

## 官方页面

- [arXiv:2502.06485（当前 v4）](https://arxiv.org/abs/2502.06485)
- [arXiv:2502.03638（当前 v3）](https://arxiv.org/abs/2502.03638)
- [arXiv:2601.08115（当前 v1）](https://arxiv.org/abs/2601.08115)
