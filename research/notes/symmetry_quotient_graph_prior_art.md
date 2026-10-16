# 周期晶体对称商图与 bond symmetry edge：先例查重

## 候选与判别标准

候选表示把周期 crystal bond graph 按完整空间群作用进一步约简：节点只保留 asymmetric-unit/Wyckoff orbit 代表；边按 bond 的空间群操作和晶格平移形成 edge orbit，并保留类似 CIF bond symmetry code 的操作标签；再由轨道代表和完整 action 展开恢复周期图。它需要与只按平移商化的 SLICES labeled quotient graph 区分，也需要与只生成 asymmetric-unit 原子/坐标、没有 bond-edge orbit 的模型区分。

## 1. SLICES：平移 labeled quotient graph，不是完整空间群 orbit quotient

### 来源与实际阅读范围

- 原始论文全文：[An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10622439/)（Nature Communications, 2023）。本次核对了 “SLICES representation”、labeled quotient graph、reconstruction 和 Eon graph-theory methods 部分；官方实现：[xiaohang007/SLICES](https://github.com/xiaohang007/SLICES)。
- 另有作者公开页面：[Nature Communications version](https://www.nature.com/articles/s41467-023-42870-7)。

### 表示的精确边界

论文将无限周期结构压缩成有限 quotient graph：unit-cell atoms 是 nodes，bonds 是 edges；为保证可逆性，每条 edge 保留连接到相邻 unit cell 的 translation vector/direction label。其 SLICES edge 形式为 `uvxyz`：`u,v` 是 unit-cell node indices，`xyz` 是平移到相邻 cell 的整数向量。论文示例中的 `001` 表示从一个节点连接到另一节点在 `+c` cell 的复制品。重建流程把 SLICES 转成 labeled quotient graph，再用周期图 barycentric/Eon embedding、几何优化和 M3GNet refinement 恢复结构。

这确实是周期 bond graph 的**平移商图**，边标签保留 translation gain；但节点仍是 unit-cell atoms，原文没有将具有相同完整空间群 orbit 的 atoms 合并为单个 asymmetric-unit/Wyckoff node，也没有将 rotation/reflection/screw/glide 操作作为 bond-edge orbit 标签。SLICES 不能据此支持“完整空间群 action quotient bond graph 已有”。

### 生成关系

SLICES 作为可逆字符串表示可与生成/逆向设计模型结合，官方仓库也包含 reconstruction 和 MatterGPT 相关工具；但原始论文的核心是 representation + reconstruction，并非按完整空间群 orbit 做 graph tokenization。候选若实现 orbit-level nodes、space-group operation edge labels 和 deterministic expansion，超出了原始 SLICES 的平移 quotient 定义。

## 2. SLICES-PLUS：把空间群操作附加成 symmetry sequence，但仍非 orbit-edge quotient

### 来源与实际阅读范围

- 原始论文：[SLICES-PLUS: A Crystal Representation Leveraging Spatial Symmetry](https://arxiv.org/abs/2410.22828)，作者 Baoning Wang、Zhiyuan Xu、Zhiyu Han、Qiwen Nie、Hang Xiao、Gang Yan；官方代码链接见论文/作者仓库：[wbnbb/SLICES-PLUS](https://github.com/wbnbb/SLICES-PLUS)。
- 本次实际阅读了官方 arXiv HTML 的 §2.1（lines 35–49）、§2.2–§2.4 生成/条件实验（lines 50–92）、Methods 2.5 representation（lines 98–105）、§2.6 dataset construction（lines 106–116）、§2.7 MatterGPT architecture（lines 117–139）和代码/数据说明（lines 159–161），不是只看摘要。

### 精确重合与差异

SLICES-PLUS 保留原始 SLICES 的 atomic symbols、node indices、translation-labeled edges，并额外编码由 CIF/Pymatgen SpacegroupAnalyzer 提取的对称操作。其 symmetry sequence 把 rotation matrix 的离散字符编码与 fractional translation vector 拼成约 6-character operation token；论文明确说这是把 rotation + translation 信息附加到原 SLICES 字符串，而不是修改原有 decoding。模型用 MatterGPT 做 crystal-system/property-conditioned generation。

这比 SLICES 更接近“把完整空间群操作作为 token”，但它的操作序列是附加的全局/位点 symmetry descriptor，不是按 bond 的 edge orbit 聚合；§2.1 还说明实验主要聚焦 general Wyckoff positions。论文没有定义“一个 edge 代表一组由完整空间群 action 互相映射的 bonds”，也没有给出从 orbit-level edge token 展开完整周期 bond graph 的算法。因此候选的 bond-edge quotient 仍不是 SLICES-PLUS 已报告的表示。

## 3. WyckoffAUGen：asymmetric-unit 生成先例，但没有已核实的 bond graph quotient

### 来源与实际阅读范围

- 官方 ICLR 2025 workshop 页面：[Crystal Generative Modeling with Explicit Autoregressive Conditional Likelihoods and Nontrivial Space Group Stabilizers](https://iclr.cc/virtual/2025/33005)，作者 Rees Chang、Alex Guerra、Nick Richardson、Ni Zhan、Sulin Liu、Angela Pak、Ryan Marr、Alex Ganose、Ryan P. Adams、Elif Ertekin。
- OpenReview PDF 页面受 challenge 阻断，不能直接下载附件；官方 ICLR 页面可读摘要/方法简介（lines 166–177），搜索索引还暴露了论文首页和部分 §1/方法文字（例如 `P` 由 fractional atom coordinates、Wyckoff positions 和 atom count 构成）。本次没有把未能逐页核对的 Appendix A.1–A.3 或完整算法细节当作已读证据；以下只记录实际可核对内容，并明确不确定项。

### 已核对到的事实

WyckoffAUGen sequentially builds crystals in the asymmetric unit with explicit hard space-group constraints；官方摘要明确说模型分布定义在 asymmetric unit，应用 space-group actions 后 tile 到整个 `R^3`，只生成 symmetrically inequivalent atoms。索引到的正文开头把 stabilizer-conjugacy 与 Wyckoff position 作为基础定义，并把连续位置建模为 asymmetric-unit facets 上的 simplex-to-polygon differentiable bijection。它是“orbit-node/ASU generation”的直接相关先例。

在已实际读取的官方摘要、索引正文和 ICLR 方法简介中，没有出现 bond graph、CIF bond symmetry code、edge orbit label、double-coset construction 或从 quotient edges 展开周期 bond graph。由于 Appendix A 未能直接下载，不能把“附录绝对没有这些内容”作为已核实结论；截至当前可读范围，其明确贡献仍是 asymmetric-unit atom/position likelihood 和 stabilizer-aware generation，不能用来证明完整空间群 bond-edge quotient 已有，也不能将 asymmetric-unit node generation 等同于候选的 edge-orbit tokenization。

## 4. IUCr CIF：bond symmetry code 已标准化，但不是商图/生成模型

### 官方规范与实际阅读范围

- [IUCr Core CIF dictionary – SPACE_GROUP_SYMOP](https://www.iucr.org/resources/cif/dictionaries/browse/cif_core)。阅读了 `_space_group_symop.id`、`operation_xyz`、Seitz/translation 定义。
- [IUCr checkCIF SYMOP01](https://journals.iucr.org/services/cif/checking/SYMOP01.html)。阅读了 `_geom_bond_site_symmetry_1/2` 的 `n_klm` 规则。

### 精确语义

Core CIF dictionary 规定 space-group symmetry operation 为 `X' = W X + w`，并要求列出生成该 setting unit cell 所需的完整 operation representatives。checkCIF 的 bond symmetry code `n_klm` 中，`n` 指向 `_space_group_symop.id` 的具体操作，`k,l,m` 编码对操作后位置施加的整数 cell translations（加 5 避免负数）。因此 CIF 已经具备候选 edge 标签所需的**操作 + 晶格平移**语义：一个 bond endpoint 可以指向某个 symmetry-transformed/translated image。

但 CIF 只是逐条记录 bond endpoint 的规范代码；它没有规定按完整 space-group action 对 bonds 做 orbit quotient，也没有规定 asymmetric-unit node/edge token vocabulary 或生成/展开算法。候选可以复用 CIF 的 `n_klm` 语义，却不能声称首次引入操作+translation bond labels。

## 综合判断

本次四项最直接先例可分成三层：

1. **SLICES：** 已有 translation-only labeled quotient graph（unit-cell atom nodes，edge translation gains）；不等于完整 space-group orbit quotient。
2. **SLICES-PLUS：** 已将 rotation + translation symmetry operations 附加成 sequence，并用于 MatterGPT 生成；但没有按 bond edge orbit 聚合，也没有 orbit-edge expansion。
3. **WyckoffAUGen：** 已在 asymmetric unit 上建模和生成，并通过 space-group actions 展开；已读摘要/索引正文/方法简介未显示 bond quotient graph，但 Appendix A 由于 OpenReview challenge 未能核对。
4. **CIF/IUCr：** 已标准化逐 bond 的 symmetry operation + cell translation code `n_klm`；这是标签语义基础，不是 learned quotient graph。

在上述实际阅读范围内，没有找到已明确实现“节点按 Wyckoff orbit/asymmetric unit 合并、边按完整空间群 action 成 edge orbit、保留 CIF-like operation+translation 标签、再展开恢复完整周期 bond graph”的晶体生成模型。这个未找到结论仅限本次四项筛查，不能推广为全球文献不存在。

可辩护的候选差异是把 SLICES 的 translation gain 推广为**完整 space-group action 的 edge-orbit quotient**，并使该 quotient 成为生成模型的显式图 token/条件状态；需要证明相比 SLICES/SLICES-PLUS 和 WyckoffAUGen，能减少 token、保持 bond/geometry legality，并可无损展开。不能声称首次使用周期 quotient graph、首次使用 CIF symmetry codes、或首次只生成 asymmetric-unit 原子。

本筛查未编码、未训练、未运行模型。
