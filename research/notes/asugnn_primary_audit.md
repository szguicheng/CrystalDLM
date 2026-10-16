# ASUGNN primary-method audit

## 来源与实际阅读范围

- Barnie Cao, Daniel Anderson, Luke Davis, *ASUGNN: an asymmetric-unit-based graph neural network for crystal property prediction*, Journal of Applied Crystallography 58, 87–95 (2025), DOI [10.1107/S1600576724011336](https://doi.org/10.1107/S1600576724011336).
- 官方文章页面：[IUCr/Wiley metadata page](https://onlinelibrary.wiley.com/doi/abs/10.1107/S1600576724011336)。文章正文 PDF 在出版社页面标记为技术不可用，不能把摘要当作正文方法。
- 关键可读原始方法材料：[官方 IUCr Supporting Information PDF](https://journals.iucr.org/j/issues/2025/01/00/ei5123/ei5123sup1.pdf)，11 页。实际阅读了 Section 1 “Graph embedding in ASUGNN”（PDF pp. 3–6），包括 Node、Distance adjacency matrix、ASU adjacency matrix 和 BaTiO3 图例；Section 2 XRD 只用于确认任务边界，没有把它当作图算法证据。
- 公开代码核查：未找到作者公开的 ASUGNN 源码仓库。作者个人 GitHub（Bin-Cao）可见多个公开项目，但没有可定位的 ASUGNN repository；Hugging Face 上有 `caobin/ASUGNN` 模型条目/活动记录，但本次未将其当作可读图构造源码。故以下图实现结论以官方 supporting information 为准。

## 1. 节点不是 prototype-only / orbit-only integer representation

Supporting Information §1.1（PDF p. 3）明确写道：ASUGNN 中 conventional unit cell 的**每个 atom 都是独立 node**；每个 node 是 106 维向量：92 维原子物理属性 one-hot、3 维 fractional coordinates、6 维 lattice constants，以及 space-group number、unit-cell atom count、ASU count、cell volume 和 cell mass 等全局量（pp. 3–4）。

因此 ASUGNN 的核心图输入仍是 full conventional-cell atom nodes，节点特征显式使用真实 fractional coordinates；它不是当前候选的 integer-only site-stabilizer/edge-orbit prototype，也没有只用 prototype + lattice 再做一次全量坐标去噪的生成接口。

## 2. 边/邻接如何编码对称关系

### Distance adjacency

§1.2（PDF p. 4）定义 `N×N` distance adjacency matrix：第 `i,j` 项是 unit-cell 中 atom pair 的 Cartesian distance；没有 operation/image label、cell translation code 或 affine matrix 字段，只有距离矩阵。

### ASU adjacency mask/weight

§1.3（PDF pp. 4–6）把 `N×N ASU adjacency matrix` 定义为施加 space-group constraint 的 mask/weight matrix。BaTiO3 例子（SG 221）中，unit cell 有 5 个 atom nodes（3 O、1 Ba、1 Ti），ASU sites 数量为 3（PDF p. 5）。对称等价的 O sites（例如 `(0,0.5,0.5)` 与 `(0.5,0,0.5)`）被判为同一 Wyckoff row；该矩阵对应项赋值 **1**。其他 atom pair 的 edge weight 是 Cartesian distance 的倒数，且为避免异常将最大 edge weight 截断到 **1.1**（PDF p. 6）。

原文定位的关键句（每条短引均少于 25 个词）：

- “each atom within the conventional unit cell ... is treated as an individual node” (Supporting Information §1.1, p. 3)。
- “The N × N ASU adjacency matrix is a mask matrix” (§1.3, p. 5)。
- “If atom i and atom j are symmetry equivalent sites ... the assigned value is 1” (§1.3, p. 6)。

### 这是否是 edge-orbit 计数合并？

在可读 supporting information 中，没有出现：

- 按完整 space-group edge orbit 聚合的 multiplicity/count 字段；
- 一个 source representative 到多个 symmetry images 的逐 orbit edge list；
- `n_klm`/Seitz/affine operation image label；
- 将多个等价 edge 合并为一条 edge 后再按 orbit count 加权；
- 从 prototype/ASU graph 可逆展开 full graph 的 procedure。

ASUGNN 的明确做法是：保留 `N` 个 conventional-cell atom nodes，并用 `N×N` 矩阵施加等价-site mask/特殊权重；它不是 candidate 的 ASU node + integer edge-orbit prototype 图。注意“等价 sites 项为 1”只说明该 mask/weight 的约束值，不能解释为完整 edge-orbit multiplicity。

## 3. 任务与可逆 prototype 边界

论文摘要和 Supporting Information §2 表明 ASUGNN 用于 formation energy、band gap、bulk modulus、metal/non-metal property prediction，并结合 powder XRD reciprocal-space features；不是晶体生成模型。Supporting Information 给出了从 graph/XRD 输入到 property prediction 的特征构造，没有生成 decoder、prototype→coordinates reversible map 或独立自由坐标 diffusion。

因此它不能支持以下更强说法：

- 已有 ASUGNN 把 integer-only stabilizer/edge-orbit prototype 作为生成状态；
- 已有 ASUGNN 用 edge-orbit count 取代全图邻接；
- 已有 ASUGNN 保存每条边的 affine operation/image 并可逆展开完整周期图；
- 已有 ASUGNN 证明 prototype-conditioned 单级坐标 diffusion。

它能支持的直接先例是更窄的：**用空间群/共轭子群信息压缩或约束 conventional-cell GNN 的邻接矩阵，并把对称等价 site 作为特殊 mask/weight 处理。**

## 4. 与当前候选的准确对比

| 方面 | ASUGNN（可核对部分） | 当前候选 |
|---|---|---|
| 节点 | conventional unit-cell 每个 atom 一个 node | ASU/Wyckoff-orbit representatives |
| 坐标 | node features 含真实 fractional coordinates | prototype + lattice，坐标作为后续去噪变量 |
| 对称关系 | ASU adjacency `N×N` mask/weight；等价 site 项为 1 | integer-only site stabilizer + edge-orbit prototype |
| edge image/operation | 未见 affine/Seitz/CIF image label | 候选显式保存 edge-orbit action/image |
| edge multiplicity | 未见 orbit count aggregation | 候选按 edge orbit 加权 radial sum/vector sum |
| 任务 | property prediction + powder XRD | 条件坐标生成组件（尚未完成完整 diffusion） |
| 可逆展开 | 未报告 | 候选目标为 prototype→full periodic graph/coordinates |

ASUGNN 因此是邻接约束/权重化 property GNN 的直接先例，但不是当前候选的完整 edge-orbit prototype generation 先例。其“open-source”声明与公开模型条目不等于本次已经核实到源码；正文不可访问部分的更细网络层实现保持未确认。

本审计未运行模型、未训练、未联系作者。
