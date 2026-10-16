# 对称 orbit graph 计算压缩的已有先例

## 候选范围

当前组件对周期 crystal graph 按 space-group edge orbit 汇总：保留 orbit representatives，以 edge-orbit multiplicity/权重做径向 message passing 和 vector sum；目标是在与 primitive full graph 相同的边权下保持前向/梯度一致并降低 GPU memory。这里区分三件事：

1. 输入结构是否只保留 asymmetric-unit/Wyckoff representatives；
2. 网络计算是否显式把重复 edge 以权重合并，且能与 full graph message passing 对应；
3. edge orbit 是否被当作可生成的离散 token。

## 1. Ruff et al.：最直接的 ASU graph 计算压缩先例

### 来源与全文范围

- Robin Ruff, Patrick Reiser, Jan Stühmer, Pascal Friederich, *Connectivity optimized nested line graph networks for crystal structures*, Digital Discovery 3, 594–601 (2024), DOI [10.1039/D4DD00018H](https://doi.org/10.1039/D4DD00018H)。
- 可读全文：[RSC HTML](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00018h)；早期版本：[arXiv:2302.14102](https://arxiv.org/abs/2302.14102)。
- 阅读了全文中的 unit-cell graph、asymmetric-unit graph、multiplicity/readout、connectivity optimization、Nested Graph/Line Graph Network 和 benchmark sections；没有只依据摘要。

### 直接重合

论文把周期晶体图的三种层次画在 Fig. 2：普通 full graph、只按平移商化的 unit-cell multigraph，以及按所有晶体对称性商化的 asymmetric-unit graph。ASU graph 将空间群对称等价 atoms 合并为一个 node，保留 multiplicity，并允许 self-loops 和 multi-edges。作者明确指出 unit-cell graph 中每个 node 代表所有 translation-equivalent atoms；ASU graph 进一步把 all symmetry-equivalent atoms 合并。

其计算主张与当前组件高度相关：普通 E(3)-invariant GNN 会给对称 atoms 相同 embeddings，导致重复 message passing；ASU graph 可以删除这些冗余，同时保持 node embeddings。原文 §2 “Exploiting crystal symmetries” 明确写道 “all symmetry-equivalent atoms are represented by a single node in the graph”，并说明 “self-loops and multi-edges can occur in asymmetric unit cell graphs”。论文文字本身没有给出 edge-orbit count，但官方实现把粒度明确到了函数级：`kgcnn/crystal/graph_builder.py` 的 `to_asymmetric_unit_graph`（v3.0.1 lines 489–527）先取 `graph.in_edges(asu_node_indice, data=True)`，然后逐个 `for e in edges_to_keep` 调用 `asu_graph.add_edge(...)`；它把 source 的 `symmop` 拷贝到每一条 ASU edge（lines 518–525），没有 edge multiplicity/count 合并字段。其上游 `add_knn_bonds`/`add_radius_bonds` 也逐条 `add_edge` 并写入 `cell_translation`、`distance`（lines 106–163、166–197）。因此，在该官方实现的图构造路径下，source ASU rep 到不同 symmetry-equivalent neighbor images 的 occurrence 是逐边保留并形成 MultiDiGraph multi-edges；不是单条 edge + orbit count weight。由于 ASU nodes 的 symmetry-related multiplicities 不同，global readout 必须按 multiplicity 加权；论文给出 readout adaptation，使 ASU 与 unit-cell graph 得到相同结果。作者将 connectivity optimization 与 nested line graph blocks 结合以提升效率。

### 边界

这是已发表的**symmetry-reduced graph + multiplicity-aware readout/计算压缩**先例，而且官方构造函数确认 occurrence-level multi-edge 路径。不能因为论文没有逐层写出 forward/gradient proof 就推断方法与当前组件不同：若相同消息函数、边消息和线性 aggregation 下把这些 occurrence 保留或按计数合并，梯度等价可由同一计算图直接随参数微分得到。真正已确认的区别是边压缩粒度：Ruff 官方实现保留每条 full-graph edge 的 `cell_translation`/`symmop` occurrence，未实现单条 edge-orbit + count weight；`MultiplcityReadout`（`_multiplicity_readout.py` lines 332–417）只处理 node multiplicity。故“ASU graph 降低 GNN 计算/内存”不能声称新；候选窄差异只能定位为**把该 occurrence-level MultiDiGraph 进一步合并成单边 edge-orbit weighted radial aggregation，并验证等价性/内存 benchmark**，而不是“首次证明梯度一致”。

## 2. ASUGNN（Cao, Anderson & Davis）：空间群 mask/weight GNN 先例，但不是 ASU-node graph

### 来源与可读范围

- Barnie Cao, Daniel Anderson, Luke Davis, *ASUGNN: an asymmetric-unit-based graph neural network for crystal property prediction*, J. Appl. Cryst. 58, 87–95 (2025), DOI [10.1107/S1600576724011336](https://doi.org/10.1107/S1600576724011336)。
- 官方页面：[Wiley/IUCr article page](https://onlinelibrary.wiley.com/doi/abs/10.1107/S1600576724011336)；官方 supporting information：[ei5123sup1.pdf](https://journals.iucr.org/j/issues/2025/01/00/ei5123/ei5123sup1.pdf)。出版社页面提示正文 full text 当前技术不可用；本次实际阅读了摘要/元数据和 supporting information §1 Graph embedding in ASUGNN（PDF pp. 3–6），没有把不可访问正文的方法细节当作已读。

### 已确认事实

摘要明确称这是 open-source asymmetric-unit GNN，按 conjugate subgroups coarse-grain sites，并结合 reciprocal-space powder XRD patterns；任务包括 formation energy、band gap、bulk modulus 和 metal/non-metal classification。Supporting information §1.1（p. 3）则明确规定 conventional unit cell 的**每个 atom 都是独立 node**，node embedding 含 fractional coordinates、lattice 和 space-group/global features；§1.2（p. 4）使用完整 `N×N` Cartesian distance adjacency。§1.3（pp. 4–6）另加 `N×N ASU adjacency matrix` 作为 mask/weight：等价 sites 的项赋值 1，其他 pair 用 reciprocal Cartesian distance（上限 1.1）。因此 ASUGNN 是 full conventional-cell atom graph + ASU symmetry mask/weight，而不是将 nodes 合并为 ASU/orbit representatives 的 graph。它是空间群约束/权重化 property GNN 的直接先例，但不能把摘要中的“asymmetric-unit-based”误读为实际输入节点已经缩减。

### 未确认与边界

Supporting information 没有说明：每层消息是否按 edge orbit multiplicity 加权、是否保留 operation/image labels、是否有 edge-orbit count、是否证明 full-graph forward/gradient 等价，或是否将 quotient edges 生成化。未找到作者公开 ASUGNN 源码仓库（Hugging Face `caobin/ASUGNN` 模型条目不能替代源码）。因此 ASUGNN 可支持“full-cell GNN 使用 ASU/equivalent-site mask/weight 已有”，不能支持“ASU node 合并或精确 edge-orbit weighted message passing 已有”。正文和源码细节保持未确认。

## 3. SymmCD：生成模型中的 ASU representative graph，但不是 edge-weight exact quotient

### 来源与实际阅读范围

- Daniel Levy et al., *SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models*, ICLR 2025，[arXiv:2502.03638](https://arxiv.org/abs/2502.03638)。本项目已完整阅读本地全文主文 §3–§4.3、Algorithms 1–2 及附录 site-symmetry diffusion/denoiser sections（PDF pp. 3–7, 17–20）。

### 表示与计算方式

SymmCD 以每个 crystallographic orbit 的 representative 建模，表示为 `(G,k,X',S,A')`；它明确利用 conventional unit cell 相比 asymmetric unit 的 orbit 数量压缩（MP-20 平均 `M=4.7` vs `N=18.9`），并在 representative nodes 上运行 fully connected message-passing GNN。site symmetry `S`、元素、代表坐标和 lattice features 进入节点/图条件，最后复制 representatives 得到 full crystal。

这已经是 symmetry-reduced representative graph 用于生成 diffusion 的直接先例，但其 fully connected graph 是在 representatives 上重新构造的；论文没有把 full-cell repeated edges 按 space-group edge orbit 汇总成带权 edge，也没有逐层的 exact forward/gradient equivalence 证明。它与当前组件的重合在“节点压缩/ASU diffusion”，差异在“edge-orbit weighted radial aggregation”。

## 综合判断

已有工作的直接覆盖可总结为：

| 先例 | 输入结构压缩 | 每层 edge-orbit 权重等价 | edge orbit 作为生成 token |
|---|---|---|---|
| Ruff et al. 2024 | 明确：ASU graph、multi-edges、multiplicity-aware readout | 官方 `to_asymmetric_unit_graph` 逐条复制 incoming full-graph edges 到 ASU MultiDiGraph，并保留每条 edge 的 `symmop`；无 edge multiplicity/count 合并 | 否 |
| ASUGNN 2025 | full conventional-cell atom nodes + `N×N` ASU mask/weight；不是已核实的 ASU-node merge | supporting information 未给出 edge-orbit count/operation labels/等价证明；正文与源码未核实 | 否 |
| SymmCD 2025 | 明确：orbit representatives、fully connected representative GNN | 未见 full-edge orbit weighted aggregation 或等价证明 | 否 |

因此不能声称“首次用 ASU/symmetry-reduced GNN 节点降低计算”。当前组件最窄、可检验的差异是：在已有 ASU/multi-edge 思路上，把可合并的完整 space-group edge orbits 改写为单边/计数权重形式，进入径向 message-passing/vector-sum，并报告与保留 multi-edge full graph 相同的前向/梯度（这是实现正确性的验证，不是单独新颖性）和 GPU memory/throughput。edge orbit 仍是**计算压缩机制**；现有三项均没有把它作为可生成的离散 graph token，因此不要把组件 benchmark 外推成完整 diffusion 或 edge-token generation 新颖性。

本审计未训练完整 diffusion，也未运行模型；ASUGNN 正文与源码的未公开细节保持未确认，以上 ASUGNN 图结论仅依据其官方 supporting information。
