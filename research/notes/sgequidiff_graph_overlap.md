# SGEquiDiff 与 ASU / space-group graph 表示的查重

## 来源与实际阅读范围

- 原始论文：[Space Group Equivariant Crystal Diffusion](https://arxiv.org/abs/2505.10994)，NeurIPS 2025 proceedings：[官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/697b2f31f99fb79f8a0a16e923b2471d-Paper-Conference.pdf)。
- 官方实现：[rees-c/sgequidiff](https://github.com/rees-c/sgequidiff)。
- 实际阅读：NeurIPS 官方 PDF 全文 34 页。重点核对主文 §3 preliminaries、§4.1 representation、§4.4 space-group equivariant diffusion（PDF pp. 3–7），附录 A.1–A.4（PDF pp. 19–23，含 uniform Wyckoff priors、Wyckoff-Element Transformer、GNN architecture、space-group/Wyckoff features）和 A.5 ablations。官方代码 README 也已核对预处理、生成路径和 ASU objects 说明；未运行代码。

## 1. SGEquiDiff 确实有 ASU/Wyckoff-orbit 节点级表示

### ASU 与轨道展开

主文 §3（PDF p. 3）定义 conventional unit cell 和 asymmetric unit `Π`；表示只保留 ASU 中 `n≤N` 个 symmetrically inequivalent atoms，并为每个 atom 记录 `W={(w_1,…,w_n)}` Wyckoff positions、元素 `A`、坐标 `X`。公式直接给出完整结构的恢复：对每个 ASU atom，应用 `g_{ij}∈G/w_i` 生成其全体 symmetry images（PDF pp. 3–4，Eq. 在 lines 193–205）。这比普通 SLICES translation quotient 更接近“orbit-node + group-action expansion”。

`§4.1 Crystal representation`（PDF pp. 4–5）进一步说明：他们用 SymPy/PyXtal 把每个 WP 与 exact ASU 相交，得到 0D points、1D line segments、2D convex polygon facets 或 3D ASU interior 的 Wyckoff shapes；晶体分解为 `M=(G,L,A,W,X)`，其中 `X` 是 ASU representatives 的 fractional coordinates，`W` 是对应 WP。官方代码 README 的预处理命令也明确把 CIF 转成 asymmetric unit objects。

### 与候选的直接重合

- 节点层面：SGEquiDiff 已将节点数从 conventional-cell atoms `N` 降为 ASU/orbit representatives `n`。
- 对称展开：完整结构由空间群作用 `G/w_i` 从 ASU representatives 重建，而不是只加整数 lattice translations。
- 几何约束：空间群 equivariant vector field 自动落在当前 Wyckoff position 的 tangent space，扩散不离开该 WP；Eq. (3) 对有限商群 `G/T_L` 做 group symmetrization。

因此，“只生成 ASU/Wyckoff-orbit nodes，再由完整 space-group action 展开”已经是 SGEquiDiff 的明确表示/生成机制，不能作为候选的独立新颖性。

## 2. 但 SGEquiDiff 没有候选所需的 bond-edge orbit / CIF symmetry edge token

### 论文 GNN 的实际图

附录 A.4.2（PDF pp. 21–22）明确写道：GNN 构造的是 conventional unit cell 中的 fully connected atom graph；每个 atom 与 conventional cell 中所有其他 atom 连接，边以 periodic boundary conditions 下的 minimum-length edge 表示，tie 时保留所有对应 edges。边特征是：

- `PlaneWaveEmbedding(x_j−x_i)`；
- RBF(minimum-image Cartesian distance `d_ij`)；
- lattice parameters `l`。

论文给出的 `d_ij=min_{n∈Z^3} ||Lx_i-L(x_j+n)||` 和 message-passing 方程（附录 A.4.2）是几何距离/相对坐标图，而不是一个 edge 代表完整空间群 action 下的 bond orbit。ASU 主要用于 canonicalization、坐标/labels 和最终 symmetry score；GNN 为了 score prediction 仍在 conventional full-cell atom graph 上工作。

### 没有找到的候选机制

在主文和附录 A.1–A.5 中未见：

- 以 ASU/Wyckoff representatives 为节点、而不是 full-cell atoms 为 GNN nodes；
- bond graph 的 edge-orbit quotient；
- 类似 CIF `_geom_bond_site_symmetry_1/2 = n_klm` 的 operation-id + cell-translation edge label；
- 对 edge orbit 使用 Seitz operation、screw/glide、旋转/反射 image 的离散 token；
- 从 quotient bond edges 显式展开完整 periodic bond graph 的算法；
- double-coset 或 stabilizer-pair edge classification。

SGEquiDiff 的 group action 出现在 coordinate score 的 SGWN / symmetrization（Eq. (3)、Theorem 1–2）中，而不是 bond graph 的离散 edge representation。

## 3. Space-group / Wyckoff features 也不是 edge action code

附录 A.4.3（PDF pp. 22–23）给 space group 62 features：6 centering、6 crystal families、32 point groups、chiral/centrosymmetric flags，以及 16 个 collision-disambiguation one-hot。每个 WP 的 231 features 包含 62 SG features、100 unique site-symmetry symbols、DoF（0–3）、17 multiplicity values 和 WP-shape midpoint/vertex Fourier features。

这些 features 用于 atom/WP embeddings；作者明确将其与 SymmCD 的 15 axes × 13 operations encoding 做 ablation 对比。它们不是每条 bond 的 operation+translation label，也不把完整空间群 action 编进 graph edge vocabulary。

## 4. 与候选的边界和结论

SGEquiDiff 是目前查到的比 WyckoffAUGen 更直接的先例：

1. **已覆盖：** ASU/Wyckoff-orbit representative nodes；由完整空间群 action 展开 full crystal；space-group-equivariant coordinate diffusion；Wyckoff tangent-space geometry；ASU 上的内存/轨迹压缩。
2. **未覆盖：** bond connectivity 作为显式生成对象；edge orbit quotient；CIF-like operation/image edge labels；由 quotient edges 展开周期 bond graph。

所以候选不能声称“首次使用 ASU/Wyckoff-orbit nodes 和完整 space-group action 展开”，因为 SGEquiDiff 已明确实现。若候选只增加 full-cell bond graph 的 quotient 还可能有窄差异，但必须把贡献限定为：**在 SGEquiDiff 式 ASU/orbit node 与 group-action geometry 之上，再把周期 bond connectivity 按完整空间群 action 约简为可逆 edge-orbit tokens，并由这些 tokens 恢复 full periodic bond graph。**

与 SGEquiDiff 的关键对照应是：

- SGEquiDiff：full-cell fully connected geometric graph + minimum-image distance/relative-coordinate edge features + group symmetrization of vector field；
- 候选：ASU/orbit-node graph + discrete bond edge orbit/action labels + deterministic graph expansion。

该差异仍需证明 token reduction、bond/geometry legality、edge-label invertibility、周期邻接恢复和生成质量收益。不能把 SGEquiDiff 的 space-group equivariant GNN 泛化为“已经有完整 symmetry-labeled quotient bond graph”。

本核查未训练、未运行模型。
