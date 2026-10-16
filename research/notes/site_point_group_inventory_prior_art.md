# Site-point-group inventory `k_{e,H}`：已有工作重合审计

## 审计对象

候选表示给定空间群 `G`，把元素 `e` 所占的每个 Wyckoff orbit 按内禀 site point group `H`（32 类晶体点群）合并为计数 inventory `k_{e,H}`；随后以 WP identity residual 恢复具体位置/取向。候选设想是让 `G,H` 决定 multiplicity 与自由度预算，inventory 决定每元素的轨道数/几何自由度预算，再用于 conditional 非自回归去噪。

这里的关键边界是：`k_{e,H}` 是从完整的 `(element, WP identity/orbit)` 多重集投影得到的统计量。仓库中的 1731-WP 精确仿射 stabilizer audit 已验证：在固定 `G` 和 32 类内禀 `H` 后，`m=|G/T|/|H|`（相应 conventional/primitive centering 版本）以及由 `H` 的不动子空间得到的 DoF 是确定的；不能把同一 `G,H` 下仍有多个 WP identity 错推成 multiplicity/DoF 不确定。inventory 仍然不能恢复具体的 WP letter、方向/取向、offset、自由参数表达式和 enumeration。

### 仓库内精确审计证据

`experiments/wyckoff_tokenizer_search/derive_site_point_groups.py` 对每个 WP 从精确仿射操作求 stabilizer：检查 `|stabilizer| × multiplicity = |G|`，以 stabilizer 的线性部分求不动子空间维数，并用 `spglib.get_pointgroup` 归入 32 类；脚本还断言同一 `G` 内同一 point-group type 的 `multiplicity`、primitive multiplicity 和 `dof` 完全一致。其输出 `experiments/wyckoff_tokenizer_search/site_point_groups/summary.json` 报告 `wyckoff_positions: 1731`、`intrinsic_point_group_types: 32`、`exact_affine_stabilizer_order_and_dof_checks: PASS`。

这不排除 identity ambiguity，恰好说明它的性质不同。例如 `site_point_groups/metadata.json` 中 SG 225 的 point-group type 7 有 WP letters `i,h,g`，三者共享 multiplicity 48、primitive multiplicity 12、DoF 1；type 4 的 `k,j` 共享 multiplicity 96、primitive multiplicity 24、DoF 2。WyFormer 附录 S 所说的同 site symmetry collision，应按这些例子理解为多个 letters/坐标形式共享同一 `m/DoF`，而不是 `m/DoF` 被 identity 不确定性改变。

## 1. Wyckoff Transformer（WyFormer，Kazeev et al.）

### 来源与实际阅读范围

- 本地全文：`/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/work/papers/wyformer-2503.02407.txt`；`source_manifest.json` 记录 arXiv v4、32 页、文本页标记完整。
- 官方来源：[arXiv:2503.02407](https://arxiv.org/abs/2503.02407)；可交叉核对 [OpenReview PDF](https://openreview.net/pdf?id=ursX3k1rTO) 和 [官方代码仓库](https://github.com/SymmetryAdvantage/WyckoffTransformer)。作者为 Nikita Kazeev、Wei Nong、Ignat Romanov、Ruiming Zhu、Andrey Ustyuzhanin、Shuya Yamazaki、Kedar Hippalgaonkar。
- 阅读了 §1.1–§1.3（本地文本 PDF pp. 1–3）、§2.1 Tokenization 和 §2.1.1 Spherical Harmonics（pp. 4–5）、§2.2–§2.4（pp. 4–6），以及附录/补充部分 N（letters vs site symmetries）、O（harmonics ablation）、P（harmonic sampling）、R（token analysis）和 S（full-space-group enumeration，pp. 26–32）。

### 精确重合

WyFormer 明确指出 oriented site-symmetry 符号是跨空间群可复用的分类，而 Wyckoff letters 依赖空间群（§1.3，pp. 2–3）。其结构表示是无序 token 集；每个 orbit 用 `(element, oriented site symmetry, enumeration)` 三元组表示（§2.1，p. 4）。同一个 `G` 和 oriented symbol 下可能有多个 WP，故在 site symmetry 内使用 enumeration 区分它们；论文用 `225` 的 `4a=(m-3m,0)`、`4b=(m-3m,1)`、`8c=(-43m,0)` 说明了这一点（§2.1，p. 4）。模型学习的概率分解也是先预测 element，再 oriented site symmetry，最后 enumeration（§2.1，p. 4；§2.3，p. 5）。

这意味着候选 inventory 可以由 WyFormer token 多重集按 `H` 聚合得到，但它不是 WyFormer 的现有 token：论文保留了每个 orbit 的 enumeration/WP identity，并将 element 与该 orbit 的 site symmetry、enumeration 绑定。其 site-symmetry-only 表示在附录 O 的消融中甚至明显弱于 `site symmetry + enumeration`；附录 S 说明同一 site symmetry 的不同 WPs 因坐标不同而发生 collision，必须通过组内编号（或 letters）消歧（pp. 28、32）。

### 与候选的边界

- **已有的部分：** 以 Wyckoff site-symmetry 符号作为跨空间群可复用的共享类别；基于 site-symmetry 条件预测；把多个同符号 orbit 视为共享语义的对象。必须区分：WyFormer token 是带方向的 oriented site-symmetry/Hermann–Mauguin-style 符号，并非候选所用 `spglib.get_pointgroup` 归并后的 32 类 intrinsic point-group type。
- **未见于该文的部分：** 显式构造计数张量 `k_{e,H}`；先预测 inventory，再以 residual 恢复 WP identity；将 inventory 解释为每元素轨道数和几何 DoF 预算；用于非自回归去噪。
- **重要限制：** WyFormer 的 `H` 不是完整的 orbit identity。其 enumeration 用来区分同 `G,H` 下具有不同坐标/方向/offset 的多个 WP；它不意味着这些 WPs 的 multiplicity 或 DoF 不同。仓库精确 audit（`experiments/wyckoff_tokenizer_search/derive_site_point_groups.py`，`site_point_groups/summary.json`）对全部 1731 WPs、32 个 point-group types 检查了 stabilizer order、`m` 和不动子空间 DoF，结果为 `exact_affine_stabilizer_order_and_dof_checks: PASS`。WyFormer 附录 S 的 collision 例子应解读为 identity/坐标 collision，而非 `m/DoF` collision。

### 结论

候选的“按 32 类 intrinsic site point group 合并”可描述为对 WyFormer oriented site-symmetry 表示的进一步确定性 coarsening/reparameterization：附录 M 的有效 token 列表包含 `.m.`, `..m`, `m..`, `.2.`, `2..` 等方向变体；候选把这些方向符号归并到同一个 intrinsic type，再由 WP identity residual 恢复方向、offset 和 enumeration。WyFormer 没有报告 `inventory → identity residual` 的生成因子化。不能声称首次使用 site-symmetry；可以谨慎主张一种比 WyFormer oriented-symbol token 更粗、元素聚合、层次化的预测分解，前提是实验证明它比直接 `(oriented H, enumeration)` 预测在 exact legality、OOD 空间群和去噪效率上有益。

## 2. SymmCD（Levy et al.）

### 来源与实际阅读范围

- 本地全文：`/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/work/papers/symmcd-2502.03638.txt`；`source_manifest.json` 记录 arXiv v3、24 页、文本页标记完整。
- 官方来源：[arXiv:2502.03638](https://arxiv.org/abs/2502.03638)；发表版本/代码入口可由论文链接到 [SymmCD GitHub](https://github.com/sibasmarak/SymmCD/)；论文 PDF 版本为 ICLR 2025。
- 阅读了 §1–§3（PDF pp. 1–4，尤其 32 类点群、orbit/stabilizer、Wyckoff position 定义）、§4.1 Representation（pp. 4–5）、§4.2 Symmetry Representation（pp. 5–6）、§4.3 Diffusion Model 与 Algorithms 1–2（pp. 6–7），以及附录中 site-symmetry diffusion/denoiser 细节（PDF pp. 17–18）和 MPTS-52/atom-count 分析（pp. 19–20）。

### 精确重合

SymmCD §3（PDF pp. 3–4）明确定义 3D 只有 32 类 crystallographic point-group types；site symmetry `S_x` 是固定点的子群，orbit 大小满足 `|P|/|S_x|`。因此 `G` 与 `H=S_x` 确实对 orbit multiplicity 提供强约束，但论文随后把每个 crystallographic orbit 作为一个独立 representative，并表示为 `(G,k,X',S,A')`（§4.1，p. 4）。`S` 是每个 representative 的 site-symmetry group，`X'` 是具体代表坐标，`A'` 是元素。

§4.2（PDF pp. 5–6）将 site symmetry 编为按 15 个可能轴、每轴 13 个操作的一组二值矩阵；论文强调该编码对不同空间群共享信息，并且是 injective，可用于复制完整 orbit。采样算法（Algorithm 2，p. 7）先对每个 representative 去噪 `S'`、坐标和 element，随后将 `S'` 投影到合法 point group，再把坐标投影到对应 Wyckoff position，最后复制成 full crystal。附录中的 site-symmetry diffusion 把矩阵视作 15 个 categorical variables；它仍然是**每个 orbit 一个 site-symmetry变量**，不是按元素聚合的计数 inventory。

### 与候选的边界

- **已有的部分：** site-symmetry 作为共享离散表示；`G` 与 site symmetry 共同约束 orbit multiplicity；以 orbit representative 为生成单位，并在最后恢复 full orbit；对 site symmetry 做离散扩散、对坐标做连续扩散。SymmCD 的 site-symmetry matrix 记录 oriented axis/operation 信息（15×13），也不是简单的 32 类 intrinsic token。
- **未见于该文的部分：** `k_{e,H}` 这种按元素聚合的 multiset/count；先生成 count inventory 再预测 WP identity residual；用 inventory 作为轨道数量/DoF 预算；非自回归地一次预测元素级 inventory。
- **关键差别：** SymmCD 的 `S` 仍然保留每个 representative 的实例身份、坐标和对应 atom type。`G` 与 `H=S_x` 已确定 orbit multiplicity（orbit-stabilizer）及 site DoF；若仅保留 `k_{e,H}`，丢失的是同 `G,H` 下不同 WP 的方向/固定坐标/offset/自由参数形式和 enumeration，必须由 residual 或条件表补回。

### 结论

候选不是 SymmCD 已有表示的原样复述，但其底层语义（32 类 site point group、`G` 决定合法操作、按 orbit 复制恢复 full crystal）直接继承了 SymmCD/标准 Wyckoff 理论。可辩护的是一个**更粗粒度、元素聚合、层次化的条件变量**，而不是新的 site-symmetry 表示或新的对称性保证机制。

## 3. Wren（Goodall et al.）

### 来源与实际阅读范围

- 官方全文：[Rapid discovery of stable materials by coordinate-free coarse graining](https://pmc.ncbi.nlm.nih.gov/articles/PMC9328671/)（Sci. Adv. 8, eabn4117, 2022）。这是作者公开的 PMC 全文；阅读了引言中的 Wyckoff representation 定义与维数比较（HTML lines 137–152）、Wren architecture/embedding（lines 216–236）、equivalent-representation invariance（lines 237–246）以及 Wyckoff-position evaluation（lines 247–250）。

### 精确重合

Wren 将结构的坐标空间粗粒化为 Wyckoff representation：每个 WP 仅保留 Wyckoff letter 与其元素，得到大小为 `2M` 的离散空间（PMC lines 139–142）。模型是对 WP 多重集做 message passing，再 permutation-invariant pooling（lines 148–152）。其 WP embedding 由 crystal system、Bravais centering、以及该 WP 所有 equivalent sites 的 algebraic/offset multi-hot 编码组成（lines 216–236）；论文还明确指出 230 个空间群共有 1731 个 WPs，并通过 site 坐标项暴露它们之间的关联。等价 origin/setting 通过 on-the-fly augmentation 和平均池化处理（lines 237–246）。

这与 `k_{e,H}` 有两层关系：一方面，Wren 已经证明“对 WP 多重集做共享嵌入和置换不变聚合”是成熟做法；另一方面，它的输入比 `k_{e,H}` 更细，保留每个 WP 的 letter/坐标 algebraic terms 和元素。按 `H` 聚合可视为把 Wren 的 WP multiset 投影到一个较粗的 site-point-group inventory，但 Wren 本身没有报告该投影，也没有 inventory-to-identity residual 解码。

### 结论

不能把“共享嵌入 + permutation-invariant set aggregation”作为候选的独立新颖性；Wren 已有直接方法先例。候选若有贡献，应落在生成任务的**因子化与预测单位**（元素级 inventory 的 conditional non-AR denoising、随后 identity residual）以及由此带来的 exact composition/DoF legality 或效率收益，而不是 Wren 式 set encoder 本身。

## 综合判断：是否只是已有 site-symmetry 表示重组？

严格说，候选的 `k_{e,H}` 是一个新的统计投影/生成分解，而不是已有论文中明确出现的 token：在所读 WyFormer、SymmCD 和 Wren 全文范围内，没有找到显式的元素×site-point-group count inventory、inventory-first residual decoder 或该变量用于 conditional non-AR diffusion 的描述。

但它也不是新的晶体学对象：

1. 每个 orbit 的 site symmetry、由 stabilizer order 决定的 multiplicity、由不动子空间决定的 DoF、固定坐标及其复制规则来自标准 Wyckoff 理论；SymmCD 将这些关系写成 `G,S,X'` 并逐 orbit 扩散。
2. WyFormer 已经把 oriented site-symmetry 符号作为跨空间群共享类别，并用 enumeration 保留同 `G,H` WPs 的 identity/坐标差异；附录 M 的 token 列表包含多种方向变体，附录 S 说明只用该符号会发生 identity collision，并不说明 `m/DoF` 不同。候选 `spglib` 32 类 `H` 比这些 oriented symbols 更粗，方向信息转移到 residual。
3. Wren 已经使用共享 WP 特征和置换不变 multiset 模型；候选的共享 embedding/集合聚合不能单独构成新颖性。

因此最稳妥的表述是：**“一种把既有 per-orbit Wyckoff/site-symmetry 表示显式重参数化为元素级 site-point-group inventory，并以 WP identity residual 恢复细节的条件生成分解。”** “首次使用 site symmetry”“首次按 multiplicity/DoF 进行合法性建模”“首次非自回归 crystal symmetry generation”都不可声称。

### 可辩护的有限差异

- `k_{e,H}` 作为明确的元素级中间变量，而不是每 orbit 一个 `H` 标签；它可使元素总量、orbit 数量和某些 DoF 预算成为可检查的条件。
- inventory-first / residual-second 的因子化若在同一 backbone、参数量和 legality constraints 下做消融，可能是生成建模贡献；需要对比 WyFormer 的 `(H, enumeration)` cascade、SymmCD 的 per-orbit joint diffusion、以及 Wren-like set baseline。
- 若把 residual 定义成在固定 `G,H` 下的合法 WP identity/orientation 选择，并证明 exact composition、orbit collision 和 OOD-G 泛化收益，可主张一种任务/解码器设计；这仍是架构与训练贡献，不是 site-symmetry 数学定义的贡献。

### 不可声称的新颖性

- 32 类 crystallographic point groups、site-symmetry groups、orbit-stabilizer multiplicity 关系：已有标准晶体学与 SymmCD §3。
- 以 oriented site-symmetry 符号跨空间群共享：WyFormer §1.3/§2.1 及附录 M 已明确提出；候选的 32 类 intrinsic `H` 是对这些带方向符号的进一步归并，不应描述为与 WyFormer 相同的 32 类 token。
- 以 WP/等价 sites 的共享嵌入和置换不变集合模型：Wren 已明确实现。
- 由 `G` 和 site symmetry 复制 orbit、用确定的 multiplicity/DoF 以及固定坐标保证合法性：SymmCD §4.1–§4.3 已明确实现；WyFormer 附录 S 也说明同 `G,H` 的 WP identity/坐标信息不能省略。
- “按 multiplicity/元素计数生成合法 WP 分配”作为一般目标：至少是 Wyckoff 表示和上述模型的直接推论；不能把 inventory 的计数形式包装为首次 composition-aware symmetry assignment。

### 仍需实验证据才能成立的主张

应报告同等参数/训练预算下：`inventory-first residual` 对直接 `(H, identity)` 或 per-orbit joint diffusion 的 exact composition rate、合法 WP rate、site/orientation recovery、OOD space-group/element split、采样步数/吞吐和最终结构稳定性的差异。若没有这类对照，最强的合理结论只能是“对已知 site-symmetry/Wyckoff变量的可逆性受限的粗粒度重参数化”，而非新的 tokenizer 或新的 symmetry representation。

本审计未训练、未运行实验，且“在所读范围内未找到”不等于全球文献中不存在。
