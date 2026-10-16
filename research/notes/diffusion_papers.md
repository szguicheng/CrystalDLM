# DynaCrys 与 MaskGXT：全文核对笔记

阅读范围：本笔记只覆盖两篇指定论文的本地 PDF、逐页 TXT 与 preflight，以及 arXiv 当前条目页面。DynaCrys 使用 `dynacrys-2608.07401.pdf/.txt`（PDF 26 页，preflight PASS，SHA-256 `76160a...c8849c70f`）；MaskGXT 使用 `maskgxt-2606.22866.pdf/.txt`（PDF 20 页，preflight PASS，SHA-256 `b1164a...a3c7e7`）。逐页 TXT 中的页眉标记与 PDF 页码一致；本文覆盖正文、附录 A–K（DynaCrys 中与表示/采样/几何/评估相关者）以及 MaskGXT 附录 A–D。arXiv 页面核对结果均为 v1：DynaCrys 2026-08-07 提交，MaskGXT 2026-06-22 提交；未发现更新版本。

## 快速对比

| 项目 | DynaCrys | MaskGXT |
|---|---|---|
| 生成范式 | 两阶段：离散/连续符号扩散 + 几何扩散；空间群在反向轨迹中可变 | 原文定位为 MaskGIT 风格 masked generative model；非 AR，掩码生成/迭代并行解码 |
| 主要输入条件 | 论文主实验为 unconditional de novo generation（MP-20 训练分区）；没有化学式、性质或给定 SG 的主实验 | composition-conditioned CSP：给定元素集合/原子类型 A；可选固定 SG 仅是采样接口，不是主报告条件 |
| 表示 | protostructure `(G,K,{(Z_i,r_i)})`：SG、Wyckoff-row 占位及元素；不含晶格/自由坐标；随后生成晶格和每个占据 orbit 的 anchor | `1+6+5N` token：1 SG、6 个晶格标量、每位点 3 坐标+1 Wyckoff+1 atom type；A 作为侧信息 |
| token/离散化 | 符号阶段没有 NLP tokenizer/合并词表：SG 与 row 由 128-d 共享对称 codebook 表示；元素为 categorical；row 通道为连续噪声向量；几何阶段为连续 diffusion | 坐标/晶格标量量化为 K=64 bins；坐标在 `[0,1)` 环上取模，圆形 ordinal smoothing；另有 sub-bin offset 回归。没有 BPE/学习合并词表 |
| 生成结果 | 最终是完整 conventional-cell 结构（晶格 + 自由 Wyckoff 坐标经对称操作展开）；中间 protostructure 不含这些几何量 | 返回 `(L,F)` 完整结构，并同时输出 SG/Wyckoff token；训练中 symmetry 一致性是软学习的，未作硬一致性约束 |
| 对称/置换 | SG–Wyckoff 关系显式约束；SG diffusion kernel 沿 maximal t/k group–subgroup 图；无 slot positional encoding，Hungarian matching 处理无序 orbit slots | continuous origin shift + Wyckoff relabel；按 symmetry orbit canonical order，做 intra/inter-orbit permutation；SG stratified sampling 覆盖多晶型 |

## DynaCrys（arXiv:2608.07401v1）

### 表示、离散化与“tokenizer”边界

* 第 4.1 节（PDF pp.4–5，Eq.1–2）定义 protostructure `P=(G,K,{(Z_i,r_i)})`：G 是空间群，K 是 orbit occupations 数，`r_i` 是属于当前 G 的 Wyckoff row，`Z_i` 是元素；occupation 是无序 multiset，同一 row 可重复。P **不包含**晶格或自由坐标。固定宽度 `M=20` slot，inactive slot 用 PAD。
* 这不是文本 tokenizer。第 4.1/附录 C（pp.4–5、15–16，Eq.25–27）说共享 symmetry codebook 由空间群操作 `(R|τ)`、row 的 orbit/stabilizer 操作与元数据构造，宽度 `d=128`；先预训练空间群/row 的结构标签，再冻结并被符号与几何阶段共用。它是学习的连续 embedding/codebook，但没有 BPE、词片合并或“学习合并词表”。有效词汇是 168 个训练中出现的 SG、1,731 个 `(G,r)` 对（附录 A.1，p.13），每个 SG 的 row 表 padding 到 27。
* 符号扩散中 G 是 categorical state；K、active mask、元素是离散头；row 用依赖当前 G 的合法 row 分类头，但 forward row channel 对 codebook 向量加连续噪声（第 4.2 节，pp.5–6；附录 B/D）。因此不能把整个模型称为纯离散 tokenizer 模型：几何阶段明确是连续 lattice/coordinate diffusion（第 4.4 节 p.6，附录 F pp.20–21）。

### AR / 非 AR 与采样过程

* 它不是自回归序列模型。符号 denoiser 是 Transformer **set** encoder：一个全局 token `(G_t,K_t,t)` 加 M 个 orbit-slot token；没有 slot positional encoding，输出对 slot permutation equivariant（第 4.2 节 p.5；附录 D pp.16–18）。
* 每个反向时间步同时预测 `p(G_0|x_t)`, `p(K_0|x_t)`, 各 slot active/row/element 的 factorized clean-state 分布。随后 legality-constrained decoder 按 Eq.7–8 采样，拒绝违反 SG 合法 row、row-capacity、atom-count 的 skeleton；接受后再采样元素。然后按同一 forward marginal kernel re-noise 到 `x_{t-1}`（第 4.3 节 pp.5–6；附录 E pp.18–19）。这属于多步非 AR 的离散/连续扩散采样，不是逐 token AR。
* 清洁符号状态在 `t=0` 去掉 inactive slots、忽略 slot 顺序，形成 protostructure。第二阶段以 P 为条件，连续 diffusion 同时生成受 crystal-family 子空间投影的晶格和每个 orbit anchor 的自由分数坐标；每步用 Wyckoff affine operations 展开成完整 orbit，故中间 geometry 也服从 decoded G（第 4.4 节 p.6；附录 F pp.20–21）。

### 条件任务、输出与实验边界

* 主实验明确是 MP-20 上的 **unconditional crystal generation**（第 5.1 节 p.7）：从训练分区学习并采样约 10,000 个候选，再经过 CHGNet 和 MACE-MP-0 两套 relaxation/evaluation。这里的 composition、SG 都是生成状态的一部分，不是用户输入条件。论文没有在实验中报告“给定化学式/性质/SG/部分结构”的 conditional benchmark。
* 因此：化学式/元素——作为 symbol stage 生成的 `Z_i` 及其 composition 约束；性质——没有性质条件输入，稳定性只在生成后以 `e_hull` 等 proxy 筛选；SG——由符号 diffusion 动态提出、可在每个 reverse step 修改，而非外部给定；部分结构——未实验验证。正文所说“conditional on P”的含义仅是第二阶段以**模型自己生成的 protostructure**为条件，并非用户给定部分结构（第 4.4 节 p.6）。
* 输出是 full structure；protostructure 是中间接口。第 6 节 p.10 明确局限：稳定性依赖计算 proxy 和有限已知结构 snapshot；生成的是零温、无限、完全有序晶体，不覆盖缺陷、无序、有限温度，也不等价于实验可合成性。

### 对称性与置换

* SG corruption kernel 在 230 SG 的 maximal translationengleiche (t) / klassengleiche (k) group–subgroup 无向图上扩散，并混入 training SG marginal reset（第 4.2 节 p.5，Eq.3；图 2 p.4）。这是动态修订 SG 的关键区别，而不是先固定 G。
* occupation 集合无序；Hungarian assignment 将 M 个预测 slot 与 K 个 clean occupations/PAD 匹配（第 4.2 节 p.5，Eq.5–6）。geometry 中 lattice 投影到 G 的 crystal-family 子空间，anchor 只在由 P 定义的自由度中更新，再通过 Wyckoff operations 展开（第 4.4 节 p.6；附录 F）。

### 结果与局限（仅作为对比证据）

* Table 1（PDF p.8）报告两种 relaxation engine；canonical DynaCrys 的 CHGNet `SUN@0=9.39%`, `SSUN@0=9.31%`，MACE-MP-0 分别 `9.55%`,`9.44%`，但这些是共同 protocol 下的计算筛选指标，并非实验材料发现。
* 论文自己指出（第 6 节 p.10）低 `e_hull` 不保证可合成；训练/评估 snapshot、零温完美晶体假设是外推限制。另一个结构性限制是合法 decoder 的接受率/容量规则与训练中出现的 168 个 SG 词汇绑定；它不声称覆盖所有 230 SG（附录 A.1 p.13）。

## MaskGXT（arXiv:2606.22866v1）

### 表示、离散化与“tokenizer”边界

* 第 4.1 节（PDF pp.5–7，图 3、Eq.1）把一个晶体编码为 `1+6+5N`：1 个 SG token；6 个 Niggli-like lattice scalar（a,b,c,α,β,γ）的 bin；每个 site 的 x/y/z coordinate bin、Wyckoff token、atom-type token。输入 A（元素/atom types）是条件侧信息，所以生成目标为 SG、晶格、坐标和 Wyckoff。
* 坐标 `f∈[0,1)` 通过 `q(f)=floor((f mod 1)K)` 量化（Eq.1）；Table 6（p.17）给 K=64。坐标 bins 视为圆，`0` 与 `K−1` 相邻；coordinate/lattice 使用 ordinal soft target，坐标用 circular distance（第 4.3 节 p.7，Eq.2–3）。另有 coordinate/lattice sub-bin offset regression，生成时将 bin center 加 bounded offset（第 4.3 节 p.7，Eq.4 附近）。
* 这是 domain-specific 离散编码，不是 NLP tokenizer；全文没有 BPE、WordPiece 或“学习合并词表”。SG/Wyckoff/元素为类别 embedding，lattice/coordinate 为固定 K-bin vocabulary，offset 是连续回归头。论文的“discrete token model”不能解读为学习了一个可合并的晶体词表。

### 严格核查：conditional non-AR、masked model 与 diffusion 的区别

* MaskGXT **确实是 conditional non-autoregressive masked generative model**。第 4 节开头（p.5）明确它把晶体表示成 partially masked sequence，学习重建 missing tokens；第 4.1 节（pp.5–7）和图 3(c)显示从全 mask、仅保留 A 开始，所有位置并行预测，再按 confidence 固定最高置信 token；尚未固定的低置信位置继续保持 mask，在下一步重新预测，重复 T 次。论文没有说把已经固定的 token 再次 mask。Transformer 是 bidirectional，不能按因果 AR 解释。
* 训练是 absorbing-state masking：每个 token 独立以随机 `t∼U(0,1)` 被 mask，masked cross-entropy 监督（第 4.3 节 p.7，Eq.3；算法 1 附录 A p.16）。采样的基础算法允许从 posterior 抽样；论文报告的 CSP 主结果用 deterministic confidence-ranked argmax greedy decoding（第 4.5 节 pp.8–9、算法 2 p.16）。Table 6 的 T=150 是迭代补全步数。
* 原文将 MaskGXT 定位为 MaskGIT 式 masked generative model，并在相关工作/比较表中与 diffusion 和 flow 分列（第 2 节 p.4；第 4 节 pp.5–9）。它包含随机 masking 与多步重建，但不能仅凭“没有 DDPM 式时间索引噪声核/噪声预测/reverse posterior”就排除与 masked diffusion 的数学联系；对比表应保留原文范式标签，而不要作绝对的“不是 diffusion”断言。本文最稳妥的标注是 **masked generative / MaskGIT-style iterative parallel decoding**。

### 条件任务、输出与实验边界

* 实验任务是 composition-conditioned CSP（第 5.3 节 p.10）：给定测试 composition，生成晶体；标准 MP-20/MPTS-52 one-to-one protocol 每个 reference 生成一个结构；METRe 用 space-group-stratified multi-candidate sampling。化学式在实验中体现为 A 的元素集合/每-site atom-type side information，而非生成 atom types。
* optional fixed SG 只出现在算法 2 的采样接口（附录 A p.16：“if a space group is given, fix its token”）；主实验不是给定 SG，而是从全 mask 仅基于 A 先算 `p(g|A)`，选 top-S 高后验 SG 分支，再将分支 SG 固定解码（算法 3、附录 A p.16；第 4.5 节 pp.8–9）。因此“支持给定 SG”是接口能力，不应写成论文已评估的独立 conditional benchmark。
* 性质条件、部分结构条件均未在本文实验中实现或评估。第 6 节 p.12 的 Future directions 只说同一 agent/tokenization **可自然扩展**到 de novo generation 和 structure-conditioned atom-type generation；这是未来设想，不能倒写成当前实验能力。文中 de novo 也只是未来方向，不是当前 MaskGXT 主结果。
* 输出是完整 `(L,F)`：6 个晶格标量和 N×3 fractional coordinates，经 offset refinement 反量化；同时预测 SG 与 Wyckoff token。图 3(a,c)、第 4.1 节（pp.5–7）清楚说明目标覆盖 full crystal。SG/Wyckoff 与 lattice/coordinates 独立解码，结构一致性由训练学习而非硬约束（图 3 后、p.7）。

### 对称性、置换与局限

* Euclidean-normalizer origin shift：训练时连续随机 `r∈[0,1)^3` 平移 fractional coordinates modulo 1，并用 normalizer coset 重标 Wyckoff letter；这是等价描述增强，不改变 SG（第 4.2 节 pp.7–8；算法 1 p.16）。
* canonical ordering 按 MCFlow 的 symmetry orbit（先 electronegativity，再 Wyckoff letter），再做 intra/inter-orbit permutation，仅在对称等价描述内打乱，而非全部 N! permutations（第 4.2 节 p.8）。
* SG stratified sampling 读取全 mask 上的一次 `p(g|A)`，选 top-S SG；每个分支固定其 SG 解码，以减少同一 composition 的重复并覆盖 polymorphs（第 4.5 节 pp.8–9；算法 3 p.16）。若每个 SG 后验都低于 0.02 mass，Table 6/p.17 的 fallback 放弃固定 SG，改为 i.i.d. draw。
* 关键局限是 SG/Wyckoff、坐标和 lattice token 独立预测，论文明确说一致性通过训练鼓励而不是硬约束；因此不能宣称每个 token 组合都必然是合法晶体（第 4.1 节 p.7）。此外，METRe 的标准预算等于测试集 reference 数，不可部署时获得；论文另给 uniform S=2–5 分析（第 5.5 节 pp.10–11、附录 C pp.18–19）。
* Table 1（p.10）MaskGXT MP-20 MR `73.79/67.06%`（unfiltered/filtered），MPTS-52 `36.75/33.34%`；Table 2（p.11）MP-20 polymorph split METRe `79.06%`。这些是 benchmark matching，不是实验合成验证。

## 最关键的差异

1. DynaCrys 把 SG、Wyckoff occupations 和元素放入随时间扩散且可修订的 symbolic state，再用连续 symmetry-constrained diffusion 生成几何；MaskGXT 把 SG/Wyckoff、晶格、坐标全部离散化，在给定 composition A 后以双向 Transformer 做 masked 并行补全。
2. DynaCrys 的 protostructure 是不含晶格/坐标的中间结构，最终 full structure 由第二阶段几何扩散产生；MaskGXT 的 token 序列直接覆盖 full crystal（但 SG/Wyckoff 与几何的一致性不是硬约束）。
3. 两者都不是自回归；DynaCrys 明确采用符号 + 几何两阶段 diffusion。MaskGXT 原文定位为“conditional masked non-AR generative model”，使用 masked generation/迭代并行解码；与 masked diffusion 的数学关系不能仅凭方法表面形式排除。
4. 条件性不可混写：DynaCrys 主实验是 unconditional MP-20 generation；MaskGXT 主实验是 composition-conditioned CSP。两者都没有本文实验中的性质条件或部分结构 completion；MaskGXT 给定 SG 是可选采样接口，DynaCrys 的 P-conditioning 是自生成 protostructure 到 geometry stage 的内部条件。

## 不能确认 / 明确未声称

* 两篇论文都没有实验性合成、真实实验性质条件或 defect/disorder/finite-temperature 评估；计算稳定性/匹配分数不能替代这些证据。
* DynaCrys 正文没有给出化学式、性质、部分结构 conditional benchmark；MaskGXT 的未来扩展（de novo、structure-conditioned atom-type generation）不应写进当前能力表。
* 本次核对依赖指定本地 v1 PDF/TXT 与 arXiv v1 条目；未下载代码、未运行训练/采样，也未将博客或摘要之外的材料作为证据。
