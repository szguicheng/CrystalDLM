# Wyckoff tokenizer novelty 审核 v2

范围限定：只审核基本晶体学概念的组合，不采用母相/畸变模式路线；不实现、不跑实验。本次核对使用 WyFormer 官方 GitHub 在 commit `e8e9add5df904284116af8a06c203555d908f290` 的源码/YAML/notebook、WyckoffDiff 论文正文与官方仓库，以及 arXiv:2602.17176（NextCrystal）当前官方正文。结论是“代码证据能证明什么”，不是首次性声明。

## 结论先行

WyFormer 目前确实有固定的组合式 WP 表示，但没有做“每元素完整占位分配 token”：

- `mp_20_CSP.yaml` 与 `mp_20_sg_multiplicity.yaml` 的 WP 序列字段都是独立 categorical fields：`elements`、`site_symmetries`、`sites_enumeration`；`multiplicity` 是由 `(spacegroup_number, site_symmetries, sites_enumeration)` 查表得到的 engineered feature，不是把重数拆成一串 token，也不是新的候选 token 词表。
- composition 是另一个 sequence-level counter：元素 token 与 count tensor 分开保存；这不能表达“某元素占据哪些完整 WP”的联合占位 token。
- `EnumeratingTokeniser.from_token_set` 从数据集中出现的每个字段值建立固定词表；没有 BPE merge、候选组合枚举器、组合 candidate embedding 或按元素完整 occupancy 的合法候选词表。
- `csp_v0.yaml` 的 `concat_token_presence: true` 是把输入中是否出现每个 token 的 presence 向量接到预测头，属于上下文特征，不是 candidate embedding；`multiclass_next_token_with_order_permutation` 是顺序置换训练选项。

所以，“元素 + 对称/WP + 重数”的重新命名或把重数作为额外标量，不能单独构成新 tokenizer。可检验的差异只能来自明确的联合候选空间、元素-占位一致性约束、等价 WP 集合不变性、或实际 downstream 任务，并且要与 WyFormer/WyckoffDiff/NextCrystal 做相同覆盖率和合法性评测。

## 1. WyFormer 官方代码核对

### 1.1 两个关键 YAML

commit 对应文件：

- `yamls/tokenisers/mp_20_CSP.yaml`
- `yamls/tokenisers/mp_20_sg_multiplicity.yaml`

补充核对：GitHub commit API 显示 `e8e9add5...` 本身是 Luna 平台文档/脚本提交，以上 tokenizer YAML、源码和 notebook 是该 commit tree 中可见的版本，并非该 commit 新增的 tokenizer 设计。因此不能把“e8e9add5 引入了完整占位 tokenizer”作为历史性结论。

二者共同的 `token_fields.pure_categorical` 是：

```yaml
- site_symmetries
- elements
- sites_enumeration
```

二者共同的 engineered 字段是：

```yaml
multiplicity:
  type: map
  inputs:
  - spacegroup_number
  - site_symmetries
  - sites_enumeration
```

这说明 WyFormer 用的是分字段 tokenization：element、site symmetry、enumeration 各自有 embedding/词表；重数通过一个确定性映射恢复。`mp_20_CSP.yaml` 另有：

```yaml
sequence_fields:
  pure_categorical:
    - spacegroup_number
  counters:
    composition: elements
```

`composition` counter 的代码路径把计数器转换为两个张量：元素 token id 和对应 counts；它不是 `(element, full-WP-occupancy)` 联合 token。

### 1.2 tokenizer 实现意味着什么

`src/wyckoff_transformer/tokenization.py` 中的 `EnumeratingTokeniser.from_token_set` 接收数据集实际出现的 `all_tokens`，按排序后的 token 集加 `MASK/STOP/PAD` 建立 `token_map`。因此：

1. 词表是字段级 observed-value vocabulary；
2. 新的 element-site-enumeration 组合不是通过 tokenizer 动态合成一个新 token，而是由模型分别预测三个字段；
3. 只要字段值在词表中，模型可以在采样时产生训练集中未共同出现的组合，但组合合法性由后续 Wyckoff 映射/转换流程处理，不等于预先列举了所有合法元素占位候选；
4. 没有 `BPE`、merge table、subword vocabulary、组合 candidate embedding 或 “每元素完整占位分配”实现。

同一文件的 `SpaceGroupEncoder` 将 SG 编成对称操作矩阵的一热/压缩向量；它不是候选词表。engineered feature 的 `FeatureEngineer`/`tokenise_engineer` 只是把 map 的结果转换为可训练 tensor。

### 1.3 CSP 配置与 presence 特征

`yamls/models/NextToken/csp_v0.yaml` 的 cascade order 是 `elements → site_symmetries → sites_enumeration`，起始 token 是 `spacegroup_number`。该配置打开：

- `multiclass_next_token_with_order_permutation: true`：对 WP 顺序置换进行 next-token 学习；
- `concat_token_presence: true`：将输入序列中的 token presence 接到预测；
- `concat_token_counts: false`：不拼接 token counts；
- `learned_positional_encoding_max_size: 0`：不使用常规 learned positional sequence length。

presence 向量是上下文摘要，不能被解释为 candidate embedding；配置还明确关闭 token-count 拼接。因此即使该配置用于 CSP，代码证据仍不支持“已建立按元素枚举全部合法 WP 占位候选”的说法。

### 1.4 官方 `tokenization_discussion_anon` notebook

notebook 在 commit 中的关键代码：

- `record_to_token_counter` 将每条记录构成 `(spacegroup_number, site_symmetries[i], sites_enumeration[i])` 的 tuple counter；
- `collect_tokens` 从 augmented fingerprint 收集同样以 `(space group, site symmetry, enumeration)` 为核心的 token counter；
- 对 MP-20 统计 token frequency，并计算 train/validation 与 test 的 token set 差异；
- notebook 的实际打印结果为：模型/生成 token counter 相对 train-val 有 **81** 个新增组合，test 相对 train-val 有 **21** 个未见组合；
- notebook 的讨论明确指出 token space 是相对小的固定集合，生成时不会出现词典外新 token，方法限制应说明这一点。

这组 81/21 结果只说明“固定字段词表上的新组合与分布外 token 统计”，不说明有 BPE，也不说明生成了词典外基础 token。它支持一个可检验的研究问题：组合泛化是否来自字段独立预测、而不是来自新的 tokenizer。

## 2. WyckoffDiff 对照

来源：WyckoffDiff 论文 `arXiv:2502.06485` 正文 §3.1–3.4 及官方 `httk/wyckoffdiff` 仓库。

WyckoffDiff 把给定 SG 的全部 WP 划分为：

- constrained `L0(s)`：无自由度的位点，最多 0/1 个原子；
- unconstrained `L∞(s)`：有自由度的位点，每个元素在每个位点允许 0…P 个原子。

其 protostructure 是 `(s, z∞, z0)`：`z∞` 是 `|L∞(s)| × Na` 的计数矩阵，表示每个元素在每个自由 WP 上的占用数；`z0` 是 `|L0(s)|` 的元素类别向量。D3PM 对这些离散变量扩散，WyckoffGNN 以每个 WP 为节点，位置 embedding 包括 WP letter、自由度、SG、时间步；元素 embedding/计数 embedding 分开输入。

这已经覆盖“按元素对 WP 占用分配”的离散矩阵思想，但它不是每个 `(element, WP)` 都变成一个新 lexical token，也没有 BPE/候选组合词表。它的创新位于：固定 SG 的全 WP 节点图、constrained/unconstrained 分拆、按元素计数矩阵和非 AR 离散扩散。其直接输出是 protostructure；自由坐标和完整结构需后续实例化/弛豫。

因此以下提法属于已知方法重命名或低差异变体：

- 把 `(元素, SG, site symmetry, enumeration)` 改称“完整占位 token”，但仍逐字段 embedding；
- 把 WyckoffDiff 的 `z∞` 计数矩阵改写为一串 `(element, WP, count)` token，而模型/约束不变；
- 只把 WyFormer 的 multiplicity map 从 engineered scalar 改名为 multiplicity token。

## 3. NextCrystal（arXiv:2602.17176）对照

官方当前页显示 v4（2026-07-06）。正文方法段给出更接近“化学式→完整占位分配”的路线，但其关键创新不是新 lexical tokenizer：

1. 先把 composition 按原子数展开成 atomic sequence `C_atoms=[A,A,...,B,B,...]`；
2. `LLM_g` 从展开的元素序列预测 230 个 SG；
3. `LLM_w` 在 SG 条件下为每个原子预测可用 Wyckoff letter，SG 通过 FiLM 注入每个 Transformer block；
4. 通过约束优化/beam search 选择 letter assignment，硬约束是每种元素在每个 letter 上的 count 必须是该 WP multiplicity 的整数倍，并且总数等于输入元素计数；
5. 取 Top-5 SG × 每个 SG Top-4 合法 assignment，得到每个 composition 的 20 个模板，再送进带对称投影的 diffusion。

因此 NextCrystal 的可检验新点是“逐原子元素序列 + LLM 概率 + multiplicity-aware constrained beam search + diffusion 几何生成”，不是把所有合法 `(element, WP)` 组合预先做成 embedding 词表。它显式解决了 WyFormer 的字段独立组合与 CrystalFormer 的 emergent stoichiometry 之间的约束缺口，但成本是外接搜索器和两个 LLM；该方法也不是母相/畸变模式路线。

## 4. 候选方案的 novelty 分级

### 低 novelty / 仅重命名

1. 独立的 element/site symmetry/enumeration embeddings（WyFormer 已有）。
2. 额外输入 multiplicity scalar 或将其离散化（WyFormer engineered map 已有；WyckoffDiff/NextCrystal 都显式使用 multiplicity）。
3. 为每个 SG 建立固定合法 WP 候选 mask（WyFormer 与 CrystalFormer 都有 SG/WP 合法性约束；WyckoffDiff 按 SG 建 WP 节点）。
4. 将每元素 occupancy matrix 序列化，但不改变 WyckoffDiff 的变量、扩散或约束。

### 有一定差异但必须做实验证明

1. **set-level BPE / compositional merge**：merge 的基本单位必须是等价 WP 集合或 `(element, WP, multiplicity)` 的可逆集合片段，而不是字符串重命名；需要证明等价 WP permutation 下 merge 稳定、可逆率、词表覆盖和 OOD 组合泛化。
2. **合法候选组合 embedding**：显式构造候选 `(SG, element multiset, WP occupancy)`，embedding 同时编码元素计数和 WP multiplicity，并在采样时只允许满足逐元素计数约束的候选。与 NextCrystal 的区别必须是训练/推理如何构造候选及复杂度，而不是“有候选”本身。
3. **WyFormer 保留 permutation/set invariance 的新 tokenizer**：若基础字段仍是 WyFormer 的 site symmetry + enumeration，但 merge 单位对 WP 集合取商，并在相同模型预算下改善长序列、稀有组合覆盖或 infilling，可构成可测差异；需与原始 10,904 token/约 3 WP 平均长度基线比较。
4. **tokenizer 与 non-AR diffusion 的联动**：只有当 tokenizer 直接改变扩散变量的维度/合法状态空间，并报告 reconstruction、legal occupancy、composition exact-match、novelty、稳定性、采样成本，才不是把 WyckoffDiff 的 D3PM 换名。

## 5. 必须避免的 novelty 表述

- 不说“首次把元素和 Wyckoff 位点结合”：WyFormer 的复合语义、WyckoffDiff 的元素-WP 矩阵和 NextCrystal 的逐元素分配均已存在。
- 不说“首次 multiplicity-aware tokenizer”：WyFormer 有 multiplicity map，WyckoffDiff/NextCrystal 把重数写入变量/硬约束。
- 不说“首次 composition-conditioned symmetry generation”：NextCrystal 的摘要和正文明确以原子组成预测 SG/WP assignment 并接 diffusion。
- 不把 notebook 的 81 个新增组合说成词典外 token；它们是固定字段词表上的组合统计。
- 不把 `concat_token_presence` 说成合法候选 embedding；它只是 presence context feature。

## 来源定位

- WyFormer 官方代码 commit：`https://github.com/SymmetryAdvantage/WyckoffTransformer/tree/e8e9add5df904284116af8a06c203555d908f290`；关键文件为 `yamls/tokenisers/mp_20_CSP.yaml`、`yamls/tokenisers/mp_20_sg_multiplicity.yaml`、`yamls/models/NextToken/csp_v0.yaml`、`src/wyckoff_transformer/tokenization.py`、`src/wyckoff_transformer/wyckoff_processor.py`、`ICML_eval/tokenization_discussion_anon.ipynb`。
- WyckoffDiff：`arXiv:2502.06485`，正文 §3.1–3.4（representation、D3PM、WyckoffGNN），附录 A（embedding 和变量维度）；官方仓库 `https://github.com/httk/wyckoffdiff`。
- NextCrystal：`https://arxiv.org/abs/2602.17176`，当前 v4；正文方法段，尤其公式 (1)–(3)、beam-search 候选生成和 diffusion 条件（官方 HTML 约行 57–82）。

本笔记只写入本文件；没有修改代码或运行训练/生成实验。
