# 4.3.5 Crystals：对称性表示证据核对

核对范围：只比较 CrystalFormer、WyFormer 和 PLaID 系列三个最能区分表示边界的案例；底层共核对四篇原始论文。以下把晶体学表示单位与生成模型分开：CrystalFormer、WyFormer、PLaID/PLaID++ 都是生成方法；“tokenizer”不应指代整个生成器。

## 对照结论

| 案例 | 表示单位 / 晶格与坐标 | 完整结构还是装饰原型 | 对称性与顺序、标准化 / invariance |
|---|---|---|---|
| **CrystalFormer** | 给定空间群 `g`，逐个表示非对称单元原子：Wyckoff 字母 `W`、元素 `A`、代表原子的分数坐标 `X`；再表示晶格参数 `L`。`W/A` 是离散量，活动的 `X/L` 是连续量。 | 编码包含恢复晶胞所需的空间群、非等价位点、坐标和晶格；等价原子按空间群操作展开。因此是完整结构参数化，不只是着色 Wyckoff 原型。 | 采样时屏蔽对该 `g` 无效的 Wyckoff 字母，并按字母序排列（特殊位点到一般位点）；相同字母的自由位点按分数坐标字典序排列。论文说明通过选定空间群原点固定平移自由度，并处理分数坐标周期性；没有据此声称对任意基矢/晶胞表示不变，也没有把它定义为独立 tokenizer。 |
| **WyFormer** | 一个空间群 token，加每个占据 Wyckoff 位点的离散三元组 `(element, site-symmetry, enumeration)`。该表示**不含晶格度量或自由原子坐标**。 | 仅为 Wyckoff 装饰原型，不能无损确定原晶胞几何。生成后另用 PyXtal 构造初始 3D 结构，再经 MLIP、DiffCSP++ 或 DFT 弛豫；这条后处理链属于生成流程，不是 tokenizer。 | 用空间群及 site symmetry / enumeration 表达合法 Wyckoff 占据；把位点作为无序集合输入，Transformer 不加 positional encoding，并用随机排列增强。论文还处理等价 conventional-cell 选择与 Euclidean-normalizer coset representative 导致的 Wyckoff 标号变化；这不是对晶格度量的编码或标准化。 |
| **PLaID → PLaID++** | 结构被序列化为文本字段：晶胞化学式、空间群、晶格边长与角度，以及非等价 Wyckoff 位点的元素、代表分数坐标、位点标签/多重性。公开实现将晶格取两位小数、坐标取三位小数，再交给 LLM 自带 tokenizer 分词；论文提出的是 Wyckoff 文本表示，不是独立的晶体词表。 | 字段含晶格与代表坐标，结合空间群和 Wyckoff 标签可展开为完整晶胞。官方 parser 调 PyXtal 构造完整结构。有限精度舍入意味着它不是原始 CIF 的逐数值无损编码。 | encoder 先调用 `SpacegroupAnalyzer.get_refined_structure()`，再按固定头部和 `pyx.atom_sites` 顺序写行；可见实现没有显式位点排序，也没有宣称集合置换不变性。对称性由空间群/Wyckoff 位点及 PyXtal 重建承载。 |

### 各项原始证据位置

**CrystalFormer.** 正式论文 §2.1.1（PDF 第 2 页）定义空间群、逐位点 Wyckoff 字母/元素/分数坐标以及晶格参数；同节给出空间群合法字母、字母排序和无自由坐标特殊位点的约束。§2.1.1 后续讨论（PDF 第 3 页）明确同字母位点按分数坐标字典序排列、以选定原点固定连续平移自由度，并说明分数坐标按周期变量处理。作者公开的 [CrystalFormer model card](https://github.com/deepmodeling/CrystalFormer/blob/main/README.md#model-card) 也列出 `g-(W-A-X)-L` 序列和“由空间群/Wyckoff 字母恢复其余等价原子”。

**WyFormer.** 原文 §2.1 “Tokenization” 定义空间群 token 与每个位点三元组，并定义表示 `R=(G,E,W)`；该定义不含 `L` 或坐标。§2.1 说明等价 conventional unit cell 会改变 Wyckoff 位置但不改变 site symmetry，并将任意性集中到 enumeration。§2.3 说明无 positional encoding 的排列不变 Transformer及打乱增强。§2.4 / Algorithm 1 将生成表示转换为 `(element, Wyckoff letter)`，再用 PyXtal 建立初始 3D 结构并交给后续弛豫。见 [arXiv 原文](https://arxiv.org/abs/2503.02407) 与[作者代码库](https://github.com/SymmetryAdvantage/WyckoffTransformer)。

**PLaID 系列。** 原始 PLaID 论文 §2.2 已明确序列字段包含化学式、空间群、晶格参数、元素、分数坐标和 Wyckoff 位点对称性。PLaID++ 正式论文 §4.3、Figure 2（PDF 第 4–5 页）说明其以整个 asymmetric unit 的 Wyckoff 文本表示结构；Appendix 7.2 “Wyckoff Crystal Representation”（PDF 第 13 页）给出逐字段格式、PyXtal 对称容差 `0.01`、晶格两位小数和坐标三位小数。官方代码固定快照 [utils.py encoder lines 11–66](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/utils.py#L11-L66) 先精修结构、写晶格和代表位点；[parser lines 104–170](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/utils.py#L104-L170) 读回晶格/空间群/代表坐标与 Wyckoff 标签，并由 PyXtal 生成完整结构。[训练入口 lines 538–579](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/llm_finetune.py#L538-L579) 使用 Qwen 或 LLaMA 基础 tokenizer，仅在缺失时补齐通用 special tokens；没有 Wyckoff 专用 tokenizer 的证据。代码快照没有在位点循环前显式排序；因此不应把它描述成置换不变表示。

## 可用于正文的简短区分

- **CrystalFormer**：空间群条件下对称独立位点 + 连续坐标/晶格的 AR 结构生成表示。
- **WyFormer**：离散、置换不变的 Wyckoff 装饰集合；没有晶格与自由坐标，3D 几何由后续构造/弛豫补全。
- **PLaID/PLaID++**：LLM 使用的 Wyckoff 结构文本序列；字段显式保留晶格和位点代表坐标，文本经现成 LLM tokenizer 处理并由 parser 展开结构。

## 准确书目信息

1. **CrystalFormer** — Zhendong Cao, Xiaoshan Luo, Jian Lv, and Lei Wang. “Space Group Informed Transformer for Crystalline Materials Generation.” *Science Bulletin* 70(21), 3522–3533 (2025). DOI: [10.1016/j.scib.2025.09.035](https://doi.org/10.1016/j.scib.2025.09.035). 原文：[publisher PDF](https://www.iop.cas.cn/xwzx/kydt/202510/P020251113378898649957.pdf)；预印本 [arXiv:2403.15734](https://arxiv.org/abs/2403.15734)。
2. **WyFormer** — Nikita Kazeev, Wei Nong, Ignat Romanov, Ruiming Zhu, Andrey Ustyuzhanin, Shuya Yamazaki, and Kedar Hippalgaonkar. “Wyckoff Transformer: Generation of Symmetric Crystals.” (2025), ICML 2025; [arXiv:2503.02407](https://arxiv.org/abs/2503.02407). 作者论文与实现：[arXiv](https://arxiv.org/abs/2503.02407), [official repository](https://github.com/SymmetryAdvantage/WyckoffTransformer)。
3. **PLaID** — Andy Xu, Rohan Desai, Larry Wang, Gabriel Hope, and Ethan Ritz. “PLaID: Preference Aligned Language Model for Targeted Inorganic Materials Design.” AI4Mat @ ICLR 2025 workshop. 无 DOI/arXiv 编号；一手论文：[OpenReview PDF](https://openreview.net/pdf?id=7aoP3ZeBfy)。
4. **PLaID++** — Andy Xu, Rohan Desai, Larry Wang, Ethan T. Ritz, and Gabriel Hope. “PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design.” *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306:140903–140922 (2026); [arXiv:2509.07150](https://arxiv.org/abs/2509.07150), [PMLR proceedings page](https://proceedings.mlr.press/v306/xu26l.html).

**选文建议：**若正文只留三项，采用 CrystalFormer、WyFormer、PLaID++ 三条；PLaID 2025 作为 PLaID++ 的前身文献列入参考文献（两者不能合并成同一条 bib）。这样三种表示的区别最清楚，也保留了 PLaID 系列书目沿革。
