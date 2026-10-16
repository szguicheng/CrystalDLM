# Wayfinder 结构语言描述字段：证据与建议

核查日期：2026-10-12。用途是确定现成 LLM 离线标注可使用哪些已知字段；不训练 CIF-to-text 模型，不运行标注，不把自然语言当作晶体的无损表示。

## 结论边界

CIF 是可选字段丰富的数据交换格式，不是性质预测器。IUCr Core 字典定义了晶胞、原子种类、坐标、占位、对称性以及若干结构结果项；具体 CIF 是否包含这些值取决于文件。CIF 还可带实验晶体描述、尺寸或实测密度等元数据，这些字段不能从原子结构反推出。

本项目当前 MP 导出记录保存 `structure`（pymatgen `Structure` JSON）和独立的 `energy_above_hull`，而不是现成 CIF 文本；记录另有公式和来源信息，标签单位为 eV/atom。[导出字段](../../research/mp_ehull030_20261011/finish.py) 结构 JSON 的核心是 `charge`、`lattice`、`properties`、`sites`，没有从结构本身直接给出总能量或 band gap。`charge` 是整个结构的净电荷，不是各元素的氧化态。若转换为 CIF，应把记录来源及额外性质另行保留：pymatgen 文档说明 `Structure.properties` 在写成 JSON/YAML 时序列化，但转换为其他格式会丢失。[pymatgen Structure](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure)

## 字段盘点

| 字段类别 | 来源、单位与推导条件 | 首轮建议 | 缺失或不确定时 |
|---|---|---|---|
| 元素、组分、化学计量 | CIF `_atom_site.type_symbol` 标记位点元素；`_atom_site.occupancy` 是该位点元素的占位分数（无量纲）；由位点及占位可汇总化学式。异位混占/空位需要保留占位信息，不能强行化为整数计量。[元素](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.type_symbol.html)、[占位](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.occupancy.html) | **核心**：报告化学式/元素组成；不必在描述里复述全体坐标。 | 占位不完整或非整数时，保留原值并标记无序；不要四舍五入成确定组分。 |
| 晶胞度量（晶胞“形状”） | CIF 的 `a,b,c` 单位 Å，`α,β,γ` 单位度；`_cell.volume` 可存储或按晶格向量三重积计算，单位 Å³。pymatgen 从 lattice 提供体积和理论密度。[晶胞长度](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Icell.length_a.html)、[体积](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Icell.volume.html)、[pymatgen 体积/密度](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure) | **可选**：若数值有用且对应同一标准晶胞，列晶系/晶格度量；短描述通常不需要重复精确参数。 | 明确晶胞设定（primitive/conventional）和单位；不要把晶胞度量叫作宏观晶体外形。 |
| 原子位点与占位 | CIF 可给分数坐标（无量纲）、笛卡尔坐标（Å）、元素和占位；也可存热位移参数和不确定度。分数坐标依赖晶胞基底/原点，不是绝对位置。[分数坐标](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.fract_x.html)、[Core CIF 字典](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/index.html) | 作为可解码结构的底层输入；生成短自然语言时不逐位点复述。 | 没有实验不确定度时不虚构；无序信息缺失时不声称全占位。 |
| 空间群、Wyckoff 位点 | CIF 可能提供空间群符号/编号、对称操作和位点 Wyckoff 字母；若缺失或需统一，应对实际结构用 spglib/pymatgen `SpacegroupAnalyzer` 或 PyXtal 推断。结果依赖对称容差、角度容差、晶胞设置、原点选择；pymatgen 默认 `symprec=0.01 Å`，其文档也说明轻微偏离的结构可能需要更宽松容差。PyXtal 的 Wyckoff 位点包含对称等价位置与 site symmetry。[CIF 空间群字段](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Ispace_group.name_H-M_alt.html)、[CIF Wyckoff 字段](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.Wyckoff_symbol.html)、[pymatgen symmetry](https://pymatgen.org/pymatgen.symmetry.html#pymatgen.symmetry.analyzer.SpacegroupAnalyzer)、[PyXtal](https://pyxtal.readthedocs.io/en/latest/Background.html) | **核心候选**：报告经固定方法重算且稳定的空间群；Wyckoff 标签更适合作为结构字段或较丰富的说明。 | 保存字段是 CIF 原文给出的，还是某算法重算的；缺失或算法失败就省略，禁止让 LLM 猜。 |
| 晶胞体积、理论密度 | 体积 Å³ 可由晶格算出。理论晶体密度由晶胞质量/晶胞体积算出，pymatgen 单位 g/cm³；占位无序会影响计数与解释。CIF 也可选存衍射计算密度、实验测量密度，两者应分开标记。[pymatgen density](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure)、[CIF 实验晶体类别](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Cexptl_crystal.html) | **可选丰富字段**；不是短结构描述的必需项。 | 标签记录 `calculated_from_structure` 或 `measured` 与来源；不要把理论值称为样品堆积密度。 |
| 局部配位数、键长、多面体 | 从结构计算位点间周期距离，再由近邻/成键规则决定邻居。CN 无量纲、距离 Å；几何类别（八面体、四面体等）依赖 neighbor 算法、距离/权重与几何拟合阈值。pymatgen 的 `StructureGraph` 明确指出图上的“bond”不必然是化学键；Robocrystallographer 默认可用 CrystalNN，也可替换 VoronoiNN 等。[pymatgen local environment/graph](https://pymatgen.org/pymatgen.core.html#pymatgen.core.local_env)、[Robocrystallographer condenser](https://hackingmaterials.lbl.gov/robocrystallographer/_modules/robocrys/condense/condenser.html) | **可选丰富字段**：只收录固定算法下稳定且可读的主要配位多面体及连接方式；记录方法/版本。 | 若算法结果模糊或对设置敏感，省略 CN/多面体/“键”；不要把某个 neighbor graph 当作唯一化学真值。 |
| 连接维度 | 以选定局部近邻构成周期图，再分析周期连通性，可得 0D/1D/2D/3D 结构组件；这是算法输出，不是空间群的晶系，也不等于样品形貌。Robocrystallographer 的 condensed JSON 明列 `dimensionality`，并可基于 CrystalNN 等近邻策略。[Robocrystallographer 格式](https://hackingmaterials.lbl.gov/robocrystallographer/format.html)、[实现与默认策略](https://hackingmaterials.lbl.gov/robocrystallographer/_modules/robocrys/condense/condenser.html) | **可选丰富字段**；只在固定近邻/周期图设置上可靠时使用。 | 与 CN 共用方法记录；不确定则省略，不凭“看起来像层状”猜维度。 |
| 宏观晶体习性/外形 | CIF 的 `_exptl_crystal_description` 可记录实验晶体的质量与 habit；尺寸、晶面另有实验字段。这描述的是实际样品形貌，不能从单位晶胞边长、局部多面体或理想结构直接推出。[IUCr `_exptl_crystal_description`](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Iexptl_crystal_description.html)、[实验晶体类别](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Cexptl_crystal.html) | **排除首轮结构描述**，除非另有样品记录。 | 无实验形貌来源时写未知/省略；不得从晶胞轴长推断“针状、片状、柱状”。 |
| 氧化态 | CIF 类型符号可显式含价态（例如 `Cu2+`），但普通元素符号和结构 `charge` 不确定每个位点氧化态。可用电中性、键价规则等做推断，但这是带方法/假设的结果。Robocrystallographer CLI 在未见氧化态时会运行 `add_oxidation_state_by_guess`，而描述器默认开启氧化态句子。[CIF type_symbol](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.type_symbol.html)、[Robocrystallographer CLI 实现](https://hackingmaterials.lbl.gov/robocrystallographer/_modules/robocrys/cli.html)、[描述选项](https://hackingmaterials.lbl.gov/robocrystallographer/robocrys.describe.html) | **不进首轮**，除非源数据明确给出每个位点价态并保留其出处；对现成 LLM 同样禁止猜。 | 缺失时 omitted；若另行推断，作为带方法标签的字段，不伪装成观测事实。 |
| ehull / 总能量 | 当前独立标签 `energy_above_hull` 单位 eV/atom，是数据源计算标签；MP 定义的 ehull 由一组形成能和组成构成的相图凸包计算。单个 CIF/Structure 不足以恢复竞争相能量；该字段不能从晶体几何反算。[MP glossary](https://docs.materialsproject.org/frequently-asked-questions/glossary-of-terms)、[本地导出字段](../../research/mp_ehull030_20261011/finish.py) | 把原值留作性质标签，不当作结构句子或模型预测事实；若文字需要提及，应明确说这是数据源标签。 | 保留原始精度、来源、单位和结构版本；缺失为缺失，不填 0。 |
| band gap | 电子结构性质，单位 eV；MP 描述的 band gap 来自 DOS、能带或其他 DFT 计算层级，且 DOS 与 line-mode 结果可不同。[MP electronic structure](https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure) | **不进首轮结构描述**；只有另行保存了指定来源/计算口径的标签才作为外部性质。 | 当前导出不含该值；不从结构图像或元素组合补出。 |
| 弹性、电学、热学性质 | 弹性张量/模量来自应变—应力计算或实验，常用 GPa；电导/介电等需电子响应计算或实验；热导率通常需热输运/声子模型或实验，单位随指标而异。MP 弹性张量通过 DFT 弛豫、施加应变、计算应力再拟合得到。[MP elasticity](https://docs.materialsproject.org/methodology/materials-methodology/elasticity)、[MP data workflows](https://docs.materialsproject.org/data-production/data-workflows) | 不当作由 CIF 几何直接提供；确有标签时作为带测量/计算来源的独立条件。 | 数值和方法都缺失时省略；不同计算口径不可混标。 |
| 磁性 | 磁矩通常以 μB/site 或 μB/f.u. 表示；磁序需自旋构型/磁对称信息或实验。CIF 有磁性扩展字典，但普通几何结构不编码基态磁序。MP 明确指出其许多结构的结果只是从铁磁初态计算后得到的计算磁序，并非真实基态预测。[MP magnetism](https://docs.materialsproject.org/methodology/materials-methodology/magnetic-properties)、[IUCr CIF dictionaries](https://www.iucr.org/resources/cif/dictionaries/cif_core) | 不进首轮结构描述；仅当磁矩/磁序字段随标签谱系提供时单独使用。 | 未提供时未知；不得从过渡金属/元素价态推测磁序。 |

## PLaID 表示版本与字段归属

当前实现 [work/PLaID/utils.py](../../work/PLaID/utils.py) 的 `get_crystal_string_wyckoff_pyx` 读取 CIF，再用 `SpacegroupAnalyzer(..., tol=0.01).get_refined_structure()` 精修；之后 PyXtal 以 `tol=0.01` 提取空间群、晶格和独立位点（失败时回退 `0.0001`）。字符串包含公式、空间群、三条晶格边（Å、保留 0.01 Å）、三个晶胞角（度、保留 0.01°）、晶胞原子数，以及每个代表位点的元素、分数坐标（保留 0.001）和 Wyckoff 标签。这些值定义了经精修/舍入的 PLaID 表示，不等同原始 MP 结构逐值拷贝；本地 PLaID 核查记录也将“精修/舍入后的描述”与源结构区分。[本地编码核查](../../research/plaid_20260930/READING_NOTE.md)

离线描述应针对拟训练的晶体序列版本：若目标序列是 PLaID 编码/解码结构，就从该解码结构计算空间群以外的局部配位、键长和网络维度；不能把原始结构上的这些字段无标记地搬过来。`ehull` 仍应保留为原始 MP 记录的标签，并另存 `source_structure` 标识、PLaID 表示/解码版本和结构摘要（如哈希）。在未重算性质前，不声称舍入后结构本身具有完全相同的 ehull。

## 首轮字段与 LLM 输入约束建议

1. **最小核心**：化学式/组成；经记录的空间群（以及可选晶系）；结构版本 ID。PLaID 结构 token 本身承载几何数值，自然语言不要复制完整坐标表。
2. **可选丰富字段**：固定程序计算后较稳定的局部配位多面体、代表性键长区间和多面体连接维度。每项配套保存算法与版本、容差/近邻策略、输入结构版本。
3. **性质标签分层**：`energy_above_hull`、band gap、弹性、磁性等保留为各自独立、有来源/单位/方法的字段；只有明确希望文字陈述它时，才作为受约束输入，不要把材料性质混成“结构事实”。
4. **提示词约束**：只把已提供字段改写成短句；不得推断氧化态、化学键、宏观习性或材料性能；不确定/缺失字段省略，不填零、不造值；含性质时需标明其为给定标签并保留来源口径。Robocrystallographer 可用于字段设计参考，但其 CLI 会默认猜氧化态，因此若借用该管线需关闭氧化态句子，或仅使用经过独立核实的中间字段。

## 一手资料

- [IUCr Core CIF dictionary](https://www.iucr.org/resources/cif/dictionaries/cif_core) 与 [Core CIF 定义索引](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/index.html)
- [pymatgen core / Structure / local environment](https://pymatgen.org/pymatgen.core.html)；[pymatgen symmetry analyzer](https://pymatgen.org/pymatgen.symmetry.html)
- [PyXtal Background and Theory](https://pyxtal.readthedocs.io/en/latest/Background.html)
- [Robocrystallographer source/docs](https://github.com/hackingmaterials/robocrystallographer)、[CLI](https://hackingmaterials.lbl.gov/robocrystallographer/cli.html)、[condensed format](https://hackingmaterials.lbl.gov/robocrystallographer/format.html)
- [Materials Project glossary](https://docs.materialsproject.org/frequently-asked-questions/glossary-of-terms)、[electronic structure](https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure)、[elasticity](https://docs.materialsproject.org/methodology/materials-methodology/elasticity)、[magnetic properties](https://docs.materialsproject.org/methodology/materials-methodology/magnetic-properties)
