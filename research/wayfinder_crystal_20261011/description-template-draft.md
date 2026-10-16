# 结构语言描述：字段与提示词草案

日期：2026-10-12。用途是讨论离线标注内容，不是调用 LLM 或构建完整训练集。用户已确认使用现成 LLM、采用本草案的字段设计和提示词，并要求提示词与描述输出均为英文。标注模型已选定为 SciencePal 网关的 deepseek-v4-pro，使用 thinking enabled 与 reasoning_effort max；局部结构的分析方法仍待选择。

## 数据来源

三条实际结构的源字段与 PLaID 解码字段分开保存在 [字段样例](description-field-examples.json)。现有数据以 Structure JSON 保存，可直接解析，无需先写 CIF 再读入；本次为复用原版 PLaID 编码接口，临时转换为 CIF 字符串。原始 ehull 保留原值与 MP 来源，不重新赋给解码结构作为已计算的性质。

## 首轮核心字段建议

- 组成：元素、元素种类数、晶胞组成和约化化学式。
- 对称：空间群号/符号、晶系；计算时记录对称容差。
- 晶胞：a/b/c（Å）、alpha/beta/gamma（度）、体积（Å³）、晶胞原子数。
- 位点：独立轨道数量、元素与 Wyckoff 标签；不把轨道记录数当成总原子数。
- 可选派生描述符：由单位晶胞质量/体积得到的理论密度（g/cm³）、原子数密度与几何原子间距。
- 丰富结构字段：配位数、近邻元素、配位多面体、角/边/面共享、链/层/三维连接。需要先选定计算方法与参数，本次未提取；字段研究说明放在 [字段证据](description-fields-evidence.md)。
- 额外性质：已有 MP ehull 单独传入；带隙、形成能、弹性/输运/热学性质需要相应标签或计算，不从几何直接生成数值。

核心字段中的原子计数、晶胞组成、体积和密度应来自同一结构版本。首轮建议使用最终 PLaID 解码结构的事实生成文字，原始结构的字段作为来源检查。显示时的舍入不能覆盖存储的原始标签；输出标签的数字精度单独决定。

## Approved English annotation prompt

```text
You are an annotation assistant for crystal structure datasets.
Write a concise, natural description of the supplied crystal in English,
using 3–4 sentences and 80–110 words.

Inputs:
- structure_version: the specific structure version being described.
- core_facts: composition, symmetry, unit-cell information, and independent
  Wyckoff orbits.
- local_structure_facts: optional coordination environments, polyhedra,
  and connectivity, together with their analysis sources.
- property_labels: optional supplied property values, units, and provenance.

Instructions:
1. Use only the supplied facts. Omit missing information.
2. Start with the composition and symmetry. When local structural facts are
   available, emphasize the main coordination environments and connectivity.
   Add unit-cell parameters, counts, or theoretical density when useful.
3. Preserve chemical formulas, element identities, space-group identifiers,
   counts, and units. Distinguish the number of atoms in the unit cell from
   the number of independent Wyckoff orbits.
4. Describe density calculated from composition and unit-cell volume as
   "theoretical density". Follow the supplied symmetry analysis rather than
   assigning a different space group or crystal system.
5. If a property label is included, state its source. A label attached to the
   original MP structure must not be presented as a property recalculated
   for the PLaID-decoded structure. Preserve supplied property-label values.
6. State structural facts and property labels separately. Do not invent
   oxidation states, bonding, morphology, material performance, or causal
   claims about why a structure improves a property.
7. Output only an English paragraph. Do not output a CIF, a complete
   coordinate table, a new structure encoding, or additional commentary.
8. When reporting coordination or connectivity, identify them as
   geometry-derived results of the supplied analysis. Use only supplied
   recognized motif labels; do not turn raw order-parameter scores into
   additional polyhedron assignments or treat graph edges as chemical bonds.
9. Use "neighbors" or "geometric coordination" for local environments,
   never "bonded" or "chemical bonds". The neighbor count per center atom
   differs from the number of independent orbits or unit-cell atoms sharing
   that environment. Summarize one or two main environments; do not list
   every environment, raw score, or full set of lattice parameters.
10. A close-packed-like local arrangement does not specify cubic or
    hexagonal close packing. Do not add either subtype unless supplied.
    Describe the crystal without its material ID or tokenization pipeline
    name. Omit unavailable facts rather than discussing their absence.

Structure version:
{structure_version}

Core facts:
{core_facts}

Local structural facts:
{local_structure_facts}

Provided property labels:
{property_labels}
```

该提示词用于离线数据标注。所有描述正文使用英文；输入事实与标签的原始数值单独保存，不由 LLM 覆写。试标核查发现模型会舍入性质数值，因此当前 LLM 仅组织 3–4 句结构文字；程序随后追加一句包含原始 ehull 值及 MP 来源的英文句子，最终为 4–5 句。完整几何数值留在结构事实文件，语言输入选用组成、对称、计数及主要局部环境摘要。

## 真实样本示例（人工按提取事实组织，非 LLM 输出）

材料 ID：mp-624928。描述对象：PLaID 解码结构。

> The crystal contains potassium, europium, copper, and tellurium, with a unit-cell composition of K1Eu1Cu2Te4. It belongs to the tetragonal crystal system and space group P4mm (No. 99). The unit cell contains eight atoms distributed over six independent Wyckoff orbits. Its lattice parameters are a = b = 4.44 Å and c = 11.37 Å, with all three lattice angles equal to 90°. The theoretical density calculated from the unit-cell composition and volume is approximately 6.138 g/cm³.

原始 MP 结构的 ehull 标签为 0.017097575234372 eV/atom，作为配对字段保存，未对本次解码结构重算。正式文字可选择保留或省略晶格/密度句子，以控制长度；以上字段不必全部塞进每条描述。

## 下一项决定

标注 LLM 已选定，接口采用 Chat Completions。接下来固定局部配位/连接事实的提取方法，完成 PLaID 编解码与字段提取，再用少量训练材料试标检查事实一致性和英文风格，确定质量验收方式。
