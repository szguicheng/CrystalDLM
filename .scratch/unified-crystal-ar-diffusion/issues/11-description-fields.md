# 核查 CIF 可提取字段与离线描述提示词

Labels: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: Codex /root
Worker: /root/wayfinder_description_fields
Parent: ../map.md
Blocked by: 01
Research context: [CIF 字段与描述依据](../../../research/wayfinder_crystal_20261011/description-fields-evidence.md)

## Question

现有结构数据和 CIF 能直接提供哪些结构信息，哪些描述符能通过晶体程序计算，哪些材料性能必须使用额外标签或计算？怎样将这些字段组织成现成 LLM 的离线标注输入和简短、可核对的结构语言描述？

核对 IUCr CIF 字典、pymatgen/PyXtal、PLaID 实际编码器和 Robocrystallographer 官方资料。至少区分组成、晶格、占位/坐标、计算对称性/Wyckoff、几何体积/理论密度、局部配位/多面体/连接维度，以及 ehull、带隙、弹性/电学/热学性质的标签来源。说明氧化态、配位/键与结构维度的算法依赖；“晶胞形状”“局部配位多面体”“宏观晶体外形”不能混为同一字段。

给出首轮核心字段和可选丰富字段的建议、字段来源与单位、未知值处理、原始与 PLaID 解码结构的版本对应、最小提示词约束。不另训练描述模型，不假定自然语言描述可无损还原晶体，不把它变成新 tokenizer 路线。

只读资料和少量样本字段；不要安装软件、调用标注 LLM、批量处理结构、修改源数据、下载模型或运行训练。报告中的提示词为待讨论建议，不代表字段/模板已获用户批准。

## Comments

### Claim — 2026-10-12

用户明确同意用现成 LLM 离线标注，并要求先查清 CIF 能提供什么性质和结构信息。主智能体在委派前认领，本票提供字段证据，不锁定模板。

### Resolution — 2026-10-12

[CIF / Structure JSON 的结构语言描述字段：证据与建议](../../../research/wayfinder_crystal_20261011/description-fields-evidence.md)

- CIF 可直接存元素、占位、晶胞度量、坐标、对称信息及部分实验元数据；缺少的对称与局部结构可通过 pymatgen/PyXtal 等程序推导，但结果取决于容差或近邻算法。
- 协调数、多面体、连接维度和氧化态需注明算法；LLM 不应猜价态、键或唯一近邻图。晶胞度量、局部多面体与实验晶体习性是不同字段。
- ehull、带隙、弹性、磁性等来自数据标签或额外计算/测量，不是从 CIF 几何直接得到。当前记录是 Structure JSON 配独立 ehull 标签。
- PLaID 先精修并舍入晶格/位点，所以描述应对应明确的结构版本；原始 MP ehull 仍带源结构谱系，不可当作舍入结构重新验证过的性质。

### 主智能体核查

已检查 PLaID 编码器、实际 Structure JSON 字段及三个源/解码结构案例，另行计算体积、理论密度和空间群；独立轨道多重性之和等于晶胞原子数。报告的 richer 配位/连接字段仅为文档证据，本轮没有运行该分析。
