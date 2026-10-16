# Hsame 003 core128 失败分解

只读核对 `hybrid_same/validation_128/evaluation.jsonl`（128 行，raw SHA-256 `87b024e74f4e95cb7330cf55ad49e914aa8c822bf739a142454e3d65c1eaa7a1`）。摘要也是全 128 分母；prefix receipt 为旧 66 行科学字段精确相等，`forward_call_differences=[]`。

| 阶段 | 互斥计数（分母128） | 既有 reason 分类 |
|---|---:|---|
| Generation | G 0；无 AR trigger 失败 | 48 END 数量/边界拆包失败；31 PAD 边界拆包失败；49 块可拆但 PLaID audit 解析失败 |
| Generation 的49个 audit 解析失败 | 49 | 8 lattice header；7 spacegroup header；3 Sites header；19 spacegroup/WP label 或 multiplicity；12 coordinate/site-row parse |
| Reference edit | E_reference 0 | 58 拆包失败（50 END、8 PAD）；另 70 可拆块中，67 个 `grade_edit` parser ValueError、3 个解析后字段比较失败 |

67 个编辑 parser error 的类别为：lattice header 23、spacegroup/WP label 14、site-row arity 12、numeric values 12、Sites header 6。三个字段比较失败行是 `mp-9024`（lengths、sites）、`mp-10298`（sites）、`mp-864621`（lengths、declared_atoms）；字段计数有重叠，总共仍只有3行。故 E 的互斥总数为 58+67+3=128。

代表输出：

- `mp-1212698` generation：既有 reason 为 `crystal block requires exactly one END`；原文末尾有 `<CRYSTAL_END><CRYSTAL_END><CRYSTAL_PAD>`。
- `mp-976412` generation：audit reason 为 `Wyckoff multiplicity does not match label`。
- `mp-864621` reference edit：可解析，但 `lengths` 和 `declared_atoms` 比较失败；原文为 `Nd2Ga6 ... abc: 6.64 6.15 4.38 ... Sites (88)`。
- `mp-1793` reference edit：既有 parser reason 为 `invalid lattice header`；原文片段 `abc: 11.844 10.845.40`。

解释边界：G 的79行均在 trigger 检查之后失败于块边界；49行则进入 PLaID audit 后先遇到格式、标签或坐标解析错误。当前没有 generation 行到达几何/物理校验并返回物理失败 reason；这表示物理结果未被解析器评估，不能推成物理正确。上述分类复用原始记录与现有 reason，不修改输出或评分器，也不影响正在运行的 Hshift。
