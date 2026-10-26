# Wyckoff 计数块 tokenizer

该编码仅组合标准晶体学量：空间群、Wyckoff 位置、重数，以及同一元素在某 WP 上的独立 orbit 占据次数。**一个计数块是编码分组，不引入新的物理结构实体。**

## 定义

空间群 G 是全局头部，每个元素单独编码。基础记录为 `(Wyckoff字母, count)`，其中 count 是独立 orbit 次数。词表把常见的、同元素的最多 3 个不同 WP 记录合成一个块；其余记录保留为一个 RLE 单元。输入计数本身不受 learned merge 词表覆盖限制。

例如空间群 62 的一条真实学习规则把 `8d × 1` 和 `4c × 1` 编成一个词。它展开后仍是两个独立 orbit，共 8+4=12 个原子，而不是两个原子。

```python
from count_block_tokenizer import CountBlockTokenizer

tokenizer = CountBlockTokenizer("count_block_tokenizer.json")
tokens = tokenizer.encode(62, {"d": 1, "c": 1})
# ((('c', 1), ('d', 1)),)

assert tokenizer.decode(tokens) == {"c": 1, "d": 1}
assert tokenizer.atom_count(62, tokens) == 12
```

`count_block_tokenizer.py` 只需要 Python 标准库。JSON 包含固定表格元数据、182 条规则和模型使用的确定词序。改变输入字典的顺序不改变编码。未见过的 `(WP,count)` 可以保留为单条记录；例如 SG1 的 `a×37` 已检查，但不代表神经模型在这种输入上经过验证。

## 还原范围

`decode` 还原每个元素的 Wyckoff 占据计数。它不生成晶格参数、自由坐标或 CIF；几何阶段仍需为每个 WP 的独立 orbit 提供相应参数，并执行空间群操作展开。

输入应为已采用相同标准设置的 WP profile。表格来自固定的 PyXtal 1.1.4，含 230 个空间群。当前不对全部原点/setting 等价关系取商，不以 token 相等作为完整物理结构等价判断。

## 已验证的编码性质

学习只使用模型共同保留的 26,086 个训练结构，没有用验证或测试频率学习规则。

Burgundy 作业 533081 对全部 **45,180 个成功预处理结构**检查了离散还原、原子计数、输入顺序不变性和相对 RLE 不变长，均通过；另对全部 32,368 个模型候选验证了与冻结训练模型的词序完全相同。

| 划分 | 成功预处理结构 | RLE 单元 | 计数块单元 | 减少 |
|---|---:|---:|---:|---:|
| train | 27,106 | 99,740 | 89,884 | 9.88% |
| val | 9,037 | 33,241 | 29,966 | 9.85% |
| test | 9,037 | 33,129 | 29,949 | 9.60% |

这些是所有成功预处理数据的 codec 统计。当前模型实验另有候选数上限筛选，测试模型只覆盖其中 8,730 个结构；两种范围不可混淆。源数据还有 49 个预处理一致性失败样本，已单独排除，不在上述通过数量中。

## 实验状态

多种子与冻结测试结果见 `ASSESSMENT.md`。编码压缩本身不证明模型质量提升，槽位/显存减少也不证明吞吐加速。下载包包含可独立调用的编码/解码器、词表、审计与结果摘要；完整训练流水线和原始检查点仍在实验目录/Burgundy，不是本小包的一部分。

完整的学习与导出流程在 `../prepare_count_merges.py`、`../export_count_tokenizer.py`。原始占据计数表示与通用频率合并均有已有工作；文献边界见 `../../../research/notes/count_block_crystal_prior_art.md` 和 `../../../research/notes/set_merge_prior_art.md`。
