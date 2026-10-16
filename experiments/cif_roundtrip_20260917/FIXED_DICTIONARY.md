# 固定规则字典与边轨道序列

在原始可逆 CIF 编解码器上增加共享字典。原有内联 JSON 格式和测试结果保留。

## 字典保存什么

`rules.json` 包含：

- `groups`：空间群号、Hall 编号、输入晶胞基底和原点下的完整操作表。
- `sites`：所属 group ID、Wyckoff 字母、位点稳定子及整数晶胞像、自由度。
- `id`：全表内容的 SHA256，用来绑定序列与准确的字典版本。

字典不保存元素、材料 ID、连续坐标、晶格长度、化学式或连接列表。相同规则只保存一次。不同晶胞基底/原点的同一空间群可能对应不同 group 条目，不能只用空间群号查表。

本次字典来自原来十个案例中的符号规则，随后冻结。它是这批设置的可复现字典，不是已经覆盖全部 230 空间群和任意原点/基底的通用字典，也不是未见规则泛化实验。对新数据，应先独立准备并冻结对应规则库，再开始评估。遇到未收录的规则，encoder 报错，不会自动扩表或把原始坐标藏入字典。

## 实际序列语法

`.tokens` 是纯文本数字/符号序列，不包含 JSON 键名或括号。空白分隔，一个完整数字或符号占一个序列单元：

```text
EO2 DICTIONARY_ID GROUP_ID N_SITES N_EDGES RADIUS
L00 L01 L02 L10 L11 L12 L20 L21 L22
Z0 SITE_RULE_ID0 FREE_PARAMETER_0 ...
Z1 SITE_RULE_ID1 FREE_PARAMETER_0 ...
...
SOURCE TARGET OPERATION NA NB NC
SOURCE TARGET OPERATION NA NB NC
...
```

以上换行仅为解释，实际文件可为一行。头部指定记录数量，每个位点的自由参数数量由字典里的自由度确定，每条边固定 6 个整数，解析器因此不需要为每条记录插入额外分隔 token。

**结构字段数**为 `1 + 9 + 2*N_SITES + SUM_DOF + 6*N_EDGES`。

**完整序列长度**再加 5 个单元：`EO2`、字典版本、位点数量、边数量、近邻半径。group ID 已包含在结构字段数中，不能重复计数。

每个位点保存元素原子序数和共享规则 ID。固定的三维分量和相互关联的分量由约束计算，只携带真正独立的自由参数。晶格仍保存完整 3×3 矩阵，避免另行引入基底变换。

这里是结构字段 token/连续数值槽位的计数。浮点数保持原有精度，没有做有限词表量化，因此不能把本长度当作任意语言模型 BPE 的输入长度，也不代表已经训练了生成模型。离散结构部分与连续几何仍可分开建模。

## 命令

```bash
# 显式构建并冻结字典；普通 encode/decode 不修改字典。
python fixed_dictionary.py build-dictionary results/*/encoded.json --output rules.json

python fixed_dictionary.py encode input.cif encoded.tokens --dictionary rules.json
python fixed_dictionary.py decode encoded.tokens decoded.cif --dictionary rules.json
```

解码只需要 `.tokens` 与 `rules.json`。没有原 CIF 或任何按材料 ID 查找几何的数据库。

## 验证

`test_fixed_dictionary.py` 从原来固定的十个原始 CIF 开始，调用独立编码和解码进程，检查元素、晶格、周期坐标、边集合与再次编码的一致性。沿用严格对称识别容差 `1e-7 Å`，不理想化输入，不改变原来的样本或晶胞。

同时检查：材料顺序改变不改变字典；改变元素、连续几何或删除所有边不改变字典；未知规则与错误版本不能通过；整个往返过程中字典字节不变。

字典的文件大小应单独报告，并在评估整体存储时计入一次。把固定规则移到共享表可以减少每个材料的序列长度，但不能把字典成本说成不存在。
