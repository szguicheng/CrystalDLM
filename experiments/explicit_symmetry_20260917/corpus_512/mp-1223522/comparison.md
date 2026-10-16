# KFeSeS (mp-1223522)

- `input.cif`：原输入。
- `encoded.sym`：我们的完整显式位点约束程序。
- `decoded.cif`：我们的解码结果。
- `wyckoff_standard.tokens`：标准晶胞的 SG / Hall 设置 / 元素 / Wyckoff 标签 / 独立自由参数 / 晶格数。该版本是理想化的常规晶胞参考，不负责逐数值还原原晶胞。
- `wyckoff.tokens`：附带原晶胞变换、输出精度、必要残差的 Wyckoff 对照，可还原原始晶胞数值。
- `wyckoff_decoded.cif`：上述可还原版的独立格式解码结果。
- `space_group_comparison.json`：原 CIF 与我们的解码 CIF 在五种相同容差下重新识别空间群。
- `representation_comparison.json`：参数含义、长度、对照解码误差和空间群。

Token 数：CIF 94；显式约束 61；标准 Wyckoff 37；可还原 Wyckoff 54。

## Wyckoff 参数对应的规则（说明文本，不计入标签版输入）

| 元素 | WP | 参数轴 | 数值 | 查表得到的代表坐标 |
|---|---|---|---|---|
| K | 2b | y | 0.358072 | 0, y, 1/2 |
| K | 2a | y | 0.638718 | 0, y, 0 |
| Fe | 2b | y | 0.003281 | 0, y, 1/2 |
| Fe | 2a | y | 0.005541 | 0, y, 0 |
| Se | 4c | x, y, z | 0.199853, 0.888783, 0.852924 | x, y, z |
| S | 4c | x, y, z | 0.1805735, 0.1084115, 0.343155 | x, y, z |

同名 WP 需要结合空间群与设置解释。我们的公式所选轨道代表可能与表中标准代表不同，但按完整空间群展开后应描述同一组位置。
