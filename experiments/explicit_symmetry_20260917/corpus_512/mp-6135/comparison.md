# Ba2PrIrO6 (mp-6135)

- `input.cif`：原输入。
- `encoded.sym`：我们的完整显式位点约束程序。
- `decoded.cif`：我们的解码结果。
- `wyckoff_standard.tokens`：标准晶胞的 SG / Hall 设置 / 元素 / Wyckoff 标签 / 独立自由参数 / 晶格数。该版本是理想化的常规晶胞参考，不负责逐数值还原原晶胞。
- `wyckoff.tokens`：附带原晶胞变换、输出精度、必要残差的 Wyckoff 对照，可还原原始晶胞数值。
- `wyckoff_decoded.cif`：上述可还原版的独立格式解码结果。
- `space_group_comparison.json`：原 CIF 与我们的解码 CIF 在五种相同容差下重新识别空间群。
- `representation_comparison.json`：参数含义、长度、对照解码误差和空间群。

Token 数：CIF 108；显式约束 54；标准 Wyckoff 24；可还原 Wyckoff 41。

## Wyckoff 参数对应的规则（说明文本，不计入标签版输入）

| 元素 | WP | 参数轴 | 数值 | 查表得到的代表坐标 |
|---|---|---|---|---|
| Ba | 8c | 无 | 无 | 1/4, 1/4, 1/4 |
| Pr | 4a | 无 | 无 | 0, 0, 0 |
| Ir | 4b | 无 | 无 | 1/2, 1/2, 1/2 |
| O | 24e | x | 0.269983 | x, 0, 0 |

同名 WP 需要结合空间群与设置解释。我们的公式所选轨道代表可能与表中标准代表不同，但按完整空间群展开后应描述同一组位置。
