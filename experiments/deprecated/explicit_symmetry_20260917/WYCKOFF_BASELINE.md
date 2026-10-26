# Wyckoff 标签对照与空间群复核

这个对照使用传统的“空间群/设置＋元素＋Wyckoff 标签＋独立自由坐标＋晶格参数”。规则通过固定公开的 PyXtal 参考表查得，不把公式放入标签版 token 序列。它是表示层对照，不是对某篇生成模型的复现。

## 每个材料的文件

- `input.cif`：原始结构。
- `encoded.sym` / `decoded.cif`：我们的显式约束编码与解码结果。
- `wyckoff_standard.tokens`：标准设置、理想化常规晶胞的紧凑 Wyckoff 表示。
- `wyckoff.tokens`：可还原原始晶胞的 Wyckoff 对照，显式记录所需 FRAME、ROUND、RESIDUALS。
- `wyckoff_decoded.cif`：可还原版的解码结果。
- `space_group_comparison.json`：在五种相同容差下，从原始和我们的解码 CIF 坐标重新识别出的空间群。
- `representation_comparison.json` / `comparison.md`：长度、自由参数含义、标准代表坐标公式和验证结果。说明中的公式不属于标签版模型输入。

所有数据采用同一 512 材料清单，失败不替换。默认对照的空间群识别容差为 1e-3 Å。

## 紧凑标准版

```text
WYCKOFF_STD1
SG 180
HALL 474
CELL a b c alpha beta gamma
SITES 3
Yb 3d
B 3c
Pt 6j 0.147658
END
```

每行先给元素和 WP 标签，再按固定表中 x/y/z 的独立参数顺序给数值。零自由度位置没有参数。标签前的重数是标准常规晶胞中的重数，例如 KrF₂ 的标准胞含 2 个 Kr 和 4 个 F，原文件可能使用只有 3 个原子的原胞。

标准晶胞经过 spglib 理想化，因此该版本可以比较常见标签表示的紧凑程度，但不能直接声称逐数值保留原 CIF。

## 可还原版

```text
WYCKOFF1
SG ...
HALL ...
CELL ...
FRAME P00 ... P22 p0 p1 p2
ROUND ...
SITES ...
element multiplicity-letter free_parameters...
RESIDUALS ...
atom_index dx dy dz
END
```

`CELL` 保留原输入晶格。可选 `FRAME` 使用 spglib 约定 `q_standard=P*q_input+p`，先展开标准 WP，再转换回原胞，必要时复制标准胞像或去重。输出精度和残差还原原始数值。FRAME 是完整的 9+3 数值布局，没有进一步做稀疏压缩，本对照不声称是全局最短的 Wyckoff 序列。

Decoder 只读取上述序列和固定参考表，不读取原 CIF 或按材料 ID 查询数据。512 个可还原版均已完成晶格、组成、周期坐标和重识别空间群检查。

```bash
python wyckoff_baseline.py encode corpus_512/mp-1095422/input.cif example.tokens
python wyckoff_baseline.py decode example.tokens reconstructed.cif
```

依赖与此前显式程序相同，使用 Python 3.10，版本见 requirements.txt。固定表文件为 `fixtures/wyckoff_list.csv`，SHA256 为 `ca56602252b7896121253c910739ef2fccf0ba947f2940358feb941f30c5e13d`。代码先将每组表的完整操作集匹配至准确 Hall 设置，避免不同原点/基底下同名 WP 的歧义。

## 空间群编号的解释

CIF 文件头写 P1，可能只是因为把全部原子显式列出来。应对结构坐标重新识别，并使用相同 spglib 版本、距离容差、角度设置比较前后。本次检查使用 spglib 2.6.0，angle_tolerance=-1，symprec=1e-7/1e-5/1e-3/1e-2/1e-1 Å。

## 比较的正确范围

Wyckoff 标签本身已经通过空间群和固定表完整定义对称规则，LLM 也可以学习标签及其上下文。显式程序的区别是把固定/自由坐标、坐标关系和生成操作直接呈现在输入中，方便共享算术与几何结构。它没有额外增加理论上的结构信息，不保证更短，也还没有验证学习或生成优势。若给 Wyckoff 基线加入位点对称、自由度或规则表特征，两种模型输入的差别会缩小。

Token 计数仍为每个数值/标签/控制符一个单元，连续值未离散化，不是通用 BPE token 数。真正的模型对照需要匹配数据、骨干、训练预算与数值预测机制。
