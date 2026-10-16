# NatureCry v0.1：100 个 case 编解码结果

2026-10-01。已完成固定句式的英文 NatureCry 编解码和两轮 CIF 往返检查。

后续定位更新：用户要求由 LLM 生成自然的空间描述，并从语义恢复结构。本文件的结果保留为固定格式程序基线，不作为目标 LLM 方法的验证。新设计见 [NatureCry LLM 编解码 v0.2](../../research/NatureCry_LLM_codec_design_v0_2.md)。

## 结果

**100/100 通过两轮结构恢复检查，失败 0 个。**

| 检查 | 实测 |
|---|---:|
| 第一轮：CIF → NatureCry → 独立 decoder → CIF | 100/100 |
| 第二轮：输出 CIF → NatureCry → 独立 decoder → CIF | 100/100 |
| 原子数、元素计数一致 | 100/100 |
| 两轮最大周期坐标误差 | 9.3814e-16 Å |
| 两轮最大晶格矩阵分量误差 | 3.0198e-14 Å |
| 第二次编码文本与第一次逐字相同 | 29/100 |
| 需要输入晶胞/标准晶胞坐标转换的 case | 71/100 |
| 需要逐原子精度修正的 case | 17/100，共 114 个原子 |

判据：坐标误差 `< 1e-6 Å`，晶格矩阵分量误差 `< 1e-8 Å`，原子数和元素计数完全相同。
比较采用同元素原子的周期性一一匹配，保持输入晶胞，不允许晶格缩放。

这证明当前 100 个样本在本版固定句式、固定晶体学表和已保存精度信息下能恢复原始结构数据。
这是程序编解码结果，不是 LLM 生成或自然语言理解实验。

## 样本与覆盖

两个作者公开生成样例：Tb₈Mg₄、Li₈V₄Ni₄O₁₆。
另外 98 个是真实 MP-20 训练集记录，沿用之前 tokenizer 测试的 CSV 版本。
先按材料 ID 的 SHA256 排序，再对每个尚未出现的数据集空间群取首个样本。选择在编码前冻结，失败不换样本。

训练 CSV SHA256：`133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d`。

98 个数据集空间群标签经统一 `symprec=1e-3 Å` 识别后，整个 100-case cohort 覆盖 96 个识别空间群和七个晶系。数据集标签与按输入数值识别的空间群并不总相同。

| 晶系 | case 数 |
|---|---:|
| 三斜 | 4 |
| 单斜 | 12 |
| 正交 | 27 |
| 四方 | 21 |
| 三方 | 12 |
| 六方 | 10 |
| 立方 | 14 |

这是一批强调空间群覆盖的样本，100% 是该固定 cohort 的通过率，不能当作整个数据集分布的无偏估计。

## 这一版 NatureCry 编码了什么

正文使用可读的英文句式，包含晶格、组成、原子数、Hall 设定、独立轨道、位点坐标关系、固定位置和自由参数。
例如氧的 `8i` 轨道写为：

> The orbit O5 is occupied by O with occupancy 1. Its Wyckoff position is 8i, with multiplicity 8. Its representative fractional coordinates follow (x, 1/4, z) in the standard frame. The independent parameters are x = 0.237, z = 0.9950000000000001.

decoder 校验轨道标签、多重性、坐标关系与参数名的一致性；读取正文中的数值，再交给已有 Wyckoff 几何后端展开结构。
它没有访问原始 CIF 或通过材料 ID 补齐坐标。两次独立 decoder 子进程都只收到自然语言文本目录和输出目录。

满占位和结构元数据范围已在 [PROTOCOL.md](PROTOCOL.md) 中声明。CIF 注释、原子标签、热参数等实验信息不在编码范围内。

## 精度修正的作用

自然语言中的对称轨道描述需要处理标准晶胞与输入晶胞的差异，也需要保留输入坐标相对于理想对称性骨架的小偏差。
本版把这些信息作为正文的一部分保存，没有隐藏旁路。

对使用精度修正的 17 个 case，另做了移除修正的对照：**17/17 都不再满足原定坐标误差标准**。
不带修正时的最大偏移为 `3.2696e-4 Å`。
因此，本轮严格保真依赖完整的晶格、轨道、坐标转换及精度修正信息；仅用理想化 Wyckoff 参数还不足以精确恢复这 17 个输入。

空间群约束描述对称性骨架，逐原子修正保留原始近似对称/有限精度几何；这不是声称输入几何数学上完全满足理想空间群。
原子的修正索引基于固定几何后端的展开顺序。后端和表的哈希保存在 summary.json 中。

## 文本大小与两轮文本差异

| 全部 100 个 case 合计 | UTF-8 字节数 |
|---|---:|
| 输入 CIF | 128,416 |
| NatureCry 文本 | 251,510 |

NatureCry 平均约 2,515 字节，是输入 CIF 总大小的约 1.96 倍。
这是文件字节比较，不是 LLM tokenizer 长度、信息论压缩率或训练效率。

两轮有 71 个 case 的正文并非逐字相同，但它们恢复的结构仍符合相同严格标准。
检查差异可见浮点字符串、对称等价代表点、坐标转换及精度修正记录变化。本版未建立唯一规范化文本。
因此，目前结论是“结构可逆”，不是“任意输入表示均产生唯一文本”。

## 最小额外检查

[test_codec.py](test_codec.py) 的检查已通过：

- 修改自然语言中的 a 边长，decoder 改变对应晶格。
- 修改自由坐标参数，decoder 改变对应原子轨道。
- 矛盾原子数、与 Wyckoff 标签冲突的坐标关系、额外尾随指令会被拒绝。

这些检查针对当前 grammar 的含义和一致性，不构成对自由改写、任意语言或模型生成描述的验证。

## 查看结果与复现

- [全部逐例记录](run_001/records.json)、[汇总和依赖哈希](run_001/summary.json)、[样本清单](run_001/selection.json)。
- [精度修正对照与覆盖统计](run_001/diagnostics.json)。
- Tb₈Mg₄：[NatureCry 正文](run_001/texts/anchor_Tb8Mg4.txt)、[重建 CIF](run_001/decoded/anchor_Tb8Mg4.cif)。
- Li₈V₄Ni₄O₁₆：[NatureCry 正文](run_001/texts/anchor_Li8V4Ni4O16.txt)、[重建 CIF](run_001/decoded/anchor_Li8V4Ni4O16.cif)。
- 目录 `run_001/inputs`、`texts`、`decoded`、`reencoded`、`decoded_second`、`audits` 保存全部中间结果。

在 CrystalDLM 根目录运行：

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 research/plaid_20260930/.venv/bin/python experiments/naturecry_roundtrip_20261001/test_codec.py
rtk proxy env PYTHONDONTWRITEBYTECODE=1 research/plaid_20260930/.venv/bin/python experiments/naturecry_roundtrip_20261001/benchmark.py --out experiments/naturecry_roundtrip_20261001/run_002
```

也可以单例调用 `naturecry_codec.py encode input.cif output.txt` 或 `decode input.txt output.cif`。
输出目录要求为新目录，避免覆盖已有证据。

## 当前边界

当前已验证：固定句式的受控英文、完整数值信息、100 个有序满占位样本的两轮结构恢复。
尚未验证：任意自然语言改写的解析、整个 MP-20/全部空间群覆盖、部分占据、LLM 学习收益和物理稳定性。
本轮到此结束，没有启动训练、性质评估或 AR＋diffusion 工作流。
