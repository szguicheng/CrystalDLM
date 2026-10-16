# PLaID++：Wyckoff 文本表示与解码核查

核查日期：2026-09-30。本文由 AI 辅助阅读论文、检查源码和运行局部编解码验证形成；没有训练或运行 PLaID 模型，没有复现 S.U.N. 或 DFT 结果。

## 版本与证据

- 正式发表：ICML 2026 / PMLR 306，140903–140922。
- 正式论文入口：https://proceedings.mlr.press/v306/xu26l.html
- 本地论文：`plaid_icml2026.pdf`，20 页。已阅读正文、实验细节、表示扩展分析、数学附录及生成样例；Figure 2 和 Table 7 另作页面渲染核对。
- 官方仓库：https://github.com/andaero/PLaID
- 克隆目录：`../../work/PLaID`。
- 固定源码提交：`9ac91f07a263594bbaef1574efbf483a98152f0c`，提交日期 2026-04-23。
- 论文正式版与公开代码快照的时间不同；下文区分论文表述与当前代码行为。

## 直接结论

PLaID 已覆盖“预训练 LLM 生成 Wyckoff 文本，再转回晶体结构”这条路径。
其输入 prompt 使用自然语言；其结构输出使用固定格式的结构化文本。
源码未把轨道等价关系、位点对称性或自由度展开为完整自然语言句子。
因此，它非常接近我们讨论的路线，但不能把“文本格式”直接等同于“结构自然语义化”。

## CIF 如何转成训练文本

实际入口为 `utils.py:get_crystal_string_wyckoff_pyx`（11–66 行），由 `llm_finetune.py` 的数据缓存与 `crystal_string` 调用；另一个 `get_crystal_string_wyckoff` 不是这条 SFT 路径的入口。

1. `Structure.from_str(..., fmt='cif')` 读取原始结构。
2. `SpacegroupAnalyzer(structure, tol=0.01).get_refined_structure()` 精修、标准化结构。
3. `pyxtal.from_seed(..., tol=0.01)` 提取空间群、晶格、非等价轨道；失败时尝试 `0.0001`。
4. 生成五行头部：晶胞组成、空间群符号、三条边长、三个角度、晶胞原子总数。
5. 遍历 `pyx.atom_sites`，每个非等价轨道输出一行 `element x y z multiplicity+letter`。

边长和角度保留两位小数，分数坐标保留三位小数。保留的是代表点的三维坐标，不是显式列出的最小自由参数向量。`Sites (12)` 的 12 是晶胞总原子数，可能只有三条轨道记录。

这里已经存在精修与数值舍入，所以不是原始 CIF 的逐数值无损编码。
源码 51 行关于“lossless structure matching”的注释应理解为容差内匹配，不能当作逐位无损的证明。

## 实际样例 A：作者公开的模型生成结果

来源：`results/qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.3_esen_ehull_results_combined_sun.csv`，零起算第 0 条数据记录，`gen_str` 列。
这里只去掉第一行自然语言 prompt，结构正文没有改写。

```text
Tb8Mg4
Spacegroup: Pnma
abc: 7.11 5.33 9.75
angles: 90.00 90.00 90.00
Sites (12)
Tb 0.818 0.250 0.084 4c
Tb 0.467 0.250 0.728 4c
Mg 0.247 0.250 0.124 4c
```

两个 Tb 轨道分别生成 4 个 Tb；一个 Mg 轨道生成 4 个 Mg。因此三行记录描述了 12 个原子。
`Tb8Mg4` 是晶胞组成，不是最简式 Tb2Mg。
完整 prompt 保存在 `released_0_full_prompt.txt`，结构文本和重建结果分别在 `released_0.txt`、`released_0_decoded.cif`、`released_0.vasp`。

## 实际样例 B：作者训练数据通过原版 encoder 得到的文本

来源：`data/basic/train.csv`，零起算第 19132 条记录，`material_id=mp-2657`，原始最简式 TiO2。
这不是本轮模型推理的输出，而是用官方 encoder 对真实训练 CIF 做转换。

```text
Ti2O4
Spacegroup: P42/mnm
abc: 4.65 4.65 2.97
angles: 90.00 90.00 90.00
Sites (6)
Ti 0.000 0.000 0.000 2a
O 0.305 0.305 0.000 4f
```

两条轨道记录生成 2 个 Ti 和 4 个 O。
原始边长为 `(4.653272, 4.653272, 2.969203)` Å，解码后的边长为 `(4.65, 4.65, 2.97)` Å。
它通过常规结构匹配，但没有通过本轮更严格的匹配阈值，直接说明舍入造成了差异。

## 正式论文自带的条件生成例子

Appendix 7.9 / Table 7 给出了三个完整例子：

- 无条件：Ho4Al4Ni4 / Pnma。
- prompt 指定空间群 15：Mn8O12F4 / C2/c。
- prompt 指定空间群 216：Ti4Ni4Sn4 / F-43m。

Table 7 的截图已核对，保存在 `table7_page19.png`；文本位于本地 PDF 抽取文件。
例如第三个输出有 `Ti (... ) 4b`、`Ni (... ) 4d`、`Sn (... ) 4a` 三条记录，生成 12 个原子。

## Decoder 具体做什么

实际入口：`utils.py:parse_fn_wyckoff`（104–170 行）；采样脚本的 `process_single_item` 调用它。

1. 按行、冒号与空格解析空间群、晶格和位点。
2. `Group(spacegroup_symbol).number` 将空间群符号映射到空间群号。
3. 每条位点记录解析元素、代表坐标和 Wyckoff 标签。
4. 从标签前缀读取多重性，按元素累加 `numIons`。
5. 用 `Lattice.from_para` 构造晶格。
6. 调用 `pyxtal.from_random(..., lattice=..., sites=..., max_count=1)`；这里传入了具体位点及坐标，并非完全无条件随机摆放原子。
7. `to_pymatgen()` 得到完整结构，再 `to(fmt='cif')` 输出 CIF。

这是 parser 加晶体学构造器，不是另一个学习式自然语言 decoder。
固定行位置与字段格式意味着它没有承诺支持任意同义改写、句子重排或散文式结构描述。

代码没有读取/校验第一行组成，也没有校验第五行 `Sites` 数值；真实组成由位点记录推算。
本轮将样例 A 的两行改为 `H999` 和 `Sites (999)`，原 decoder 仍产生同一结构。
这是实现观察，不意味着所有生成结果都存在该矛盾。

采样脚本 `llm_sample.py` 的无条件循环会丢弃解析失败项并重采样，直到得到目标数量的可解析结构。
因此复现时应区分原始尝试数、解析成功数、结构有效数与最终 S.U.N.；不能从保存 CSV 的大小推导第一次输出的解析率。

## 模型与训练关系

- 模型是预训练 Qwen-2.5 7B，自回归 next-token 生成；不是 masked DLM。
- 使用原生 tokenizer 对普通文本、数字、元素和 Wyckoff 标签分词；当前代码没有添加 Wyckoff 专用词表。
- SFT 包含约 2/3 生成任务、1/3 元素 infilling 任务。
- 论文列出了组成、形成能、带隙、凸包能和空间群等 prompt 属性；本次代码快照的 `all_attributes` 实际列出带隙、凸包能、空间群，且选择属性时加入组成，bulk modulus 有单独分支。
- 后续迭代 DPO 用 eqV2 构造偏好，用 eSEN 进行主要评估，并逐轮调整采样温度。
- 正式论文 Table 1 的 SFT Wyckoff、DPO Wyckoff、旗舰版本 S.U.N. 分别为 3.58%、6.25%、7.74%。这是作者报告，未在本轮复现。
- “只改表示”不能解释全部旗舰提升；稳定性/新颖性偏好、联合任务训练和温度调度也有贡献。
- 文中 RL 阶段未优化的空间群可能已出现在 SFT 数据中；不能自动称为训练完全未见的空间群泛化。

## 本轮运行验证

未修改官方源码。使用独立 `.venv`，仅安装编解码所需依赖。
旧 `requirements.txt` 的 PyXtal 1.0.6 对 tuple 位点参数在 `_check_consistency` 报错；按 `pyproject.toml` 的下限改用 1.0.8 后通过。其他实际版本见 `verification.json`。

| 样例 | 来源 | 轨道记录 | 原子数 | 常规匹配 | 更严格匹配 |
|---|---|---:|---:|---|---|
| Tb8Mg4 | 发布生成 CSV，第 0 条 | 3 | 12 | 通过 | 通过 |
| Dy4Al4Ni4 | 发布生成 CSV，第 1 条 | 3 | 12 | 通过 | 通过 |
| Ti2O4 / mp-2657 | 训练 CIF 经 encoder | 2 | 6 | 通过 | 未通过 |

常规匹配使用 `StructureMatcher(scale=False)` 的其余默认阈值。
严格匹配为 `ltol=1e-4, stol=1e-4, angle_tol=1e-3, scale=False`；依然是匹配测试，不是逐位相等。
结果 JSON、源文本、输入 CIF、解码 CIF 和 POSCAR 均已保存。

运行方式（在 CrystalDLM 根目录）：

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 research/plaid_20260930/.venv/bin/python research/plaid_20260930/reproduce_examples.py
```

## 从这篇论文出发的候选问题

以下是相对 PLaID 的差异假设，不是全领域创新性认证。

最贴近老师想法的问题：把标签的关系与自由度写成可执行的受控自然语言，是否能在同一预训练模型上提高语义约束理解、组合泛化或可控编辑？
例如原始 `O 0.305 0.305 0.000 4f` 可以表达为：“In space group 136, oxygen occupies the four-site 4f orbit. Its representative fractional position is (u, u, 0), with u = 0.305.” 这是我们提出的表达，不是 PLaID 原文。
这里真正要验证的是模型能否理解并执行 `(u,u,0)` 的关系、自由度及相关修改，单纯将字段换成英语句子不构成有力贡献。

| 已有能力 | 可以进一步检验的差异 |
|---|---|
| 空间群和标量属性的模板式条件生成 | 同义改写、多个结构关系约束的组合与未见组合泛化 |
| 元素 infilling | 在保持其余结构不变的条件下修改自由参数、指定轨道或晶格方向 |
| Wyckoff 固定格式编解码 | 可逆的自然语言语义解析与结构一致性检查 |

已有 infilling 与条件生成应保留为基线；不能把它们单独重新包装为创新。
如果比较表示，应固定预训练模型、数据划分、物理后处理与评估器，同时报告 token 长度和计算成本。
decoder 的约束能力与语言表示带来的学习收益需要分开评估。
下一轮应先选一个明确机制，再设计实验；本轮未启动训练、采样或新的科研实验。
