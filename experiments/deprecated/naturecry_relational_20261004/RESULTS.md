# NatureCry 关系型受控自然语言：首版结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-04
- Verification Status: VERIFIED for the frozen offline cohort
- Version Label: relational_cnl_v1

## 结论

程序编码、程序解析的关系型英文表示已经实现。冻结队列先完成 30/30，
随后在相同源码哈希下完成 100/100。没有调用 LLM，没有训练，没有筛样、
替换输入或源数据修补。旧的 inline/LLM 实验及其失败记录保持不变。

这验证了数值与几何回环，不是自然度的人类评估，也不是任意英文的解码保证。

## 四项验收

| 检查 | 首批 30 | 完整 100 |
|---|---:|---:|
| 从英文恢复的构造计划与源 facts 逐项一致 | 30/30 | 100/100 |
| 重建 CIF 重新读取后与原结构严格匹配 | 30/30 | 100/100 |
| 文本回环后的标准化表达稳定 | 30/30 | 100/100 |
| 标准化文本再次解析仍得到同一计划 | 30/30 | 100/100 |

完整队列有 1,207 个原晶胞原子、513 个独立轨道实例、273 种实际使用的
空间群/位置定义。其中 71 个结构具有非恒等原始坐标系；17 个结构共有
114 个位点需要精度修正。这些情况全部保留并通过。

- 最大周期性、按元素匹配的坐标误差：`9.381384558082572e-16 angstrom`。
- 最大原晶胞晶格矩阵分量误差：`3.019806626980426e-14 angstrom`。
- 验收阈值分别为严格小于 `1e-6` 和 `1e-8 angstrom`。
- CIF 使用 14 位有效数字导出，再重新读取评分。
- API 请求与 LLM 调用均为 `0`。

源计划比较采用排序后的 JSON 数值序列化，既检查数值，也保留 `-0.0`
与 `0.0` 的区别。这项转录检查比最终几何容差更严格；没有声称几何计算
或 CIF 文本本身逐位无损。

## 为什么不是把字段包进英文

轨道文本不写 Wyckoff 标签，也不附带一个自由参数列表。解析器必须从
代表位置的平面约束、共享偏移、倍数及平移关系构造仿射关系，才能结合
空间群和标准晶胞轨道重数找回公共定义。关系句参与解码，不是标签旁边
可以忽略的解释。

完整英文文档还必须携带原晶胞尺寸、组成、原始到标准坐标系的变换、
舍入规则和必要的完整修正向量。解码入口只有英文文本及公共表，不读取
源 CIF、facts、oracle plan 或旧修正地址映射。保存的 plans/CIFs 是验收
产物，不作为解码的附加输入。

## 真实样例

下面是 `mp-11192` 的完整程序输出，不是手工润色的展示文本：

```text
The input cell edges a, b, and c measure 4.80623199, 4.80623199, and 6.604105 angstrom; the angles alpha, beta, and gamma measure 90.0, 90.0, and 119.99999990000002 degrees.
The public setting is space group 181, with literal Hall symbol P 64 2 (0 0 2).
The input cell contains 9 atoms in total.
Silicon contributes 6 atoms to the input composition.
Tantalum contributes 3 atoms to the input composition.
The standard and input fractional frames coincide.
The input-frame fractional coordinates are rounded to 6 decimal places before any listed corrections.
Tantalum forms a nominal 3-site symmetry orbit named o000 in the standard cell.
Its representative lies in the plane at fractional a offset 0.5.
Its representative lies in the plane at fractional b offset 0.0.
Its representative lies in the plane at fractional c offset 0.0.
Silicon forms a nominal 6-site symmetry orbit named o001 in the standard cell.
Along a, its representative has the free fractional offset 0.15910699999999997.
Along b, its representative doubles the offset along a.
Its representative lies in the plane at fractional c offset 0.0.
No input-frame fractional site corrections are applied.
```

这里的 Si 关系恢复为 `(x, 2x, 0)`。固定平面句不等于所有展开原子都在
同一个平面，描述对象是单个代表位置。轨道重数属于标准晶胞，而不是
一律等于原晶胞中该元素的原子数。

另两个完整样例：

- [mp-18127：半周期平移、非恒等坐标系及两项修正](/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM/experiments/naturecry_relational_20261004/run100_001/texts/mp-18127.txt)。Ni 的关系为 `(x, x+1/2, z)`，Tin 的关系为 `(x, x+1/2, 0)`。
- [mp-567772：三个坐标共用一个偏移](/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM/experiments/naturecry_relational_20261004/run100_001/texts/mp-567772.txt)。Na 的关系为 `(x, x, x)`；标准轨道有 16 个位置，原晶胞有 8 个原子。

这些句子是规范的科学构造语言，仍然偏技术性。复杂坐标系的完整高精度
系数会使文字较长；当前没有为了自然度而把这些值理想化或删除。

## 语法可逆性的依据

对哈希冻结的公共表独立审计得到：230 个空间群、1,731 个位置定义；
所有自由轴都有可直接绑定的单位坐标行；每行最多一个变量项；同一空间群
内 `(重数, 代表位置 A, 代表位置 b)` 唯一。共出现 25 种坐标行形式。

因此逆过程不需要浮点拟合代表位置：读取独立数值，解析其余关系形成
`A` 和 `b`，选择唯一公共定义，恢复自由参数。对合法有限数值，十进制
`repr` 和浮点解析负责数值转录。文本允许 `shares` / `matches` 等有限的
同义表达，回环后统一到标准表达，不要求保留原措辞。

11 项测试全部通过。其中一个紧凑的闭包测试遍历全部 1,731 个定义，
验证编码、解析及标准化回环。该测试使用合成构造计划，不是 1,731 个
真实晶体的物理有效性实验。测试报告有 9,434 项来自 spglib 的既有
弃用警告，没有测试失败。

审查中修复并测试了：关系重复/改变不能被忽略、缺失修正状态不能被当作
空列表、恒等坐标系短句不能抹去负零、运行途中变动源码的 pilot 不能
放行 100-case。使用“名义对称轨道”区分初始对称构造与修正后的真实结构。
独立源码审查没有剩余关键发现；主智能体还独立从文本重建并核对了
倍数、平移、共享、修正及负零坐标系对应的五个真实样例。

独立产物审计随后重新读取了全部 100 对原始/重建 CIF，100/100 再次通过，
最大误差与运行摘要完全一致。全部文本重新解析后与保存计划一致，
原始/重建 CIF、文本及 facts 的哈希各验证 100/100，11 项源码快照与
当前文件一致。计划文件没有独立的逐项 SHA 字段，因此通过重新解析
文本及比较保存计划、二次计划进行核验。参考计划的后验对照属于验收，
不是解码输入。

## 范围与溯源

这轮 100/100 支持的是未改变的冻结队列及当前受控语法。没有证明任意
自由英文、任意模型生成文本、部分占位/无序结构或任意 CIF 属性可逆。
编解码协议保存的是约定范围内的晶体几何，不是 CIF 注释或字节级文件。
从任意原始结构提取事实也不因语法闭包通过而获得无条件成功保证。

selection SHA-256：
`03527304a32e11c15414e0f368e72c694e754b3add76f660701437b6bf22d54a`。

编码器 SHA-256：
`365f122905399a73477b92a7af97c05b78483ea5af12347f9a6050b1a3506813`。

执行器 SHA-256：
`3d329b9a486bb4a2cddf90e0977468034080db7d03251082836b7d9e9a4c4cdd`。

`pilot30_001` 是已通过的早期运行；补齐 CIF/仿射解析函数的源码哈希后，
`pilot30_002` 重跑通过，才作为 `run100_001` 的正式门槛。两次 pilot
均保留，不覆盖。正式 pilot 与完整运行的跟踪源码哈希在开始/结束时一致。

原始文本、计划、重建 CIF、二次计划、全部记录及源码快照在
`run100_001` 中；报告数字直接对应其中的 `summary.json` 和 `records.json`。

## 重现命令

工作目录为项目根。输出目录必须使用新的名称，避免覆盖记录。

```bash
rtk proxy research/plaid_20260930/.venv/bin/python -m pytest experiments/naturecry_relational_20261004 -q --disable-warnings
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_relational_20261004/run_roundtrip.py --limit 30 --output experiments/naturecry_relational_20261004/pilot30_reproduce
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_relational_20261004/run_roundtrip.py --limit 100 --pilot experiments/naturecry_relational_20261004/pilot30_reproduce --output experiments/naturecry_relational_20261004/run100_reproduce
```
