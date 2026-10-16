# 首轮单性质数据与评价依据：证据盘点

盘点日期：2026-10-11。范围限定为已有 MP—ehull 收集结果、PLaID 序列化/旧评价代码和最多两个公开替代性质；本报告提供数据与评价证据，不替用户选定性质。

## 已有数据：事实

| 阶段 | 数量和口径 | 本地证据 |
|---|---|---|
| 已有来源快照 | `groupdb.mp_api_all_2025_7` 共 154,879 条；记录版本全为 `2025.06.09`。 | [inventory receipt](../mp_ehull030_20261011/collection_001/inventory_receipt.json) |
| 元数据筛选 | 64,043 条满足原始 `energy_above_hull` 有限、0–0.3 eV/atom、非 deprecated、1–20 sites、1–4 元素；首个排除原因计数为原子数范围外 67,157、ehull 范围外 19,564、元素数范围外 4,115（按筛选顺序记首因）。 | [collection README](../mp_ehull030_20261011/collection_001/data_001/README.md)、[inventory receipt](../mp_ehull030_20261011/collection_001/inventory_receipt.json) |
| 结构纳入 | 64,043 条均导出并通过有序、完整占位和晶格/坐标检查；没有结构排除。 | [download receipt](../mp_ehull030_20261011/collection_001/download_receipt.json) |
| 最终抽样 | 45,629 条、88 种元素、178 个空间群；原子数 1–20、元素数 1–4。5 个 ehull 区间分别为 `0`: 10,000、`(0,0.05]`: 10,000、`(0.05,0.10]`: 10,000、`(0.10,0.20]`: 10,000、`(0.20,0.30]`: 5,629。 | [dataset README](../mp_ehull030_20261011/collection_001/data_001/README.md)、[sampling report](../mp_ehull030_20261011/collection_001/data_001/sampling_report.json) |
| 划分与完整性 | train/val/test 为 36,491/4,517/4,621。对 45,629 条做过结构与原标签逐条比对，全部通过；artifact hashes 全匹配。 | [manifest](../mp_ehull030_20261011/collection_001/data_001/manifest.json)、[verification](../mp_ehull030_20261011/collection_001/data_001/verification.json) |

区间等额封顶是有意的分层抽样，不能当作原始 MP 标签频率分布；`(0.20,0.30]` 未补满是因为候选不足。结构重复键仅覆盖同一晶胞下的平移/位点顺序不变情形，按 8 位小数去重，本次删去 1 条。它没有做跨晶胞、对称等价或近邻几何合并，也不按化学式分组；因此划分避免了该重复键跨 split，却不能排除多晶型、等价晶胞或近重复结构导致的泄漏。细节在 [collection code](../mp_ehull030_20261011/collect.py) 与 [README](../mp_ehull030_20261011/collection_001/data_001/README.md)。

**标签与配对事实。** 导出的是 MP 材料级同一文档中的 `structure` 和连续原始 `energy_above_hull`，单位 eV/atom；没有用预测字段替换标签，也没有重新算或舍入原标签。[finish.py](../mp_ehull030_20261011/finish.py) 保留快照、结构来源和 task-entry 类型信息。最终结构来源中，能与条目类型对应的有 GGA 23,576、GGA_U 7,942、R2SCAN 13,902；209 条没有 task-entry 匹配。快照没有显式 `thermo_type`，因此不能据这些计数断言全体标签都来自某一种凸包口径。三条官方 API 对照的值与快照分别只差 `1.25e-16` 至 `2.78e-16` eV/atom，并有旧 ID 到新 ID 的映射；这是三例来源佐证，不是全库计算溯源。见 [官方核对回执](../mp_ehull030_20261011/mp_official_three_check_001.json) 与 manifest 的 `limitations`。

MP 将 ehull 定义为相对于由形成能/组成构成的凸包的稳定性量；它是 0 K 计算指标，官方也提醒其数 meV 级计算精度、化学体系依赖性和混合计算口径问题。[MP glossary](https://docs.materialsproject.org/frequently-asked-questions/glossary-of-terms) 因而数据中的 0–0.3 和五个分层边界是采样范围，不等于跨化学体系通用的稳定性分类阈值。

## PLaID 表示对配对的影响：事实与未知

实际训练调用的 [PLaID Wyckoff 编码](../../work/PLaID/utils.py) 先以 `SpacegroupAnalyzer` 容差 0.01 精修结构，再用 PyXtal 提取 Wyckoff 表示；晶格长度和角度输出到 0.01，独立位点坐标输出到 0.001。另一旧版编码路径把坐标舍入到 0.01。解码路径由舍入后的参数重建晶格和对称位点。故标签仍准确对应原始 MP 结构记录，但不能直接宣称它就是精修、舍入或生成解码后结构的标签。

**未知：** 当前记录没有把原结构经此编码/解码后的能量或 ehull 与原标签作对照；也没有量化 0.01 晶格舍入、0.001 坐标舍入及空间群精修对标签的影响。不能据静态精度位数推算 ehull 偏移。若目标性质最终定为 ehull，训练数据应保留原始结构和原始标签；对生成/解码结构的分数必须作为新评价结果另行计算或校准。

## 最多两个替代性质：官方可取，当前快照未知

| 候选性质 | 数据可得性与优势 | 主要障碍 |
|---|---|---|
| Formation energy per atom | MP Summary API 支持 formation-energy 查询；官方口径为每原子标量。相比 ehull，不需要先确定竞争相形成的凸包才能定义该结构自身的形成能。 | 本次 `META_FIELDS`/`RAW_FIELDS` 和最终数据没有保留此字段，现有样本数、缺失率和分布均未知；生成结构的预测能量还需同一能量基准的元素参考与一致的 GGA/GGA+U 校正。 |
| Band gap | MP Summary API 支持 band-gap 查询，单位 eV；是一个容易写成短条件的标量。 | 本次快照未保留该字段，样本数、缺失率和分布未知。CHGNet 的官方输出是能量、力、应力和磁矩，不是电子能带或 band gap；需单独的电子结构计算或经过校准的 band-gap 模型。 |

上述字段存在于 MP 官方 [SummaryRester 查询 schema](https://materialsproject.github.io/api/_autosummary/mp_api.client.routes.materials.summary.SummaryRester.html)，但不能据此推断本地 archive 有数据，也没有为本报告查询或下载额外性质数据。

## 新生成结构的评价器：已知与未验证

ehull 不能靠查一个未知新结构的 MP material ID 得到；需要该结构的能量及同一化学体系中竞争相的兼容能量来构建/使用凸包。官方 `MPRester.get_stability` 接收计算条目并指定 `thermo_type`，默认混合 GGA/GGA+U/R2SCAN；MP 当前 [API 实现](https://materialsproject.github.io/api/_modules/mp_api/client/mprester.html)也说明不同体系要保持共同能量尺度。EHull 适合作为候选，是因为现有标签已配对且 PLaID 旧代码曾有稳定性管线，不代表新结构评价现成可用。

本地 [旧 PLaID 评价代码](../../work/PLaID/evals/e_above_hull.py)有“生成 CIF → MLIP 弛豫 → `ComputedStructureEntry` → `PatchedPhaseDiagram`”的框架，但当前受检路径不构成已验证的评价器：`relaxations.py` 中 M3GNet/CHGNet 单结构函数返回 `(0, None)`，批量 CHGNet 返回空表且未加载；EQv2 路径需要未在该目录找到的 checkpoint。凸包函数还依赖未找到的 `computed-structure-entries.json.gz`，并从 entry ID 含 GGA、无 R2SCAN 校正的条目构建旧式 hull；没有证据证明其能量尺度与本次 thermo_type 缺失的 2025 快照标签一致。该代码的存在不是模型成功运行或物理验证结果。

官方 [CHGNet 实现](https://github.com/CederGroupHub/chgnet)能对输入结构预测能量、力、应力和磁矩，并可做结构优化；其预训练数据是 2022 年 9 月 MP 的 GGA/GGA+U 轨迹。因此它可作为潜在的快速能量模型，却不直接给 ehull；把它用于从 MP 训练/测试集上的评分还存在同源预训练泄漏风险，精确重合数未知，不能视为独立外部验证。EQv2/其他 MLIP 的推理和弛豫速度、能量校准、以及本项目内 48 A100-GPU 小时能覆盖的样本量均未测量；DFT 级评估预算也未核算。

## 建议、估计与下一决策

- **建议：** 暂将 ehull 与 formation energy 留作 shortlist。ehull 是目前唯一已导出并验证的目标标签；formation energy 可能更直接连接每结构能量评价，但要先确认本地数据覆盖和能量基准。band gap 先排在后面，因为现有快速结构势模型不能直接产出它。
- **建议：** 用户选定性质后，再明确“目标条件”的数学语义（连续值、区间或上限）、单位/容差和是否把结构弛豫后数值作为评价值；在此之前不固定稳定阈值、文字模板或性质。
- **主要障碍：** ehull 的 archive 凸包口径缺失；PLaID 表示对标签的影响未量化；代码中评价器资产缺失且 MLIP 与 MP hull 的能量兼容性未验证；当前 split 不是对称等价/组成隔离 split；预算可行性未知。
- **未知：** 两个替代性质在 64,043 条轻筛记录或 45,629 条最终样本中的覆盖数量；PLaID 变换前后目标值偏差；独立外测中 predictor 的误差/域外率；每结构评价耗时及 48 GPU 小时覆盖量。

## 来源与本地证据索引

- 官方： [MP stability glossary](https://docs.materialsproject.org/frequently-asked-questions/glossary-of-terms)、[MP SummaryRester schema](https://materialsproject.github.io/api/_autosummary/mp_api.client.routes.materials.summary.SummaryRester.html)、[MP stability API implementation](https://materialsproject.github.io/api/_modules/mp_api/client/mprester.html)、[CHGNet official implementation](https://github.com/CederGroupHub/chgnet)、[PLaID++ official implementation](https://github.com/andaero/PLaID)。
- 数据： [collection plan](../mp_ehull030_collection_plan_20261011.md)、[inventory/download receipts](../mp_ehull030_20261011/collection_001/inventory_receipt.json)、[final README](../mp_ehull030_20261011/collection_001/data_001/README.md)、[manifest](../mp_ehull030_20261011/collection_001/data_001/manifest.json)、[verification](../mp_ehull030_20261011/collection_001/data_001/verification.json)、[official three-record spot check](../mp_ehull030_20261011/mp_official_three_check_001.json)。
- 表示/评价代码： [PLaID utilities](../../work/PLaID/utils.py)、[PLaID evaluation pipeline](../../work/PLaID/evals/e_above_hull.py)、[relaxation implementations](../../work/PLaID/evals/relaxations.py)。

未访问数据库、未批量下载、未调用付费 API、未读取凭据、未加载权重，也未执行几何、训练或 HPC 计算；替代性质统计、精修误差和评价耗时仍需另行验证。
