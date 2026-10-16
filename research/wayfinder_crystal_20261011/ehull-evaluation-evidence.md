# PLaID 解码晶体的 ehull 评价路线：证据与后续门槛

研究日期：2026-10-11。只核查本地代码、已有 MP 标签和公开一手资料；本票没有下载模型、运行能量计算/弛豫或执行训练。

## 结论

PLaID 代码中有稳定性评分流程，但当前工作区不具备可复现所需资产，且其 MLIP 能量与 MP 凸包的兼容性没有成立证据。现阶段保留两条路线：PLaID 的 MLIP 流程只能作为待校准的代理分数；若要把结果解释为与已有 MP 标签同口径的 ehull，优先核实并采用 MP 设置的 VASP/DFT 能量和同一 thermo_type 的完整参考凸包。第二条更便于解释计算口径，但仍有 DFT 方法误差且可能昂贵，耗时未测，不能据此承诺 48 GPU 小时足够。

## 可确认的本地数据与表示边界

当前数据集含 45,629 个结构及未重算、未舍入的原始 `energy_above_hull` 标签，单位 eV/atom。来源 task-entry 匹配计数为 GGA 23,576、GGA_U 7,942、R2SCAN 13,902，另有 209 条未匹配；归档没有显式 `thermo_type`。因此 task-entry 类型不能代替各标签凸包的 thermo type。[数据证据](data-evidence.md#标签与配对事实)；[归档清单 limitations](../mp_ehull030_20261011/collection_001/data_001/manifest.json#L119-L122)。

PLaID Wyckoff 路径用 0.01 容差精修结构，晶格参数写到 0.01、独立位点坐标写到 0.001；旧路径把坐标写到 0.01。[本地编码器](../../work/PLaID/utils.py#L11-L66)。故原始 MP 标签对应数据库原结构，不能直接赋给 `encode → decode` 重建出的结构。旧评价主流程也只把 `relaxed_cif` 与 `relaxed_energy` 送去算 hull，没有同时保存未弛豫结构的评分。[本地评分入口](../../work/PLaID/evals/e_above_hull.py#L89-L119)；[EQv2 弛豫路径](../../work/PLaID/evals/relaxations.py#L33-L54)。

## 两条候选路线

| 路线 | 实际做法与所需资产 | 当前判断 |
|---|---|---|
| **1. PLaID MLIP 弛豫 + 相图** | 官方代码用 EQv2 或 eSEN 经 FIRE 与 FrechetCellFilter 同时弛豫晶格和原子，再把模型总能量包装为 `ComputedStructureEntry`，调用 `MaterialsProject2020Compatibility`，最后与 GGA(+U) 条目建立的 `PatchedPhaseDiagram` 比较。[上游评分代码](https://github.com/andaero/PLaID/blob/main/evals/e_above_hull.py#L211-L280)；[弛豫实现](https://github.com/andaero/PLaID/blob/main/evals/relaxations.py#L33-L54) | PLaID 引用的 EQv2/ESEN OAM 权重以 OMat 预训练，再以 MPtrj + sAlex 微调；[FairChem 官方模型卡](https://huggingface.co/facebook/OMAT24)列出这组权重，并说明 OMat 训练能量与 MP 计算不兼容、跨体系能量差需谨慎。模型卡将“不要使用 MP2020 修正或 MP 元素参考”的明确指示限定在 OMat-only checkpoint；它没有证明 OAM 权重与 MP2020 修正已兼容。PLaID 代码把 OAM 型 MLIP 能量套入 MP2020 修正，故不能据代码存在就称为兼容 MP hull。MPtrj 又包含 MP 的 GGA/GGA+U 轨迹，故在现有 MP 样本上有预训练重合风险；精确结构/ID 重合数未知。[MPtrj 官方说明](https://github.com/CederGroupHub/chgnet#mptrj-dataset)。适合作为待校准的 MLIP 代理分数，不直接替代现有 MP 标签。 |
| **2. MP 设置的 VASP/DFT + 同口径参考 hull** | 对候选结构计算 VASP 能量，用与 MP 输入设置相容的能量修正，再把候选项放入覆盖其完整元素体系及所有子体系的兼容参考相凸包。PLaID 有类似后处理：读取 VASP `vasprun.xml` 最终能量，应用 MP2020 修正，并加载 `mp_02072023/2023-02-07-ppd-mp.pkl.gz`。[官方 DFT 后处理](https://github.com/andaero/PLaID/blob/main/dft/ehull_correction_newest.py#L22-L23)、[修正与条目构造](https://github.com/andaero/PLaID/blob/main/dft/ehull_correction_newest.py#L52-L72)、[凸包加载与评分](https://github.com/andaero/PLaID/blob/main/dft/ehull_correction_newest.py#L162-L195)。 | 这是让新候选 ehull 接近 MP 口径的优先路线，但旧 PLaID hull 是 2023 年资产；本地训练标签来自 2025.06.09，且 thermo_type 不明，二者不能自动视为同一口径。Pymatgen 文档限定 `MaterialsProject2020Compatibility` 用于 MP 参数的 VASP 计算；MP API 则说明应按指定 thermo_type 和同一化学体系构建兼容 entries。[Pymatgen 文档](https://pymatgen.org/pymatgen.analysis.compatibility.html#pymatgen.entries.compatibility.MaterialsProject2020Compatibility)、[MP get_stability 与 get_entries_in_chemsys](https://materialsproject.github.io/api/_autosummary/mp_api.client.mprester.MPRester.html)、[MP 混合方法](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability/gga-gga%2Bu-r2scan-mixing)。若不能恢复匹配的 thermo_type、修正与完整竞争相集合，应暂停并报告“口径未知”。 |

**资产现状（本地只读核查）：** `work/PLaID/evals/OCPCalcs/` 下 EQv2 与 eSEN checkpoint、`work/PLaID/computed-structure-entries.json.gz`、`work/PLaID/mp_02072023/` 相图目录均不存在。官方 GitHub [Releases](https://github.com/andaero/PLaID/releases) 当前没有发布包；官方 README 将 PLaID 语言模型权重指向 [Hugging Face](https://huggingface.co/HOPE-Lab-HMC/PLaID/tree/main/train_dpo)，并说明 ehull 评估所需数据需按论文另行取得。语言模型权重不等于 MLIP checkpoint 或竞争相凸包资产。[官方 README](https://github.com/andaero/PLaID#data)。

## 最小后续核验设计（本票未执行）

1. **先做资产闸门。** 记录 evaluator/MLIP 的版本、checkpoint 来源与校验值、训练数据、能量单位和 reference；为每个候选化学体系确认相图覆盖所有子体系，写明快照、thermo_type、entries、能量修正及来源。检查是否使用了与 MLIP 相同来源/尺度的参考项。缺 checkpoint、缺完整 hull、或 MP/OMat 能量口径混用即停止，不计算正式 ehull。
2. **固定配对校准集。** 从现有验证/开发数据中冻结约 30 个专用校准结构（数量为待确认的设计建议），校准结构不得计入最终独立测试，也不能用最终 test split 调整误差门槛或评价器。覆盖原始 ehull 五个区间和多种化学体系；尽量分层记录 task-entry 来源，但不把它误当 thermo_type。每个结构分别保留原始 CIF、PLaID `encode → decode` 结构、未弛豫分数及弛豫后结构/分数，并都使用同一明确版本的能量模型与 hull。报告相对原始 MP 标签的偏差、排序相关性、分箱误差，以及 round-trip 和弛豫造成的结构/ehull 变化。若不能补回精确 thermo_type，此比较只称代理分数校准，不宣称复现 MP 标签。
3. **单独审计训练重合。** 对 MPtrj、sAlex 等训练来源，用材料 ID 与标准化结构指纹同测试/候选集比对，保留重合数和未匹配数。无法获得完整训练成员清单时，把结果标记为有训练数据重叠风险的内部校准，不能作为独立泛化证据。
4. **最后测时间。** 资产闸门通过并获准执行后，在目标计算节点对少量覆盖原子数/化学组成的结构计时，分别记录载入、未弛豫单点评分、弛豫、相图计算耗时，成功率、步数和未收敛数。先用实测投影总评估成本，再决定批量大小；把这些 GPU 小时计入总 48 小时预算。

**停止依据：** thermo_type/参考项/能量修正不明或不匹配时，不输出 MP ehull；MLIP 与 MP 标尺无法校准时，只保留明确命名的模型代理分数；先由主流程确定允许的偏差与排序门槛，校准未达门槛或实测成本挤占训练/对照预算时，不扩大到整批生成结构。失败和未收敛样本保留在分母中。当前未确定数值误差门槛，也没有运行时数据。


## 主智能体复核

已重新核对本地 checkpoint/相图路径不存在、PLaID 的 MP2020Compatibility 调用及 FairChem 模型卡。原报告的 test split 校准建议已改为专用验证/开发校准集，避免用最终测试集选择评价器或调整门槛。DFT 路线的表述收窄为口径可解释，而非无误差真值。研究完成不等于评价器可执行验证完成。
