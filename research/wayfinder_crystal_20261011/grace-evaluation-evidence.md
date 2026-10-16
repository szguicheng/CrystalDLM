# GRACE-3L-OAM-L：固定模型接入与 ehull 标尺

核查日期：2026-10-11。只读官方配置、文档、代码及三条本地记录字段；没有下载权重、安装环境、加载模型、弛豫、DFT、提交 Slurm 或训练。本文完成的是事实核查，不代表 GRACE 评价器已在本机验证。

## 已确认

Matbench Discovery 的模型键为 `grace_3l_oam_l`，正式展示名 `GRACE-3L-OAM-L`，对应上游全名与权重目录名 `GRACE-3L-OMAT-large-ft-AM`。官方文档说明 OAM 是先在 OMat24 预训练、再用 sAlex 与 MPtraj 微调；目标模型是 3 层半局域网络、固定 6 Å cutoff、原生 fp32，并带 UQ。Matbench YAML 记载 42,112,223 个参数、Python 3.12、TensorFlow 2.20.0、目标 `EFS_G`，训练/测试任务为 `S2EF` / `IS2RE-SR`，评估设定为 ASE FIRE、`FrechetCellFilter`、`fmax=0.01`、最多 500 步。这个已选模型仅作为外部能量评价器，不是生成模型，也不在本任务里微调。[Matbench 固定配置](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-3l-oam-l.yml)、[GRACE Foundation Models 文档](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/#oam-models)。

官方 ASE 入口是 `from tensorpotential.calculator import grace_fm`，以全名 `grace_fm("GRACE-3L-OMAT-large-ft-AM")` 创建 calculator；`atoms.get_potential_energy()` 给总能量 eV，`atoms.get_forces()` 给 eV/Å。[ASE 快速开始](https://gracemaker.readthedocs.io/en/latest/gracemaker/quickstart/#usage-in-ase)、[GRACE TensorPotential calculator](https://github.com/ICAMS/grace-tensorpotential/blob/master/tensorpotential/calculator/foundation_models.py#L258-L265)。公开配置只泛称 foundation models 支持很广的元素范围，没有给出该权重的精确元素清单。Matbench YAML 报告 WBM 全集仅有 2 个缺失预测；这不是精确元素覆盖表。[foundation models 文档](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/)、[模型配置](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-3l-oam-l.yml)。

**存在可复现性缺口。** Matbench 配置将 TensorPotential 固定在 `3115a9314`，GitHub 将它解析为 2025-03-12 的提交；该提交的 [foundation_models.py](https://github.com/ICAMS/grace-tensorpotential/blob/3115a93142943f0a976bd09c63435719e91a7c1d/tensorpotential/calculator/foundation_models.py) registry 没有 3L OAM 名称。当前上游 registry 已列出该名称，但这不能证明旧 pin 可加载它。权重 URL 也写作 Hugging Face `resolve/main`，没有在 Matbench YAML 固定文件摘要。HF 的 v4 commit `d81e55b58dcb20f13fcac0dfac68b8bcd778d04d` 记录同名模型 archive 从 SHA256 `1636ec2a87e7cbfc0cbb46f5b55af117fb1fab271765652a9633aa6492f38e3d`、156,714,415 字节变为 `ede0fc88af9ed54fee2358d342f6f5a38a1b4b5b2e3ec78ae408c10315ce0f40`、191,368,412 字节；因此 YAML 中的 leaderboard 分数尚不能追溯到确切 archive 修订。[Matbench YAML](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-3l-oam-l.yml)、[TensorPotential 当前 registry](https://github.com/ICAMS/grace-tensorpotential/blob/master/tensorpotential/calculator/foundation_models.py)、[HF v4 模型文件变更与 LFS 摘要](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/commit/d81e55b58dcb20f13fcac0dfac68b8bcd778d04d)。

## Matbench 的 GRACE → formation energy → ehull

Matbench Discovery 将 WBM 初始测试结构交给 ASE calculator，并按模型配置弛豫。写出发现分数时，它读取对应 WBM `ComputedStructureEntry`，把能量和结构分别替换为模型的弛豫总能量与弛豫后结构，再调用 `MaterialsProject2020Compatibility.process_entry(clean=True)`。所以用于计算的是 GRACE 弛豫几何，但条目仍来自 WBM/DFT 记录并保留其计算元数据；这不是只凭模型输出新建、完全独立的 MP 条目。[WBM 结构与弛豫入口](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/discovery.py#L254-L328)、[条目替换与修正](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/discovery.py#L680-L755)。

修正后的能量再减去 Matbench 保存的 MP 元素参考能量，除以原子数，得到预测形成能。指标代码将预测 ehull 算作 `DFT true ehull + predicted formation energy - DFT reference formation energy`；README 明确评价凸包使用 DFT 参考能量，而非 GRACE 预测能量重建的 hull。[形成能与 MP 元素参考实现](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/energy.py#L77-L154)、[预测 ehull 与指标](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/metrics/discovery.py#L179-L238)、[Matbench README](https://github.com/janosh/matbench-discovery#readme)。因此 YAML 中的 `0.018 eV/atom` 是指定 WBM 测试集、MP2020 修正、元素参考和固定 DFT hull 下的 leaderboard MAE；它不是本项目 45,629 条数据的预测误差，也不能单独证明 GRACE 总能量与任意 MP 凸包兼容。

## 本地参考数据与未决事实

本地 45,629 条样本记录含 `structure` 与原始 `energy_above_hull`，没有总能量、形成能、ComputedStructureEntry 或竞争相条目。仅查看 `records.jsonl.gz` 前三条的字段和值：`provenance.ehull_thermo_type` 均为 `null`；其中 `matching_structure_entry_types` 出现 GGA/GGA_U，但它不能替代 hull 的 `thermo_type`。manifest 也将 archive 的 thermo type 标为未显式保存。[本地数据 README](../mp_ehull030_20261011/collection_001/data_001/README.md)、[manifest](../mp_ehull030_20261011/collection_001/data_001/manifest.json)。

GRACE 官方说明其 OAM 微调数据含 MPtraj，但与这批 MP 样本按 ID/结构核对的精确重合量未知。因此验证集结果应注明有同源数据重合风险，不能仅凭固定 train/val/test 划分称作独立外部泛化结果。[GRACE foundation 配置](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-3l-oam-l.yml)、[OAM 数据说明](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/#oam-models)。

## 建议的固定接入、标尺与后续校准

1. **隔离接入。** 在生成模型环境之外建立 GRACE 专用环境；选择一个 registry 确实含 `GRACE-3L-OMAT-large-ft-AM` 的 TensorPotential release/commit，固定 Python/TensorFlow/ASE 版本。权重采用不可变 HF revision，并保存完整 SHA256、registry commit 和 `grace_fm` 初始化回执。完成前不照抄 Matbench 的旧 TensorPotential pin。随后只做单结构能量/力 smoke check，确认 eV 与 eV/Å、支持元素、有限输出及调用路径。
2. **确定评分名义。** 若采用 **GRACE-self-hull 代理 ehull**，候选与该化学体系所有竞争相都必须由同一 GRACE 权重、同一结构/弛豫协议得到能量，再由这些 entries 构建 GRACE hull；此值只解释为模型内相对稳定性。若采用 **以 MP 相图为参考的预测 ehull**，可以使用经过能量口径核查/校准的 GRACE 候选能量，加上明确版本、兼容修正与元素参考的完整 DFT 竞争相 hull；结果必须称为预测值，不是候选结构的 DFT 计算结果。需要说明预测误差，不能不经校准把 GRACE 原始能量套入 MP2020 修正。只有要求候选本身也是 DFT 计算的 ehull 时，才需要候选 DFT 能量。当前 archive 不含完整参考资产，少量经验拟合不能替代缺失 reference hull；MP-reference 预测与 GRACE-self-hull 也不能混作同一标签。
3. **小规模校准与计时（建议，未执行）。** 资产闸门通过后，从 validation 冻结 30 条开发结构，按五个标签区间每区 6 条并尽量跨化学体系；单独保留原始结构与 PLaID encode→decode 结构。仅在 reference hull 明确后计算分数，报告带符号偏差、MAE、Spearman 排序、分箱误差、失败/未收敛比例；validation 不参与训练或最终测试。先以 4–8 条不同原子数/元素数组合的结构测冷启动、重复单点、FIRE+FrechetCellFilter 弛豫和 hull 计算耗时，记录每阶段 wall time、GPU 小时、峰值显存、步数和收敛状态，再投影样本量。所有工作计入单 A100 的累计 48 GPU 小时；当前没有耗时证据，不能承诺预算足够。

**结论边界：** GRACE 已固定为指定外部模型；benchmark 的评分链路可复现地说明其使用 MP2020 修正与固定 DFT hull。确切的上游权重修订、与其匹配的 TensorPotential registry、模型精确元素表、本地竞争相 hull、当前本地能量口径兼容性和运行耗时仍未知。此票关闭表示事实核查完成，不表示评价器已安装、校准或验证。

## 主智能体复核补充

已核对 Matbench 当前 `calc_discovery_metrics` 的固定 DFT hull 预测公式和 `write_discovery_artifacts` 的条目替换/MP2020 调用。严格候选 DFT ehull 与以 MP 相图为参考的 MLIP 预测 ehull 已分开，不把采用预测评价误说成必须为所有候选跑 DFT。当前 leaderboard 的 `prepare_model_predictions` 还使用已知 DFT 形成能过滤大误差预测，并进行三位小数舍入；任意新生成结构没有该真值，不能照抄这种真值依赖的过滤来删除失败。正式评价需保留原始结果、失败和未收敛记录。以上仍为静态核查，没有运行评价器。
