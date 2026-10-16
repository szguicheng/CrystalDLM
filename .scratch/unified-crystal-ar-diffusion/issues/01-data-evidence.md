# 核查首轮单性质数据与评价依据

Labels: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: Codex /root
Worker: /root/wayfinder_data
Parent: ../map.md
Blocked by: none
Research context: [数据与评价证据](../../../research/wayfinder_crystal_20261011/data-evidence.md)

## Question

现有与可获取的公开晶体数据，能支持哪些首轮单性质目标？优先核查当前 MP—ehull 配对数据和收集回执，而非重复收集；至多比较两个有明确优势的替代性质。需要哪些结构/标签口径、单位、目标范围、缺失值规则、重复结构与划分约束，才能建立可追溯的目标条件生成任务？

依据本地回执和官方来源，说明数量与分布、标签来源/计算口径、数据已收集与仅有计划的区别。特别核查现成评价器是否真的能给“新生成结构”评分、是否只对原数据库结构有效、预测与重新计算的区别、PLaID 精修/舍入是否影响结构—性质标签配对，以及评价所需预算和域外风险。

交付是带出处的证据表、候选比较和未确定事项，不替用户锁定性质，不调用付费接口、不批量下载、不训练或重新计算性质、不读取或输出凭据。

## Comments

### Resolution

[首轮单性质数据与评价依据：证据盘点](../../../research/wayfinder_crystal_20261011/data-evidence.md)

- 已有 MP 快照 154,879 条；64,043 条通过元数据/结构筛选，最终确定性抽样 45,629 条，ehull 原值范围 0–0.3 eV/atom，train/val/test 为 36,491/4,517/4,621；全量源结构/标签保持验证通过。三条官方值对照仅佐证字段来源，archive 未记录凸包 `thermo_type`。
- PLaID 训练序列化先做空间群精修，晶格舍入到 0.01、Wyckoff 位点坐标到 0.001；精修/解码结构的性质偏差未知，因此原标签不能直接转作新生成结构的评分。
- 当前本地生成结构评价器并未就绪：旧管线依赖未找到的模型/相图资产，CHGNet 与 M3GNet 代码路径是 stub；ehull 评价还要求与参考相同能量尺度的化学体系凸包。formation energy 与 band gap 是 MP schema 中两个备选字段，但本地快照未保留，数量/覆盖率未知。

研究证据票据已解决；性质选择、目标条件语义和评价标准仍待人工决策，本 resolution 不锁定任何性质。
