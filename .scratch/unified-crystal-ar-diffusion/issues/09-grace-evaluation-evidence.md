# 核查 GRACE-3L-OAM-L 接入与 ehull 标尺

Labels: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: Codex /root
Worker: /root/wayfinder_grace
Parent: ../map.md
Blocked by: 08
Research context: [GRACE 固定模型评价依据](../../../research/wayfinder_crystal_20261011/grace-evaluation-evidence.md)

## Question

用户已确定 GRACE-3L-OAM-L，不再比较其他 MLIP。这个明确版本的官方权重、名称映射、元素支持、ASE 接口、输出单位、运行依赖和官方评估协议是什么？能否给出可复现的后续接入方案，且不改变已有生成模型环境？

追踪 Matbench Discovery 官方 GRACE 配置与预测/评分代码，弄清 leaderboard 的能量/ehull 是如何获得的：模型输出的总能量、元素参考/修正、DFT 参考能量、相图或基准中已知值的使用分别是什么。不得只依据表中 MAE 或排名就宣称可直接评价任意新晶体，也不能把 EQv2/eSEN 的能量兼容性结论直接套用。

针对当前 MP 2025.06.09 配对标签与缺失 thermo_type，明确生成结构与参考竞争相分别需要什么资产、该用哪种能量标尺、哪些参考数据目前没有。提出小规模开发校准集和计时设计；实际计时/校准未进行时保留未知，不承诺48 GPU小时预算足够。

只读官方文档、模型 registry/文件元数据、公开代码及必要本地 schema/回执。可以读取少量原始配对记录的字段名，但不要扫描全量数据、读凭据、访问私有 DB、下载模型权重或大数据、安装依赖、加载模型或提交计算任务。本票解决接入与评价事实，不执行校准，也不训练 GRACE。

## Comments

### Claim — 2026-10-11

已确认官方 benchmark 名称 `GRACE-3L-OAM-L` 对应发布文件 `GRACE-3L-OMAT-large-ft-AM-model.tar.gz`。主智能体在委派前认领；选型已经由用户完成，调查不得重新打开模型选择。

### Resolution — 2026-10-11

已完成官方配置、GRACE 文档/registry、Matbench Discovery 推理与评分代码及少量本地记录字段核对，报告见[固定 GRACE 接入与 ehull 标尺](../../../research/wayfinder_crystal_20261011/grace-evaluation-evidence.md)。Matbench 将 GRACE 弛豫后能量套入来自 WBM/DFT 的条目，应用 MP2020 修正与 MP 元素参考得到形成能，再相对固定 DFT hull 算预测 ehull；`0.018 eV/atom` 是该 leaderboard 测试口径下的 MAE，不能当成本地数据误差。当前本地样本没有总能量与竞争相 hull，前三条记录的 `ehull_thermo_type` 均为空。

发现 Matbench 配置声明的 TensorPotential commit `3115a9314` 不含 3L OAM registry 项，而当前上游已有该名称；同名 Hugging Face archive 也有内容变更且配置未固定 SHA。因此后续必须在独立环境固定兼容 registry 和不可变权重修订。精确元素清单、实际 benchmark 使用的 archive 修订、MP 口径 reference hull、本地重合量及实测耗时仍未知。未进行安装、下载权重、模型推理/弛豫或训练；本票 resolved 仅表示事实核查完成，评价器尚未 validated。

### 主智能体复核

已核对官方评分与能量代码。候选 DFT ehull 与 MP-reference 的 GRACE 预测 ehull 分开；后者可作为待校准的预测评价路线，不必先为所有候选执行 DFT。benchmark 的 DFT 真值依赖异常过滤不得直接照搬到未知生成结构。所有未运行与口径不确定事项保留。
