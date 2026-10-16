# 核查 ehull 评价路线与能量口径

Labels: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: Codex /root
Worker: /root/wayfinder_ehull
Parent: ../map.md
Blocked by: 01
Research context: [ehull 评价路线证据](../../../research/wayfinder_crystal_20261011/ehull-evaluation-evidence.md)

## Question

用户已经决定优先沿用 ehull。对于实际新生成的 PLaID 解码晶体，最多两条具体评价路线分别需要什么能量模型、弛豫方法、元素参考/能量修正和竞争相凸包资产？它们怎样与当前 MP 2025.06.09 原始配对标签保持可解释的口径；哪些能核查，哪些不能从 archive 缺失的 thermo_type 推定？

优先核对 PLaID 官方稳定性评价实现、发布的模型与相图资产及当前本地资产存在状态，并参考可获取的官方 MLIP/MP 实现。说明原始生成结构与弛豫后结构的评分差别、标签/校正兼容性、预训练数据重合和域外误差。不能仅用 checkpoint 的存在或 API 的返回来宣称评价可信。

给出最小的后续验证设计：哪些已知参考结构可用于检查能量/ehull 偏差与排序，PLaID 编码前后需要怎样比较，先检查什么才值得跑生成评价，怎样记录失败、预算与推进/停止依据。没有相应实测时，不能承诺 48 GPU 小时足够。验证设计是供后续决策的方案，不在此票执行。

仅进行资料、代码和轻量资产元数据核查。不要读凭据、查询私有数据库、调用付费接口、下载大资产、加载模型、运行弛豫/DFT、改训练代码或提交计算作业。只读 SSH 元数据核查可选，连接不可用时记录未验证，不启动 VPN 或索取登录信息。

## Comments

### Claim — 2026-10-11

前置数据研究已 resolved，用户已选择优先 ehull。主智能体在委派前完成认领；本票提供证据，不替用户选择最终评价口径，也不授权执行验证设计。

### Resolution — 2026-10-11

研究报告：[PLaID 解码晶体的 ehull 评价路线：证据与后续门槛](../../../research/wayfinder_crystal_20261011/ehull-evaluation-evidence.md)

关键事实：官方 MLIP 线需模型与相图资产；本地对应 checkpoint、entry archive 和 2023 相图均缺失。EqV2/ESEN OAM 使用 OMat 预训练并以 MPtrj/sAlex 微调；FairChem 警示 OMat 能量与 MP 计算不兼容，但其明确禁止 MP2020 修正的条款指向 OMat-only 权重，未提供 OAM 与 MP2020 兼容证明。MPtrj 带来 MP 训练重合风险，故不能直接把现成代码的输出称为 MP ehull。DFT + 同 thermo_type 完整 MP hull 是较可解释的候选，但当前 2025.06.09 数据缺 `thermo_type`，旧 PLaID 2023 hull 不能视作一致；耗时未测。报告给出配对校准、资产核验、训练重合审计、时间测量和停止门槛。本 resolution 只关闭事实调查，不表示评价器已经运行验证或授权执行。

### 主智能体复核

本地资产存在状态及官方能量兼容性表述已核对。后续校准使用开发/验证专用集，不使用最终测试集来调评价器；DFT 结果也有方法误差。没有运行性质评价或计时，未证明评价器可用或总预算可行。
