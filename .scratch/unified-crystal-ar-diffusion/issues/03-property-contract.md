# 确定首轮性质目标与数据口径

Labels: wayfinder:grilling
Type: grilling
Mode: HITL
Status: claimed
Assignee: Codex /root
Parent: ../map.md
Blocked by: 01, 08, 09, 10

## Question

依据数据研究的候选比较，首轮选择哪个单一性质、哪个来源和计算/标签口径？目标是值、范围还是类别，哪些结构约束必须满足，采用哪种可验证的性质评价，并怎样划分数据以避免目标或近重复结构泄漏？

需要用户实际选择并明确预测器评价与物理/实验验证的结论边界。已有 ehull 收集不能自动视为本票已决定。

## Comments

### Discussion — 2026-10-11

用户要求继续本对话讨论。前置数据研究已解决，主智能体复核了 manifest、verification、README 和本地评价代码。现有 45,629 条原结构—原 ehull 标签有完整性回执，但尚没有经过验证的新生成结构评价器；archive 未显式记录 thermo_type，不能据此声称全库能量口径一致。待用户决定是否优先沿用 ehull 并补齐评价依据，或先调查替代性质；本票仍未解决。

### Partial decision — 2026-10-11

用户选择“就按照 A 来”：优先沿用已有 MP—ehull 数据，把 ehull 定为首轮候选，先核查新生成晶体的 ehull 评价路线与预算可行性。暂不调查替代性质作为主线，也不开始训练。

这项回复没有确定目标值/范围/上限语义、标签与评价能量口径、性质容差、结构约束或最终划分。本票保留 claimed，等待 [核查 ehull 评价路线与能量口径](08-ehull-evaluation-evidence.md) 的证据后继续人工讨论，不将候选选择误记为完整契约已解决。

### Partial decision — MLIP selection, 2026-10-11

用户通过截图明确指定评价用 MLIP：**GRACE-3L-OAM-L**。该选择已确定，不再比较或改用 EQv2/eSEN/CHGNet；它是外部性质评价器，不是待训练的共享 AR/diffusion 生成模型。

Matbench Discovery 的官方配置将此名称对应到 `GRACE-3L-OMAT-large-ft-AM-model.tar.gz`，模型版本标为 `1`；训练来源为 MPtrj/OMat24/sAlex，参数数为 42,112,223。官方 benchmark 环境记录 Python 3.12、TensorFlow 2.20.0 和 tensorpotential 指定 commit。见 [官方配置](https://github.com/janosh/matbench-discovery/blob/main/models/grace/grace-3l-oam-l.yml) 与 [GRACE 官方 foundation 说明](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/)。这些是公开配置事实，不是本地运行结果。

具体权重 revision/hash、评价输入和弛豫协议、竞争相参考与 MP 标签口径仍未冻结；交给 [核查 GRACE-3L-OAM-L 接入与 ehull 标尺](09-grace-evaluation-evidence.md) 调查。旧研究中关于 EQv2/eSEN 的兼容性表述不能未经核查直接套用到 GRACE。本票不据本次选型关闭完整性质契约。

### Scope refinement — 2026-10-11

训练条件与生成评价的关系交由 [明确训练性质条件与生成评价的关系](10-target-semantics.md) 保存唯一讨论与决定。本票最终整合引用该票，继续确定数据/评价口径、结构约束和划分，不重复记录其细节。
