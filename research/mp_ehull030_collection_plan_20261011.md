# MP 晶体与 ehull 数据收集方案

## 当前范围

收集结构与性质配对数据，ehull 范围为 0–0.3 eV/atom。沿用 PLaID 类晶体表示；replace/swap 仅是老师举例，不构建操作链。本阶段不训练、不做消融、不评价生成晶体的稳定性，也不重新计算原始 ehull。

## 来源与标签

- 优先读取用户现有 MongoDB 中的 MP 数据；先检查集合结构、快照日期、标签与结构的关联方式，再统计候选数量。
- 备用来源为 MP API 批量查询。它支持大量记录下载，官方说明每秒 25 次请求开始限流；计划采用 1,000 条/页、保守请求节奏及一次下载后的本地快照。
- 使用同一 MP 快照和同一 thermo_type/凸包口径，不混用旧 MP-20 标签与新版 MP 标签。优先考虑 GGA/GGA+U 兼容体系；实际选择以数据库可追溯的配对数据为准。
- 保留 MP 原始材料记录中的结构与标签配对，记录 thermo entry/task 谱系；不能仅凭 material_id，把不同快照的结构与 ehull 人工拼接。MP 官方材料级记录本身的结构和能量可以有不同 task 来源，不把这类官方配对改写成单一 task 的新样本。
- 标签为原始连续数值 energy_above_hull / e_above_hull，单位 eV/atom。不分稳定类别、不四舍五入后替换原值。
- ICSD 仅作为 MP 记录已有的来源标记。独立 ICSD 结构不因编号/组成相同就继承某条 MP 的 ehull。

## 首轮纳入规则

- 0 <= ehull <= 0.3，标签有限且存在对应 MP 弛豫结构。
- 使用 MP 原始配对结构的 nsites：1–20；不在抽样前随意替换成另一种晶胞或重新弛豫。
- 1–4 种元素；元素种类和空间群不另设白名单。实验来源与理论预测结构均纳入并保留来源标记。
- 保留有序、完整占位的周期晶体；排除 deprecated、缺结构、缺标签和无法表达的无序/部分占位记录。记录每项排除的数量和 ID。
- 不继承 MP-20 的 formation_energy < 2 eV/atom 或仅 ICSD 来源筛选。

## 规模和随机抽样

先统计全部合格候选的轻量元数据，不从 API 返回的前 N 条中截取。首轮目标上限 50,000 个不同结构，以下五层各至多 10,000 条：

| 层 | 原始 ehull (eV/atom) | 上限 |
|---|---|---:|
| 1 | = 0 | 10,000 |
| 2 | (0, 0.05] | 10,000 |
| 3 | (0.05, 0.10] | 10,000 |
| 4 | (0.10, 0.20] | 10,000 |
| 5 | (0.20, 0.30] | 10,000 |

这些边界用于抽样，不是热力学稳定性分类标准。某层不足 10,000 时全部保留，不重复采样，也不从低 ehull 层补满；实际总量可以少于 50,000。

每层按元素集合定义化学体系，例如 Fe-O。按 sqrt(该体系候选数) 分配该层预算，配额不得超过候选数；取整与余量按固定体系顺序分配。体系内部按 SHA-256("17|" + material_id) 排序，无放回选取；并列时按 material_id 排序。这样既保留常见体系，也减少少数高频体系主导的情况。记录随机规则、原始人数与实际人数。

每个 MP material_id 只保留一条选定热力学口径下的配对记录。重复结构归为同组；不同多晶型不因组成相同而删除。按分组后的固定随机规则划分 train/val/test，目标比例 80/10/10，重复结构不跨划分。这是数据组织，不是消融或生成性能实验。

## 保存字段与第一阶段交付

保存 material_id、原始 structure JSON/CIF、原始 ehull、formula、elements、chemsys、nsites、spacegroup、theoretical/ICSD 来源标记，以及可获得的 snapshot、thermo_type、thermo_id/task_id。缺失的来源信息明确记录为缺失，不推定对应关系。

输出候选统计、选定 ID 清单、原始配对数据、原始文件哈希和抽样/划分说明；另存排除记录。保留原始结构和标签，后续 PLaID 编码作为单独派生产物。

## 待获取的信息

MongoDB 部署机器或 SSH 别名、已有连接配置文件路径、MP 数据库与集合名。不要将数据库密码写入本文件或聊天日志。实际候选量、各标签层是否达到配额，须查询数据库后确认。

## 2026-10-11 实际来源确认

使用 `groupdb.mp_api_all_2025_7`，154,879 条记录均标记 MP 数据库版本 `2025.06.09`。用户说明 `energy_above_hull` 应为 MP 直接下载字段，并要求核对三个官方材料。MP 当前官方 API 的三个原始 ehull 数值与存档相符（差异 < 3e-16 eV/atom），同时记录旧编号到新编号的返回映射。官网例子也说明结构 task 与能量 task 可能不同，因此采用原始 MP 材料级配对，不改用预测字段、不改换另一套结构。热力学口径在该存档中没有单独字段，不声称已逐条追溯凸包计算；三例对照只作为来源佐证。

## 官方依据

- MP 大规模下载说明：https://docs.materialsproject.org/downloading-data/using-the-api/tips-for-large-downloads
- MP summary 查询字段：https://materialsproject.github.io/api/_autosummary/mp_api.client.routes.materials.summary.SummaryRester.html
- MP thermo 类型查询：https://materialsproject.github.io/api/_autosummary/mp_api.client.routes.materials.thermo.ThermoRester.html
