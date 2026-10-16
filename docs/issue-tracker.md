# 本地 Markdown 决策票

本项目当前使用本地文件，不连接外部 issue tracker。

## Wayfinding operations

- 地图：`.scratch/<effort>/map.md`，标签 `wayfinder:map`。
- 子票：`.scratch/<effort>/issues/NN-<slug>.md`，一票一个文件，`Parent: ../map.md`。
- `Type:` 取 research/prototype/grilling/task，标签为对应 `wayfinder:<type>`。
- 开放状态为 `Status: open`；认领前写入 `Status: claimed` 和 `Assignee:`，完成后为 `Status: resolved`。
- 本地 tracker 没有原生依赖图，使用 `Blocked by: NN, NN`。只有所有前置票为 resolved，才可进入该票。
- frontier 是 open、未认领、所有前置 resolved 的子票；按文件编号排序。已认领但等待用户的票必须保留等待说明，不能自行解除认领或假定答案。
- 调查前先认领；解决时把证据和结论追加到 `## Comments` 下的 `### Resolution`，置为 resolved，再由主智能体向地图 Decisions so far 添加名称链接和一句摘要。
- 研究资产放在 `research/wayfinder_crystal_20261011/`，票中使用名称链接。研究报告不是数据、模型或实验效果交付。
- 用户可见说明以票名及链接引用，编号仅用于文件身份与依赖字段。
- 2026-10-13 已按用户要求初始化项目根目录 Git（main），尚无提交或远程仓库；研究资产继续用本地目录隔离，既有实验快照保持原样。
