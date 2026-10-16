# 核查共享 AR 与 diffusion 的论文配方和实现差距

Labels: wayfinder:research
Type: research
Mode: AFK
Status: resolved
Assignee: Codex /root
Worker: /root/wayfinder_method
Parent: ../map.md
Blocked by: none
Research context: [方法与复用证据](../../../research/wayfinder_crystal_20261011/method-evidence.md)

## Question

在固定 PLaID 类 token 表示、共享核心模型、简短客观文字和单张 A100 累计 48 GPU 小时的条件下，Efficient-DLM、FLARE、Llamole 中哪些方法已有直接证据，哪些需要迁移到晶体？最小的可信候选配方有哪些，现有 PLaID 原型可以复用什么、有哪些不能沿用的假设？

核查干净因果流、当前带噪块的可见性、预测位置与配套采样、互补 masking/监督平衡、明确性质条件入口、文字—晶体边界和实际结构反馈。分别标出论文配方、既有代码与新方案建议；不能声称旧原型已完整复现论文，也不能把原先编辑任务当成性质生成数据。

提出可供后续决策比较的共享模型与纯 AR 对照选项。时间/显存估计仅用已有日志或来源证据，区分实测和估算；没有证据则留空。只读研究，不改源码、不加载大权重、不提交作业，不替用户冻结架构。

## Comments

### Resolution

研究报告：[固定 PLaID 的共享 AR/扩散方法证据](../../../research/wayfinder_crystal_20261011/method-evidence.md)。

事实区分：Efficient-DLM 给出语言的干净前缀块扩散；FLARE 给出同 checkpoint 的 clean causal/noisy diffusion 双流与互补遮盖；Llamole 以 trigger/query 调用独立 Graph DiT/GNN，显式接收性质条件。三篇没有直接实现共享 PLaID 文本—晶体模型。本地原型支持同位置去噪与 1/k 释放，但旧 prepare 仅有公式/空间群、拆分的生成/读取/编辑任务，没有性质标签或完整文字—晶体—文字样本。

报告提供单视图 PLaID 扩散与 FLARE 启发的互补视图两种候选，并配有匹配纯 AR 对照。旧单 A100 42:40 仅是 8 材料编辑试点收据，不能外推新性质任务；48 小时足够与否仍未知。性质、标签/评价来源和方案选择留待人工决定。
