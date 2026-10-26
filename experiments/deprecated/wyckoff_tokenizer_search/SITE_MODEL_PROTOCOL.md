# 位点类型聚合：首轮模型对照（训练前固定）

## Material Passport

2026-09-08；开发性表示实验；使用既有 MP-20 cap256 候选集和既定 2048 个开发验证结构。预留 6620 个验证结构不评估；既有官方 test 不再次用于方案选择。此处不验证完整扩散和连续几何。

## 问题与唯一主要变化

固定完整 WP 候选与 fine branch，在 coarse branch 比较：

- `aggregate`：先按内禀位点点群类型 H 汇总 WP orbit counts，再计算 `sum_H E(H) * E(sum_WP count)`。
- `records`：逐 WP 计算 `sum_WP E(H_WP) * E(count_WP)`。

两者的 fine branch 都含 WP identity、primitive multiplicity、DOF、H 和 count。新增 H embedding 在两者中相同，不能把与旧模型的差异直接归因于 tokenizer。两者继承同一个 forward、使用同一个参数构造器；唯一分支在 count embedding 前是否聚合。

该模型仍把每元素的完整 profile 作为被遮盖状态，采用完整候选联合打分；它是对 inventory 表示价值的初筛，**不是 inventory token 逐个扩散或 inventory-first 生成模型**。若没有帮助，不投入更大的分阶段实现。

## 固定训练与评估

- Seed 17；每种 10000 个更新；batch 32；AdamW lr 3e-4、weight decay .01；width 96、3 层 transformer。
- 数据顺序、随机遮盖率和全遮盖概率、梯度裁剪以及评估循环与 frozen unaggregated control 相同。
- 给定空间群与元素计量；隐藏 target profile 时不输入真实 inventory、轨道数、DOF 或长度。
- 每 1000 步评估既定 2048 开发样本，按 profile NLL 选择 checkpoint。
- 相同 exact fixed-site MAP 解码，同时约束每元素组成与不同元素的固定 WP 互斥。
- 主结果：完整结构 profile exact match、每元素 NLL；另报告运行耗时与峰值 GPU 分配内存。
- 配对 bootstrap 在同一结构记录上重采样，报告 aggregate − records 的 exact 和 NLL 差值及 95% 区间。因为存在开发集选择和单 seed，这些区间只描述首轮差异，不是确认性结论。

## 决策规则

进入多 seed 与预留集合复核，需要看到有实际量级的正向信号：exact 至少提高 1 个百分点，或 NLL 相对下降至少 2%；另一质量指标不能同时恶化。小于这个量级、方向相反或明显劣于对照时，停止扩大本分支。只有跨 seed、独立复核支持时才主张稳定收益。

即使初筛通过，后面也必须显式改变实际预测/去噪单位并验证，才能把结果称为 tokenizer 的模型收益，而非共享 embedding 的改进。

## 运行验证

在计算节点先做全部 32368 个候选的 fine parity、相同 inventory 的 coarse invariance、无重复 H 时两种 coarse 编码一致性、完整遮盖后的 target invariance 和相同初始参数检查；再各运行 16 步 GPU smoke。通过后在同一 A100 分配与 tmux 队列连续运行两种正式版本。
