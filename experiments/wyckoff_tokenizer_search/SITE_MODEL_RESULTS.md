# 位点类型计数聚合：模型初筛结果

## Material Passport

- 范围：开发性 masked-profile 表示实验，不是完整条件扩散或几何生成。
- 协议：`SITE_MODEL_PROTOCOL.md`，源文件与协议摘要在 `site_inventories/model_protocol_receipt.json`。
- 执行：Burgundy A100，Slurm 533539，tmux 会话 work；006-site-model-pilot 与 007-compare-site-models 均 exit 0。
- 数据：训练 26086 个结构；既定开发验证 2048 个结构；单 seed 17；每种 10000 步。
- 预留 6620 个验证结构没有用于本轮模型评估。
- 状态：完成；未达到继续扩大本分支的预定门槛。

## 结果

| 指标 | 位点类型聚合 aggregate | 逐 WP 记录 records |
|---|---:|---:|
| 完整结构 WP profile 准确率 | 66.5039% | 66.7480% |
| 每元素 profile NLL，越低越好 | 0.566231 | 0.568510 |
| 参数量 | 558624 | 558624 |
| 训练及周期评估用时 | 587.40 s | 588.67 s |
| 峰值 CUDA 分配内存 | 132340736 B | 133327360 B |

同结构配对 bootstrap（2000 次）得到 aggregate − records：

- 准确率差 -0.2441 个百分点；95% 区间 [-1.2219, +0.7813] 个百分点。
- NLL 差 -0.00227964；95% 区间 [-0.011076, +0.006421]。
- NLL 点估计相对改善仅 0.401%；峰值分配内存约下降 0.74%，不足以构成明显资源收益。

区间都跨零，且仅涉及一个 seed 和用来选择 checkpoint 的开发集。这里不能作稳定优势、等效性或非劣效性结论。原协议要求至少 +1 个百分点准确率或 2% NLL 改善，并且另一指标不能恶化；本轮没有达到。

## 对照完整性

两种模型共享同一构造器、forward 和训练循环，初始参数逐项相等；完整 WP 分支在全部 32368 候选上完全一致。两者均可见相同的 WP identity、重数、自由度和内禀位点类型特征。唯一主要变化是 coarse branch 在计数嵌入之前还是之后聚合。

全部候选的 inventory invariance、同类型无重复时的编码一致性，以及隐藏目标后更换真实 profile 不影响输出的检查通过。训练数据、验证 ID、标签、参数量及相关代码摘要在统计脚本中逐项核对通过。

## 决策

停止扩大当前“位点类型计数聚合”模型分支，不追加多 seed、不使用预留验证组、不据此实现更大的 inventory-first diffusion。表示的精确预算和无损还原结论仍成立，但这不等于模型收益。

这次结果只约束已测试的 encoder 聚合方式，不能证明所有位点类型生成分解都无用。当前证据也不足以把它推荐为最终 tokenizer。

后续研究优先检查能够保留几何连接信息的表示，而非继续只对同一 WP 计数变量反复分组。一个待审查方向是：以 asymmetric-unit/Wyckoff orbit 为节点，用 CIF 已有的 symmetry-operation + translation bond 标签表示键轨道，再展开完整周期图。该方向尚未编码或验证；SLICES、SLICES-PLUS 和 asymmetric-unit 生成的先例边界正在核对，不能提前声称创新。

详细数值：`site_inventory_model_runs/paired_seed17.json`；逐结构预测、checkpoint 选择历史、配置及源码快照均已保留。
