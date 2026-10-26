# 计数块的多步条件解码诊断

目的：一步 all-mask MAP 不完全等于多步生成；用已训练的强打包基线检查实际逐步预测的行为，不重新拟合词表或模型。

## 固定设置

- 使用 `packed_word_runs/{rle,rlemerge}_seed17` 的最佳验证 NLL 检查点，10,000 步预算相同。
- 只评估既有固定的 2,048 条验证结构，不读取已用过的测试模型结果来选择步数。
- 步数 T={1,4,8}，确定性分支，不加 Gumbel 或其他采样噪声。所有步数都报告。
- 输入仅为 SG、组成与 context 索引。生成 batch 显式移除 `profile_id`、`target`；目标词从零状态开始，必要槽位全部 mask。

## 解码

1. 并行预测全部未知词，给完整候选的 canonical word 序列评分。
2. 只保留与已固定、由模型自己生成的词一致的候选，再以相同固定 WP 排他 MAP 选择一个合法完整 proposal。
3. 以 proposal 中词的网络后验概率为置信度，仅对当前 mask 位置排序。
4. 第 k+1 步剩余 mask 数为 floor(N_initial*cos(pi/2*(k+1)/T))，最终强制为零。固定高置信词，保留其余 mask。
5. 最终词序列必须与最终合法 proposal 一致，不能用真实答案修复。

这是对 MaskGXT 并行置信度解码概念的约束适配。其原论文预测坐标/晶格等字段，我们加入了既有候选读出，不能称逐字复现 MaskGXT 或精确 D3PM。

## 检查与指标

- T=1 的完整占位匹配率必须严格复现已有一步 MAP 收据。
- 报告展开到相同 protostructure 后的匹配率、配对结构差异区间、实际 batch forward 次数和耗时。
- 合法性由共同候选读出保证，只是管线检查，不能作为 tokenizer 的性能收益。
- 不把途中条件 NLL 当成多步生成分布的 marginal likelihood。
- 本次一个训练种子，任何正向信号仍需复核。连续几何、物理稳定性和随机采样多样性不在本轮结果中。

原文定位和迁移边界见 `../../research/notes/maskgxt_sampling_protocol.md`。
