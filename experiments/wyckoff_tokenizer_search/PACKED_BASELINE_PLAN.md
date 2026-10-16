# 下一项对照：条件决定的有效槽位打包

冻结测试已经完成，预测质量差异不确定。不要修改 `direct_words.py`（已冻结 SHA），不要覆盖 `compact_10k_runs/` 或 `frozen_word_evaluation/`，也不要把复用的测试集称为新的 untouched test。

## 问题

旧模型给所有元素固定的全局最大槽位 RLE=9 / block=6。每个已知 `(SG,N)` 的完整候选库实际上能给出更小的最大编码长度，且与目标 profile 无关。两个 tokenizer 都应使用这个信息，才能检验压缩收益是否超过容易消除的 padding 成本。

## 实现边界

在新 `packed_words.py` 中继承冻结 `DirectWordModel`，不改其参数或字段嵌入。

1. 从各 context 的完整候选计算 `context_max_words`，不是从目标词数计算。
2. `valid_slots = Z>0 and slot_index<context_max_words[context]`。只遮盖/训练这些槽位；其中真实 PAD 仍未知并需预测。
3. 保留每个元素内的 slot embedding，先生成原来的 `[B,E,L,D]` 特征，再把每个样本有效槽位拼接、仅 pad 到当前 batch 的最大有效总长度，送入 Transformer。
4. 用索引 scatter 回原位置，继续使用现有 word head 和完整候选读出。条件已确定不可能非 PAD 的槽位固定为 PAD；不能用目标 `words!=0` 决定有效位置。
5. 新训练入口与结果目录独立，数据包仍为冻结 `count_merge_candidates/data.pt`，对 RLE 和 rlemerge 采用相同打包策略、499,776 参数和预算。
6. 两者都只计算词内实际非零 WP 记录的嵌入，消除原编码器对每个词展开 27 个 WP 浮点特征再乘零的开销。对全部词定义核对稀疏与原密集计算等价，再测训练内存。参数与数学表示不变。

## 最小检查

- 全遮盖后替换真实词不改变输出。
- `valid_slots` 仅由 G/N/元素存在性决定。
- 在 eval 模式，打包版与同权重、使用相同有效 key mask 的 dense 版在有效槽位上输出一致；这是表示打包的语义检查。
- 现有候选分数 gather 和固定 WP 排他解码检查继续适用。

先用 seed17、各10,000步、batch32做验证开发对照，同时记录概率预测、匹配率、峰值内存、耗时和实际有效槽位数。此阶段只读取既有验证集模型结果。若收益消失，停止把当前计数块称为已证实的效率改进；若有质量或资源信号，再制定新的复核范围，并明确已有测试集的使用历史。
