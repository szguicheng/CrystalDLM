# 联合配比实验 V8：完整对照结果

Slurm job `601653` 于 2026-10-03 正常结束：`COMPLETED`，elapsed `02:41:16`，exit `0:0`。远端和本地 `rebalance_001/summary.json` 均为 `status=complete`。两模型各完成 4096 次联合更新；均从对应 `recovery_001/<mode>/joint` 终点恢复，训练 seed 为 17。H、AR 的训练初始指纹分别与各自 recovery 终点一致，source snapshot 13 个源文件逐项匹配训练 receipt。

冻结 exposure 共 65,536 个样本、1,024 个训练材料，每个材料 64 次。逐 16 个样本核验全部 4,096 个优化器批次，均为 8G/4E/4Q；计数为 G 32,768、E 16,384（含 E_varied）、Q 16,384。训练源文件 `training_exposure.jsonl` 的 SHA-256 为 `2d0773845491266b0d71bd9ff76b40416ac8b91c9ecdebfb7f55038544ee64c0`。曲线中所有已记录点的 task_counts 一致；完整逐批分配由冻结 exposure 序列及其审计支持。H 覆盖 32 个噪声级别；AR 不使用扩散噪声级别。

物理有效性使用 `orbit_membership_v3`，文本读取使用 `named_axis_v2`。全部原始评估记录保留；本地 `rescore.score_evaluation` 重算与 v2 已存输出逐行一致，核心和 old8 的行数、ID 顺序、分组及 raw SHA-256 均核对通过。test256 未读取或使用。128 个 validation 材料沿用 original64/later64 分组，两组都已在前轮和本轮使用，不能视为新测试。

| 指标 | H / hybrid | 纯 AR |
|---|---:|---:|
| 有效生成 G | 14/128 (10.9%) | 62/128 (48.4%) |
| 参考结构读取 Q_reference | 128/128 | 128/128 |
| 参考编辑 E_reference | 111/128 (86.7%) | 119/128 (93.0%) |
| 严格闭环 closed_loop | 12/128 (9.4%) | 59/128 (46.1%) |
| 流程到达 flow_complete | 13/128 (10.2%) | 61/128 (47.7%) |
| 替换结构读取 Q_swap | 127/128 | 126/128 |
| 无结构读取 Q_no_crystal | 0/128 | 0/128 |
| 额外问法读取 | 127/128 | 111/128 |
| old8 生成 / 闭环 | 3/8 / 3/8 | 6/8 / 6/8 |

`flow_complete` 与 `closed_loop` 是不同指标，保留分列。H 生成分类为有效 14、触发或 END/PAD 边界失败 36、解析失败 49、结构或解码失败 29。AR 为有效 62、触发或 END/PAD 边界失败 0、解析失败 31、结构或解码失败 35；其中 22 个结构/解码失败含条件不匹配原因。失败全部保留在 128 分母中，没有修值或重试。

额外读取的自动 `reader_v2` 下降集中在第一种问法。AR 为 111/128，较前轮 125/128 少 14 个自动判对；17 个失败全部来自 “For the structure currently in front of you...” 这一问法（32 例中），其余三种问法各32/32，且没有模型调用异常。H 为 127/128，唯一自动失败也是第一种问法，自动评分将其标为 sg 错误；其余三种问法各32/32。自动分数及原始记录保持不变。

对这17例原文的语义补充核对显示：9例没有给出orbit数，6例明确报错orbit数（其中1例同时报错SG），1例只有SG错误；mp-567545的所有事实均正确，但 `reader_v2` 未识别其表达。16例采用 “cell population is N” 的答案中，N实际正确，但当前v2解析器不识别该写法；完整正确的mp-567545还使用了未被解析器识别的 “independent orbits are 5”。因此不能把17例全部称为模型事实错误：至少mp-567545是parser-only失败，原自动结果仍固定为111/128。本补充只是对原文的人工核实，不是统一重评分或新指标。H唯一失败为实际SG错误。V9主reader仍冻结为 `named_axis_v2`；本轮不改评分器，也不把READ_TEST问法加入训练。

| 分组 | 模型 | G | Q_reference | E_reference | closed_loop | flow_complete |
|---|---|---:|---:|---:|---:|---:|
| original64 | H | 8 | 64 | 57 | 7 | 7 |
| original64 | AR | 30 | 64 | 62 | 29 | 29 |
| later64 | H | 6 | 64 | 54 | 5 | 6 |
| later64 | AR | 32 | 64 | 57 | 30 | 32 |

同一轮 V8、同一材料逐例配对时，AR 相对 H：G gained/lost `51/3`，E_reference `12/4`，closed_loop `48/1`，flow_complete `50/2`，Q_swap `0/1`；Q_reference 和 Q_no_crystal 均无变化。

与各自前轮 recovery 在相同 128 个材料上配对，变化如下。数字是 gained/lost 例数；验证 seed 沿用相同的材料确定性设置。

| 模型 / 分组 | G | E_reference | closed_loop | flow_complete |
|---|---:|---:|---:|---:|
| H 全部 | 11/0 | 39/4 | 10/0 | 11/0 |
| H original64 | 8/0 | 15/2 | 7/0 | 7/0 |
| H later64 | 3/0 | 24/2 | 3/0 | 4/0 |
| AR 全部 | 13/7 | 17/7 | 14/6 | 13/8 |
| AR original64 | 8/5 | 8/2 | 10/5 | 8/6 |
| AR later64 | 5/2 | 9/5 | 4/1 | 5/2 |

H 前后总数为 G `3→14`、E `76→111`、closed_loop `2→12`；纯 AR 为 G `56→62`、E `109→119`、closed_loop `51→59`。本轮相对前轮同时增加联合更新预算并改变 G/E/Q 配比，不能将前后差异单独归因于配比。只有一个训练 seed。

纯 AR 的 G 为 62/128，仍低于 50% 门槛所需的 64/128；混合 H 的 G 与闭环也都低于门槛。因此本轮对照不构成目标达成，AR 结果不能替代混合模型目标。

| 原始输出 | SHA-256 |
|---|---|
| H validation_128/evaluation.jsonl | `0d573c0fd5360c81bc8fc2a2e457b63df630d565c8b5209fa233629d74929aaa` |
| AR validation_128/evaluation.jsonl | `e0bdec9d956dff0b27fcb12499ff8fad13293f17ded498b6719c1ba28c4c9ace` |
| H validation_128/unseen_reading.jsonl | `f52863779c2ea7c197b44a1c4ceb41e5484a10116da46d3bc1cbf8c28e59ab2c` |
| AR validation_128/unseen_reading.jsonl | `4aea833a3603dfa1a9281e60af2949315202ed4eb224838b6429c7ecf40ab683` |
| H old8_regression/evaluation.jsonl | `1834b5235abdab84175ab0ec4e197fbd3f3d3ad119d0f46f4cc88a03f0e53b4f` |
| AR old8_regression/evaluation.jsonl | `f6ca203a3bf468378d2357f20646cd7acfbcb1477fdc9021b6e8d5207f8c726c` |

## 模型与收据

| 模型 | 初始指纹（等于 recovery final） | V8 最终指纹 | 训练秒 | 峰值显存 |
|---|---|---|---:|---:|
| H | `9b5081d45aaebb18e234bc8e5d3d913f9d1c7d443b4a2c1575a5e2385f64664c` | `5e33596b7296806a7bca26a0b4d2843a2dd95927e269e633f42c1aa5a0ecf755` | 3206.1 | 11,735.9 MiB |
| AR | `77f269f9608e37dc535b65bdc5d8f7526c7c0cd6fc2db79268f5c7acca35b66f` | `7567d5dd878a1e02a51b05a9ec4ecd67e68cd1a79da88235c975e1eeb4b21c85` | 3154.8 | 11,736.1 MiB |

主收据位于 `rebalance_001/{hybrid,ar}/status.json`，H 的逐项训练核验另存于 `rebalance_001/hybrid/training_verification.json`。H 的 `verified_analysis.json` 和两模型的 `rescored_summary_v2.json` 保留原始输出及评分哈希。`rebalance_001/summary.json` 为完整终态。

候选 `diagnostics/rebalance_probe.py` 未运行，不属于上述结果；未提交或启动额外 GPU 任务。
