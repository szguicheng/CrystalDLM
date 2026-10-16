# 冻结模型的独立保留集复核结果

## Material Passport

2026-09-09；Burgundy Slurm534769，队列020–022。固定训练模型，评价阶段无拟合、无checkpoint选择。规则为 `PROTOTYPE_REPLICATION_PROTOCOL.md`。

## 数据与模型锁定

预先保留的6620个候选与训练集、原2048个开发候选不重叠。按相同规则，5343个可评价、1274个无内部几何自由度、3个在标准化后对称像原子匹配检查中失败。3项失败均复现，成员资格未变。

checkpoint SHA-256：`2051b039f809199e6c311c70e7fa78b218c94d7fdbfeb2ea2350ac94439c75d2`。评价核对了该摘要、训练config摘要、全部训练源码摘要及输入CSV摘要。噪声seed32179，每结构8次，sigma=0.1Å。

## 结果

| 指标 | 数值 |
|---|---:|
| identity归一化MSE | 0.00976237 |
| 模型归一化MSE | 0.00281572 |
| 相对MSE下降 | 71.16% |
| 模型−identity差值的配对结构95%区间 | [-0.00704825, -0.00684215] |
| 平均误差改善的结构 | 5248 / 5343 |
| 同权重ASU/edge-orbit最大向量差异 | 1.72e-6 |

通过了预设的单次固定模型复核判据。结果与开发集的72.36%下降接近，支持离散prototype＋指定晶格作为条件的单级几何去噪接口在未参与模型选择的数据上可用。

这不是多个训练seed的验证，也不是完整扩散采样。MSE改善是模型相对identity的学习效果，不是tokenizer相对ASU图的精度提升；两条路径维持相同预测。新增边压缩的资源收益/时间代价仍以 `ASU_CONTROL_RESULTS.md` 为准。

原始结果和逐结构记录见 `prototype_full_model/replication_result.json` 与 `replication_predictions.jsonl`；锁定信息和失败审计见 `prototype_full_corpus/replication_lock.json`、`replication_manifest.json`、`replication_failure_audit.json`。
