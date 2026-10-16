# 参考论文实际训练数据规模核对

核对日期：2026-09-23。范围沿用 research/model_comparison.md 中的八篇论文。核对正文数据集/实验设置、相关附录和必要的作者代码/数据；不把数据集总量、生成样本数、性质预测实验或模型参数量当成训练结构数。

## 我们实际用了多少

- 每个模型训练集：**6,000 个结构**。两种表示、两个种子复用同一训练集，并不是四组不同数据。
- 验证集：512；最初测试集：512；续跑新增测试集：512。新增测试数据从未加入训练。
- 每个模型：4,000 optimizer steps，batch 32，共 128,000 次样本抽取；除以 6,000，约 **21.33 个等效 epochs**。训练是随机抽样，这表示平均暴露次数，不是逐个无放回遍历的实测轮数。
- 参数量：2,357,727。续跑只改采样和探索性后处理，没有增加训练数据或继续更新权重。
- 本地配置依据：[explicit_17/config.json](../experiments/explicit_model_pilot_20260922/runs/explicit_17/config.json) 与 [数据检查](../experiments/explicit_model_pilot_20260922/data/summary.json)。

## 逐篇比较

| 模型/论文设置 | 实际训练集规模或可支持的估计 | 相对我们的 6,000 | 证据性质 |
|---|---:|---:|---|
| CrystalFormer v3 / MP-20 | **27,136** | 4.52 倍 | 正文 II.1.2 明确报告 training dataset 数量 |
| WyFormer v4 / MP-20 论文基准 | **27,136** | 4.52 倍 | Appendix D Table 5 与 L.1 均明确写 full MP-20 train；不是总量 45,229 |
| SymmCD v3 / MP-20 作者公开训练数据 | **27,136** | 4.52 倍 | 固定作者仓库 commit 后对 train.csv 流式 CSV 计数；正文总量存在冲突，见下文 |
| DynaCrys v1 / MP-20 | **约 27,136** | 约 4.52 倍 | 正文指定 MP-20 training partition；整数按标准 split 对应，原文没有直接列出该整数 |
| MaskGXT v1 / MP-20 | **约 27,136** | 约 4.52 倍 | 正文说明标准 MP-20 60/20/20 划分；整数按标准 split 对应，原文没有直接列出该整数 |
| WyckoffDiff v4 / WBM 主实验 | **236,838** | 39.47 倍 | 作者 CSV 实测行数；哈希与原始 Dataset 代码一致，默认训练 loader 不去重或筛行 |
| Matra-Genoa v2 / MPAS 大数据模型 | **约 2,401,848（约 240 万，估计）** | 约 400 倍 | 论文给出 2,668,720 结构语料，另说 10% 留验证；取 90% 估计，不能将其称为作者明确列出的精确 optimizer train 行数 |
| WyckoffDiff-Adaptor v1 / MP-20 | **实际训练条数未披露/未确认** | 不算精确倍数 | 论文明确 MP-20、base 900 epochs + finetune 200 epochs；没有足够依据确定它使用哪一份训练 split |

### 「平均用了多少」的可解释口径

最可比的是前五个 **MP-20 实验设置**：平均训练规模约 **27,136 个结构，即约 2.7 万**。这里包含两项标准 split 推定值，因此不应把它包装成五篇均直接报告精确整数的统计。

我们的 6,000 是该规模的 **22.11%**，少 21,136 条；完整 MP-20 train 约是本次的 **4.52 倍**。这些模型很多复用了同一基准数据，五篇论文并不意味着五份独立训练语料。

WBM 的 23.68 万与 MPAS 的约 240 万属于不同数据来源和规模。把它们硬混入一个均值，会被百万级设置主导；若 Matra-Genoa 改选其 MP-only 模型，均值又会显著变化。因此本次不声称存在一个唯一、公平的「八篇论文平均训练数」。用于下一轮可比实验，约 2.7 万的完整 MP-20 train 更有参考价值。

## 不能忽略训练预算

| 设置 | 已核对训练预算 | 解读 |
|---|---|---|
| 我们 | 4,000 steps × batch 32 / 6,000 ≈ **21.33 等效 epochs**；约 236 万参数 | 很短的小规模试验 |
| CrystalFormer | **3,800 epochs**；Table S1 约 484 万参数 | 文中报告约 13 小时/A100；不同架构不直接按小时比较 |
| SymmCD 作者公开 MP-20 配置 | **max_epochs=2,000**，batch 512 | 这是代码配置上限，不冒称某次运行实际达到该轮数；MPTS-52 论文另报 1,500 epochs |
| MaskGXT | **最多 3,000 epochs，validation METRe 早停**；约 248M 参数 | 不是保证跑满 3,000；模型也比我们的试验大很多 |
| WyckoffDiff / WBM | **1,000 epochs**，batch 256 | 文中另有 100-epoch 消融，不能代替完整训练设置 |
| WyFormer / MP-20 | **900k full-batch epochs**，batch=27,136，论文约 11 小时 | 每轮随机化 orbit 顺序/已知序列长度/字段；其训练单位与其他目标不同，不能仅凭 epoch 数比较算力 |

4,000 steps 不是 4,000 epochs。如果数据扩到 27,136 而仍只跑 4,000 steps、batch 32，每个结构平均只被抽到约 4.72 次。扩数据时必须同时考虑训练步数、验证收敛和采样约束。

目前只能说：在 6,000 样本、约 21 等效轮、小模型及当前实现下，显式表示没有优于同预算 Wyckoff 对照。这个结果不足以否定表示的潜力，也不能反过来保证单纯增加数据就会解决语法、计量和晶格对称问题。后续合理的基准是使用完整 MP-20 train、保留独立验证/测试，并对两种表示给予一致且充分的训练预算。

## 特殊口径与不确定性

1. **SymmCD 原文与公开数据不一致。** v3 §5 把 MP-20 总量写成 40,476，并说留 20%+20%；其附录 MPTS-52 也给出 40,476。不能未经确认就宣布它只是笔误。本次采用作者当前公开代码所指向的 train.csv 实测数 27,136，并明确该来源。固定 commit 为 a50bc003e8c5fa2348336dc77477d2439f165fd4；train.csv SHA256 为 9c5a34b3684ff4b86d87a86c1c42298d1d28322ea0f03059d9e80ab6a2b729a7；计数证据见 [JSON](notes/symmcd_train_count_20260923.json)。MPTS-52 的 27,380/5,000/8,096 是 SymmCD 附录直接报告的另一实验，未与 MP-20 混加。
2. **WyFormer 论文实验不等于发布 checkpoint。** 论文明确 27,136。HF 发布配置出现 production_training=true；当前代码生产模式可能合并 train+val，若加载 test 还会合并 test。仅凭配置不能断定某个 checkpoint 是 36,183 或 45,229 条训练；本表只比较论文 benchmark。
3. **Matra-Genoa 的百万级数是估计口径。** 文中报告 MP 115,663、筛选后 Alexandria 2,553,057、合计 MPAS 2,668,720，并说排除 SG1。总数与两分项恰好相加，过滤顺序/最终分项数未完全清楚。再按留出 10% 得到约 240 万实际训练规模。MP-only 的最终 post-SG1 数未披露；用 115,663×90% 得到约 10.4 万只可作规模估计。ORB 的约 72M 是查重库，不是训练集；MatraGenoa3M 是生成数据，不是训练集。
4. **MaskGXT MPTS-52 暂不报精确实际行数。** 原文同时说 chronological split 和 60/20/20，与常见的 27,380/5,000/8,096 不完全一致；没有作者实际 split 文件证据前，27,380 只能作标准版本参考。主表只选它的 MP-20 实验。
5. **增强不增加独立材料数。** 原点变换、等价 Wyckoff 编码和排列打乱会增加模型见到的表示变体，但不能直接算成新增独立晶体。

## 原始来源与详细笔记

- CrystalFormer：[原文 II.1.2、Appendix A](https://arxiv.org/html/2403.15734v3)。
- WyFormer：[原文 §3.1.1、Appendix D/L.1](https://arxiv.org/html/2503.02407v4)。
- SymmCD：[原文 §5、Appendix E/F](https://arxiv.org/html/2502.03638v3)；[固定版本训练 CSV](https://github.com/sibasmarak/SymmCD/blob/a50bc003e8c5fa2348336dc77477d2439f165fd4/data/mp_20/train.csv)；[固定版本 MP-20 配置](https://github.com/sibasmarak/SymmCD/blob/a50bc003e8c5fa2348336dc77477d2439f165fd4/conf/data/mp_20.yaml)。
- DynaCrys：[原文 §5.1、Appendix D/F](https://arxiv.org/html/2608.07401v1)。
- MaskGXT：[原文 §5.1/5.3、Appendix B](https://arxiv.org/html/2606.22866v1)。
- WyckoffDiff：[原文 §4.1、Appendix A/E](https://arxiv.org/html/2502.06485v4)；[初始公开 Dataset 实现](https://github.com/httk/wyckoffdiff/blob/58a5afb432bc38498807fb29357b07a4547275bb/wyckoff_generation/datasets/dataset.py)；[作者训练数据](https://public.openmaterialsdb.se/wyckoffdiff/data/wbm/raw/wbm_train.csv.bz2)。
- Matra-Genoa：[原文 §II、IV.1、IV.3](https://arxiv.org/html/2501.16051v2)。
- WyckoffDiff-Adaptor：[原文 §2.2](https://arxiv.org/html/2601.08115v1)。
- 详细核对：[AR 模型](notes/training_sizes_ar_20260923.md)、[离散扩散模型](notes/training_sizes_diff_20260923.md)、[DynaCrys/MaskGXT](notes/training_sizes_recent_20260923.md)。

本次仅核对与整理数据规模，没有启动新训练或改变既有实验。
