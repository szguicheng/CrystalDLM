# AR-understanding 正式训练结果

2026-10-26。训练、final/best 两个 checkpoint 的完整 validation/test 评价均已完成，`stage.txt=complete`、`pipeline.exit=0`，GPU 已释放。逐条材料 ID、参考回答、四字段评分和 NLL 汇总已复核，见 [result-audit.json](result-audit.json)。这是正式保留集结果，不是此前 32 条过拟合检查。

## 设置与完成情况

- 原始 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；LoRA rank32/alpha64，BF16，学习率2e-4，seed42。
- 三张 H20，global batch12；36,362 条 training、4,501 条 validation、4,602 条 test，保留原冻结划分。
- 固定 **3 epoch、9,093 updates**，没有早停；92,303,360 个可训练参数。正式训练的初始化参数 SHA 与原始基座诊断初始化一致，没有从32条过拟合权重接着训练。
- 正式训练及过程/末次验证共约 **3小时30分**，随后四组独立评价约 **40分**；从正式训练开始到全部结束约 **4小时11分**。流程于 2026-10-26 09:29 UTC（PDT 02:29）完成。

## 最佳与最终 checkpoint

最佳 checkpoint 是 `full/checkpoint-7200`（约epoch2.38），选择依据为训练过程中的最低 **validation token NLL**；最终 checkpoint 是 `full/final`（完成3epoch）。以下 NLL 全部来自重新加载 checkpoint 后的同一独立评价入口，单位 nats/目标token。

| checkpoint | Validation 四字段全对 | Validation NLL | Test 四字段全对 | Test NLL |
| --- | ---: | ---: | ---: | ---: |
| validation选出的 best，step7200 | 4,498/4,501（99.9333%） | 1.238303e-4 | 4,597/4,602（99.8914%） | 1.713649e-4 |
| final，step9093 | 4,498/4,501（99.9333%） | 1.279163e-4 | **4,600/4,602（99.9565%）** | **8.223095e-5** |

两版完整回答 exact match 的计数均与四字段全对计数相同。Final 在本次 test 少错3条；best 的名称仍按预先确定的 validation 标准定义。两个 checkpoint 与评价都保留，本次汇总未重新选择或替换 checkpoint。

Final 的 test 分项如下：

| 字段 | 正确/总数 | 准确率 |
| --- | ---: | ---: |
| 化学式 | 4,602/4,602 | 100% |
| 空间群编号 | 4,601/4,602 | 99.9783% |
| 晶胞原子数 | 4,602/4,602 | 100% |
| 独立轨道数 | 4,601/4,602 | 99.9783% |
| 四字段全部正确 | 4,600/4,602 | 99.9565% |

## 错误样本

Final test 只有两条错误：

| 材料 | 输入证据 | 正确答案 | 预测 |
| --- | --- | --- | --- |
| `mp-37207`，Li10Ga2O8 | `Spacegroup: P42212` | 空间群94 | 空间群97 |
| `mp-1227831`，Ba1Sr3Co4O12 | `Sites (20)`，18行Wyckoff轨道 | 独立轨道18 | 独立轨道20；其余三字段正确 |

第二条预测的轨道数恰好等于晶胞原子数。Best test 除这两条外，还把三条 `P63cm` 结构的空间群185预测成180。Validation 两版错的是相同3条，全部是空间群符号到编号的错误：`Pba2` 32→28、`P43` 78→76、`P4332` 212→215。所有原始回答保留在对应 `predictions.jsonl`。

## 曲线与解释

训练过程的 validation token NLL：step600 为0.0050087，epoch1结束为0.00036916，epoch2结束为0.00012963，最低点step7200为0.00012285，最终为0.00012783。后期处于低误差平台并有小幅波动；按协议评价了best和final，没有改epoch重训。

现有曲线：[训练答案CE](full/training_train_answer_ce.png)、[验证token NLL](full/training_eval_token_nll.png)。训练日志最后窗口的答案CE约1.68e-6；它是训练批次统计，不是重新计算的全训练集NLL。

重新整理的[training/validation合图](loss_curves/understanding_training_validation_loss.png)使用统一答案token平均CE口径：909个training窗口、18次全量validation，原始记录不平滑，主图为对数纵轴，局部框放大后期validation。[PDF](loss_curves/understanding_training_validation_loss.pdf)、[源数据CSV](loss_curves/loss_values.csv)和[绘图元数据](loss_curves/plot_metadata.json)同时保留。CSV另含原生Trainer的 `loss/eval_loss`，避免将四位小数舍入后的training `loss=0.0`解释为真正的零误差。

这批结果支持的结论是：**当前“PLaID→化学式、空间群编号、晶胞原子数、独立轨道数”的读取任务已基本学会，基准接近饱和。** 化学式和原子数直接出现在输入头部；空间群主要需要符号到编号的映射，轨道数可由轨道行计数得到。因此该结果适合作为结构事实读取基线，不衡量复杂晶体推理、性质预测或晶体设计能力。

本次未运行新的模型或修改数据/评分器，仅复核已保存的评价回执。没有原始未微调Qwen在同一test上的自由生成成绩，不能从本表单独计算微调带来的准确率增益。固定划分仍按源结构哈希划分，没有新做跨划分的StructureMatcher去重；结果不代表严格结构新颖的测试表现。

Understanding 这条线已完成；Phase3的AR-design仍须等完整Design数据就绪。后续主比较应继续围绕Design的有效性、条件满足、性质、唯一性与新颖性展开。
