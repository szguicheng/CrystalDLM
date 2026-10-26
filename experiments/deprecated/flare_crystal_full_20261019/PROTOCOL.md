# FLARE式Qwen3-8B完整固定子集训练与验证

2026-10-19 用户授权“做完整的训练和验证看看结果”。完整指已冻结的1/5英文标注数据：7272训练、900验证；不重新标注或扩展原始四万多条。

## 两个训练分支

共同起点为`structure_conditioned_ar8b_20261017/run_001/checkpoint.pt`：Qwen3-8B预训练权重＋已经训练好的结构条件AR适配器。**不使用8条小试验过拟合后的权重**。从同一个源状态分别继续：

1. `ar`：G文字和晶体全部next-token AR；Q读取任务也是AR。
2. `flare`：G干净序列保留完整AR，再加互补遮盖的晶体去噪；Q保持因果AR读取任务。共享核心、LoRA与输出头。

固定源数据hash、seed42及`train.epoch_order`顺序，完整一轮909updates，每更新8材料（G/Q各一）。每材料损失为(G+Q)/2；联合G中干净AR总和与互补晶体去噪总和按同一个assistant目标token数归一化。沿用小试验LR5e-5 constant、AdamW fresh state、weight_decay0.01、clip1、BF16和原LoRA参数集合。固定最终909检查点，不用验证集选择多个配方。本轮是源AR已训练一轮后再训练一轮，比较对象为同预算AR继续训练；源模型历史结果仅作参考。

实现复用已核验的4目标token块、前缀anchor、相同position IDs、clean因果/noisy仅见过去clean及本块的模式。保留先mask目标再取前位logit的无泄漏规则。没有逐行复现FLARE当前仓库中存在分歧的shift/mask路径；没有换Qwen3.5/GDN，也没有使用外部diffusion权重。历史小试验代码固定、来源hash记入运行身份。

## 验证与生成

- 全部900条验证材料：分别计算冻结检查点G/Q验证损失，固定每材料的mask seed，逐条保存。不给验证材料更新梯度，也不要求逐条复原结构作为生成成功标准。AR和联合总loss不同，只有共同cleanAR/Q分项可直接比较。
- 主生成评价：沿用7个空间群（1、15、38、119、143、194、216）各100次、固定逐次seed，共700请求/模型。ehull目标上限0.08eV/atom。用户prompt只含结构与性质要求，**文字lead由当前模型生成**，不提供参考lead或参考晶体。
- AR组：文字和晶体AR生成。联合组：文字AR，晶体每块4token去噪，每forward按采样概率提交一个位置，完成块后接入clean前文。温度0.8/top-p0.95；文字最多256、晶体最多544含END。END停止，尾部PAD不作为实际生成目标；双方保持一致。无正文修复或失败补采。
- 有效晶体之后，模型读取**实际生成**的晶体并回答formula、SG、cell atom count和independent orbit count，4字段与解析事实比较；无效晶体注明跳过。此读取率单独报告，不代替结构或性质分数。
- 保存全部700原始结果，包括未触发、未结束、格式/结构错误。只有严格有效者进入原有GRACE评分流程，失败仍保留在700分母。

## 性质、去重与新颖性

复用既有`structure_request_eval_20261017/grace_score.py`、GRACE-3L-OAM-L、MP2020能量修正、固定MP2023凸包、FIRE fmax0.05/300steps、已有StructureMatcher设置和共同7272训练参考结构。主报告≤80meV/atom且满足请求空间群、有效、去重、新颖的数量/700；另报0/100meV。结果是GRACE/参考凸包代理ehull，不是新DFT。

## 验收与阶段控制

1. CPU测试：样本分配、损失/注意力、真实微型Qwen train/infer parity、断点身份和生成评分接口。
2. 两步真实训练并恢复，直到909步；FLARE双卡GPU0/1、AR单卡GPU2，共不超过三卡。两组每update有效材料数相同；单卡/双卡浮点归约与遮盖RNG布局不同，记录world_size并禁止不同world_size恢复。
3. 每组训练完成，在七请求上先各四次端到端生成（共28次，避免7次过小造成误拦）。不把程序exit0当质量通过。若28次全部结构无效，保留记录并停止该组700生成，进行诊断；不把这一预检当统计充分的模型否定。
4. 通过小样本质量检查后，完整900验证、700生成及GRACE评分；联合组生成使用两卡分片，每卡350次。两个分支各自的阶段状态与exit记录独立。
5. 汇总909update身份、900验证唯一材料、700唯一尝试、700评分记录、四项生成后读取指标、时间及显存。

## 持久化与存储

代码、环境、数据、权重、缓存、日志和中间结果全部在LiuGangGPU现有`/home/liugang/src/crystalDLM`内；使用项目mamba环境，评分用既有grace环境。新目录`experiments/flare_crystal_full_20261019`，旧数据/结果/检查点不动。启动前磁盘约25GiB空闲；新增两个分支latest/final和评分输出预计在该余量内，写大文件前重新检查。

每50步原子保存本轮latest检查点，包含optimizer及所有rank RNG；可恢复至最近保存点，日志截断到相应update。验证、生成和评分逐条保存，可断点续跑。固定源码hash与manifest，改变代码/配置必须新建运行或明确记录迁移。
