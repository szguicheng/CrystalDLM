# AR warm-start 与 predictor alignment V9

状态：`predictor_align_1024_001`因CPU训练misconfiguration终止；`predictor_align_1024_002`因评估器对带尾随句点的数值token抛`ValueError`而失败。两次partial原始证据均保留。`score.py`现仅把无法转float的晶胞长度字段标False，不改raw、不改容差；20项`test_checks.py`通过。下一个全三臂输出候选为`predictor_align_1024_003`，尚未提交。

## 问题与三臂对照

从同一个已完成的 `rebalance_001/ar/checkpoint.pt` 终点建立三个顺序运行的分支：

| 分支 | 训练模式 | diffusion prediction offset |
|---|---|---:|
| `ar_control` | AR | 0 |
| `hybrid_same` | hybrid | 0 |
| `hybrid_shift` | hybrid | 1 |

三臂都从同一个 AR state fingerprint 开始、seed 17、AdamW 和 `train_phase` 配方不变。两个 H 分支参数张量及其形状完全相同；唯一差别是用于扩散损失与采样的预测位置。已有 AR checkpoint 是显式迁移输入：逐文件核验其 V8 `source_snapshot` 与旧 receipt、核对 `rebalance_001/ar/status.json` 和 checkpoint 最终 fingerprint；记录旧/新源码 SHA。新训练的 status receipt 与 checkpoint `config` 都写 `prediction_offset`。活动 checkpoint 恢复走 `Hybrid.restore_checkpoint`：有字段时验证并恢复0/1；无字段只接受 V8 同位置 legacy `model.py` SHA `d7abb6bcc6da9b339408ee928caef4b80a4cf8a40204a0b61b8a68b7c9aca79a`，并按offset0处理，未知旧来源拒绝。AR→H迁移先恢复AR源offset0，再显式设置目标arm offset。offset 不加入 state_dict，因此不增加参数或改变键形状。

## 预测位置与可见性

offset 0 保留当前 hybrid `x_i <- h_i`。offset 1 使用 `x_i <- h_(i-1)`，即第一个晶体 token 由最后一个 `<CRYSTAL_START>` 查询位置预测，最后一个由输入块倒数第二个位置预测。扩散 attention/time span 从 `<CRYSTAL_START>` 开始并覆盖全部 K 个噪声位置；START 查询可见同一块全部 noisy/released token，并带当前 k embedding。文本 lead 的标签仍由原因果 AR target positions 预测，START 标签对应的前一文本查询保持因果，不能看到 START 或噪声块。

训练和采样共用同一预测位置映射。offset 0 必须逐位置复现原行为；offset 1 只改变 hybrid 扩散分支。AR 文本、PLaID codec、tokenizer、数值容差和物理/读取审计保持不变。

## 更新与冻结数据

`rebalance.joint_stream(train, 17)` 每个连续16例优化器批次只对应一个 condition material。已有无GPU实测为：首512批只覆盖512/1024训练材料；首1024批覆盖全部1024材料且每项一次。因此三臂都固定为 1024 更新，每臂 16,384 个样本，使用同一份 `training_exposure.jsonl` 顺序回放。冻结前逐批核验 1024 个材料各出现一个批次，每批为 8G/4E/4Q；E_varied 计入 E、Q1/Q2/Q_varied 计入 Q。启动时将 `data_002/manifest.json` SHA 固定核对为 V8 manifest 记录的 `e3a5a86e58eaae871fa2718d3c9f646551cb297047ef724bf841b24bc48c0004`。

扩散噪声使用既有 `train_phase` joint policy：50% k=32，其余在 k=1..31 均匀采样。训练外的主采样仍为 K=256、32步、temperature 0.8、随机每步 1/k 释放。G 的 AR trigger 监督保留，READ_TEST 问法不加入训练。

## 评估与停止边界

每臂完成 1024 更新后，执行全部128个原 validation 材料、extra reading 128、old8 回归8；按 original64/later64 分开汇总，保存所有原输出、失败类别、配对变化、初末指纹、源码和曝光 hash。禁止重试、修值、筛样本或放宽标准。最终 test256 不读、不训练、不用于选择。该三臂仍是一个 seed 的小机制试验，不构成正式目标完成，也不声称完整复现任何外部模型。

## 本地小检查

`test_checks.py` 核对 offset0 默认回归、offset1 首末预测位置、mask 标签对应、START 双向可见而 causal lead query 不可见、state_dict 键/形状不变，以及保存/加载后从 checkpoint config 恢复 offset；另核对 malformed trailing-period number 严格记为长度字段失败而不抛异常。V9 CPU误跑确认runner每臂`restore_checkpoint`后必须先迁移到CUDA再传给`train_phase`；现已加`model.cuda().train()`、参数device断言与逐臂device启动日志，且在共同`train_phase`入口拒绝非CUDA参数并把实际parameter/device index及计时方式写入receipt。新的完整三臂输出使用`predictor_align_1024_003`。

## 执行纠正记录

`predictor_align_1024_001`对应Slurm 601839仅执行了6个AR更新后被取消，状态`CANCELLED/00:11:07/0:0`。本地和远端原始状态、6条曲线、16384行冻结曝光、源码快照与Slurm logs均保留；无checkpoint、summary或validation。根因是`Hybrid.pretrained`与strict restore均在CPU，而旧runner只在训练结束后重新加载checkpoint时调用`.cuda()`；`train_phase`没有设备迁移。计算节点PID使用约320% CPU/25.8GB RSS，GPU过程显存416MiB、GPU总显存442MiB且0%利用率，6步耗时84.7–99.3秒。该目录只标记执行配置错误，不作为模型效果证据。三臂、offset、种子、预算、冻结曝光和评估合同未变；修复后的唯一新运行目录是`predictor_align_1024_002`。

`predictor_align_1024_002`对应Slurm 601858运行到`FAILED/01:11:15/1:0`。AR臂完整validated；hybrid_same已完成1024训练并将核心128验证跑到66行。`run.py`准备打分`Q_no_crystal`时，`score.py`的晶胞长度正则捕获到`7.02.`并在`float()`抛异常。第67个当前record尚未写入evaluation JSONL；66条既有raw行、1024更新checkpoint、exposure/source snapshot、partial summary和stack trace均保留。没有完整Hsame分数，hybrid_shift未启动。修复只在长度转float抛`ValueError`时把该字段标为False继续评估；不编辑回答、不放宽精度、不重采单例。新完整运行如获准使用独立目录`predictor_align_1024_003`，由同一V8 AR终点三臂同起点重做完整对照。
