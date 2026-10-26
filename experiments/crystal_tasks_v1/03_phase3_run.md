# Phase 3：Understanding 与 Design 补标并行

最新状态（2026-10-26）：Understanding 已完成3epoch/9093updates及best/final两组完整validation/test，exit0。Final test四字段全对4600/4602（99.9565%），best为4597/4602（99.8914%）；详见[正式结果](phase3_understanding_20261026/RESULTS.md)。下文保留启动时记录。

2026-10-25 PDT / 2026-10-26 UTC，负责人通过闸门 2，并明确恢复 Design 缺失描述标注。三卡训练、global batch 12 保持不变。

## Understanding 执行链

服务器：LiuGangGPU，项目实际位于 `/data1/users/liugang/crystalDLM`。独立 tmux 为 `crystal-understanding-phase3`；运行根目录为 `experiments/crystal_tasks_v1/phase3_understanding_20261026/`。

1. 原始 Qwen3-8B 单 GPU 前向/反向预检已通过：92,303,360 个可训练参数，峰值显存 17.22 GiB；没有优化器更新。回执在 `phase3_preflight/check8b.json`。首次按脚本路径执行发生 Python 包路径错误，改为 `python -m` 后通过，失败日志保留。
2. 原生 LLaMA-Factory trainer 的微型 CPU 集成检查通过，确认返回 token 加权 `eval_token_nll` 并记录分段 CE。验证通过 `gather_for_metrics` 去除分布式尾批填充；训练目标、优化器和梯度累积沿用原生实现。
3. seed42 从完整训练集固定抽取 32 条，运行 60 个诊断 epoch，共 180 updates。每 30 updates 评价和保存，最多保留 3 个诊断 checkpoint；不将这 60 epoch 当作正式训练。样本 ID、初始化参数哈希写入该运行 `run_meta.json`。
4. 重新加载诊断 final adapter，greedy 生成同 32 条并精确计算答案 NLL。进入全量的条件：四字段全对 32/32、完整参考回答复现 32/32、NLL ≤ 0.05 nats/token。失败时 supervisor 退出，禁止自动进入全量。
5. 检查通过后从**原始基座重新初始化**，运行完整 36,362 条训练 / 4,501 条 validation、固定 3 epoch。每 600 updates 评价与保存，epoch 边界另评价和保存；最低 `eval_token_nll` 与 final 都保留。
6. 训练后自动评价 final 的完整 validation/test；best 与 final adapter 权重不同则再评价 best。每次包含 greedy 四字段、精确匹配计数及 token NLL；运行结果以各 summary/run_meta 为准。

截至 2026-10-26 05:21 UTC：32 条检查已通过，四字段全对和完整回答复现均为 **32/32**，精确答案 NLL 为 **4.781482273256619e-6**（1,161 个目标 token）。原生三卡验证统计也恰为 1,161 个目标 token。全量训练已实际进入更新阶段，使用 36,362/4,501 条、三卡/global12；其初始可训练参数 SHA 与诊断运行的原始初始化完全相同，证明没有接着过拟合权重训练。全量尚未完成。

启动入口为 [launch_understanding.sh](launch_understanding.sh)，训练和评价分别使用 [train_ar.py](train_ar.py)、[evaluate_ar.py](evaluate_ar.py)。本次训练使用 GPU 0/1/2，后处理最多两卡；第四张卡不使用。所有 Python 通过 `python.sh` 在环境激活后固定项目内缓存和临时目录。

## Design 补标执行链

本机活动目录：`data_preparation/crystal_tasks_v1/active_resume_20261025/`，tmux 为 `crystal-design-caption-resume-20261025`。

- 复制并校验服务器 SQLite 一致性快照，仅重绑定输入文件的本地绝对路径；37,293 个输入的请求哈希逐条核验，原 1,312 个成功 ID 保留。
- 沿用原 `deepseek-v4-flash`、thinking enabled、reasoning max、stream、8 workers、每轮最多 4 次尝试。使用原本机凭据，不传输密钥到服务器。
- 历史描述 8,172 条仍复用。恢复后的成功描述持续写入本机 SQLite；服务器旧进度库保持 1,312 条快照，并记录本机为当前唯一 writer。
- 当前出现 SSL EOF、读超时等 transient 网络错误，由原有四次重试处理；耗尽重试的条目保留失败记录，不能当作已补齐。
- 本次只启动标注，未执行 `finish.sh` 或 dataset finalize。当前训练依赖的冻结 JSONL 和 manifest 保持不变；补标完成后再处理正式数据版本发布。

两个长任务均独立运行。Phase 3 完成后仍需闸门 3 汇报；Hybrid 的 END 修复及训练仍受后续闸门约束。

05:21 UTC 补标快照：新增描述成功 1,837 条（本次恢复增加 525 条），加历史复用共 10,009 条；仍缺 35,456 条，包含 4 个耗尽本轮四次 transient 重试的材料。当前计数以活动 SQLite 为准，快照见本机 `data_preparation/crystal_tasks_v1/active_resume_20261025/progress-current.json`。
