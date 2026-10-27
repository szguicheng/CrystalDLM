# 分离 Design / Understanding 并准备全量 JSONL

Status: complete
Authorized: 2026-10-25

## 2026-10-26 完整数据就绪

完整Design已导出并通过45,465条全量校验，train/validation/test为36,362/4,501/4,602，未重新划分。本地与远程9个任务/metadata JSONL哈希一致；实际模板长度最长799，cutoff2048不截断。当前完整版本位于 `data_preparation/crystal_tasks_v1/exports/design_full_20261026`，manifest SHA为 `360c68297db7e14e9db38815c68a6b18e2f38d61d17063133694c20d2f6e3969`。旧 `datasets/crystal_tasks_v1` 保持不变，AR-design和Hybrid-design配置均引用这个完整版本。

AR-design已通过32条诊断并进入全量训练，见[当前运行](../../../experiments/crystal_tasks_v1/04_design_phase3_run.md)。以下记录保留此前各阶段状态。

## 2026-10-26 补标完成

主轮完成37,284条后留下9条失败（4条网络错误、5条HTTP499）。使用既有 `retry_failed=True` 入口作一轮仅失败条目的有界重试，9条全部成功；新增描述37,293/37,293，历史复用8,172，共45,465条，失败和进行中均为0。重试回执：`data_preparation/crystal_tasks_v1/active_resume_20261025/retry_20261026_9/result.json`；原37,284成功ID全部保留。完成于2026-10-26 12:05 UTC。

完整journal仍在本机；服务器旧1,312条库保留作快照，完成状态另写入 `data_preparation/crystal_tasks_v1/annotation-completion-20261026.json`。正式Design JSONL导出、全量校验及数据版本发布仍未执行，不能直接运行服务器旧快照的finalize。

## 2026-10-25 恢复执行

负责人重新授权继续补标。当前唯一 writer 在本机 `data_preparation/crystal_tasks_v1/active_resume_20261025/annotation_run/annotation.sqlite3`，tmux 为 `crystal-design-caption-resume-20261025`；服务器旧库保持1,312条快照，并用 `annotation-resume-status.json` 标记writer归属。保留原模型、提示词、8并发及每轮4次尝试。05:21 UTC时新增描述成功1,837条，历史复用8,172条；正式Design JSONL仍未发布。不得在活动writer运行时直接启动服务器旧库，也不覆盖正在用于Understanding训练的冻结manifest。

以下为初次准备记录，旧目录位置以最新运行记录为准。

用户确认 Understanding 改为仅输入 PLaID 晶体并回答原四字段；Design 保持空间群＋Ehull 上限到英文结构描述＋PLaID。两个任务独立，使用 LLaMA-Factory JSONL 模板和全部可用 train/validation/test 数据，不启动训练。

- 完整可用材料：36,362 train / 4,501 validation / 4,602 test，共 45,465；保持原划分。原采集 45,629 中已有 164 条预处理错误，不属于本轮采样。
- Understanding：服务器全量导出和检查通过；现有 Qwen3-8B tokenizer 下最长 622 tokens。
- Design：复用已整理的 8,172 条描述；另 37,293 条正在用原标注流程补齐。原模型、提示词和 8 并发，逐条持久化，不重复调用已有材料。
- 本地 tmux：`crystal-full-data-20261025`；本地代码、日志和 SQLite 在 `experiments/design_understanding_full_20261025/`。`finish.sh` 补齐后自动导出、验证、同步至服务器并再验证。
- 服务器目录：`/home/liugang/src/crystalDLM/experiments/design_understanding_full_20261025/`。
- `data/manifest.json` 的 `both_tasks_ready` 和全量 `validation-report.json` 的 `passed` 才表示两任务都完成；当前仍为 `understanding_ready_design_awaiting_captions`。

任务定义、输入输出样例、LLaMA-Factory 登记、原始来源和复现步骤见 [README](../../../experiments/design_understanding_full_20261025/README.md)。没有安装训练框架或启动模型训练。
