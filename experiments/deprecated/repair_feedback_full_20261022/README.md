# 修正模型：700 组循环生成与 ehull 评分

使用已训练的共享 AR＋diffusion 修正模型，空间群 1、15、38、119、143、194、216 各 100 组请求。输入目标为对应空间群及 ehull ≤80 meV/atom。

每组比较两种方式：直接重新生成、根据结构检查反馈修改。首轮生成共享，各方式最多三次，首个严格有效且满足目标空间群的晶体为该组输出。GRACE 不进入反馈；最终每组只选一个输出评分，失败组也保留在 700 的分母内。

生成后复用原 GRACE-3L-OAM-L 弛豫、MP2020 能量修正及固定 MP2023 凸包参考，计算 ehull proxy、去重、训练库新颖性；主要达标计数要求生成时成功、弛豫收敛、弛豫后结构有效且空间群仍匹配、ehull≤80 meV/atom。唯一新颖计数仅在这些合格候选中确定代表，防止失败组占用代表名额。

## 执行及恢复

从 LiuGangGPU 的 `/home/liugang/src/crystalDLM` 启动（实际位于 `/data1/users/liugang/crystalDLM`）：

```bash
bash experiments/repair_feedback_full_20261022/launch.sh smoke
bash experiments/repair_feedback_full_20261022/launch.sh formal
```

两条命令依次执行，先核验 smoke 再运行 formal。生成使用物理 GPU1/2/3，三个独立分片；随后 GRACE 使用 GPU1/2，前一阶段退出后才启动后一阶段。脚本检查目标卡空闲和 `/data1` 挂载。环境沿用 `envs/crystaldlm` 与 `envs/grace`，均通过 mamba 执行。

`smoke_001` 为独立七组检查，`run_001` 为正式七百组。重复同一命令时，已持久化的组/轮次和 GRACE 结果会校验身份后恢复。中断前尚未保存回执的一轮可能按原种子重放；不会重新挑选已完成轮次的结果。

运行进度：实验根目录 `smoke-stage.txt` / `formal-stage.txt`，结束码对应 `*-exit.txt`，日志在 `logs/`。生成回执为 `run_001/shard-*/groups/sg-*/group-*.json`，每组保存两条策略的完整轨迹。

最终候选导出在 `run_001/selected/{independent,feedback}`，每组对应一个标准生成回执，各自的 `grace_score/` 保存评分。完整结果为 `run_001/scoring-summary.json` 和 `scoring-RESULTS.md`。原 GRACE 汇总保留为补充指标，主指标另行加入生成阶段的约束门槛。

协议见 [PROTOCOL.md](PROTOCOL.md)。本轮只做推理和评分，不修改模型权重；本文件描述执行方式，完成状态以运行回执为准。

## 2026-10-22 启动记录

9项CPU合约测试通过。`smoke_001` 七组、27次实际候选生成、10次反馈重试均进入晶体段；生成、两臂GRACE与汇总完整完成，退出码0。smoke旧版评分汇总代码留存为 `smoke_001/scoring_bridge_snapshot.py`；正式汇总另外补充按空间群划分的主指标，判定规则未改变。

正式任务已在 tmux `repair-loop-full-20261022` 启动，socket为项目 `tmp/crystaldlm-tmux.sock`。小样本累计候选生成耗时约766秒，按三卡扩展到700组粗估生成约7小时；该估算不含模型初始化及后续GRACE评分，正式运行速度以实际回执更新。
