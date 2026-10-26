# FLARE式完整训练与验证

用户2026-10-19授权。已完成；完整范围和终止条件见[协议](PROTOCOL.md)。

2026-10-20最终复核：两组909步、900验证、700生成及评分全部完成，exit0、GPU释放。AR严格有效169/700、最终80meV新颖达标25/700；FLARE121/700、24/700。读取化学式空格解析错误已使用共享评估器修复，逐条回执与正式汇总已重评分；四项全对分别为162/169和115/121。见[最终报告](RESULTS.md)与[历史重评记录](../../research/evaluation_refresh_20261020/RESULTS.md)。原始生成文本和晶体保留，错误派生评分已替换。

2026-10-19已核验：远程10项CPU检查通过，两个训练分支的真实8B两步检查及optimizer/RNG恢复成功，已进入正式909步训练。15步时两组源权重、数据/源码哈希和材料顺序一致，约5秒/update；训练预计约80分钟，后续生成和评分另计。独立只读审查未发现阻断问题。

- 服务器：`LiuGangGPU:/home/liugang/src/crystalDLM`，mamba项目环境。
- tmux socket：项目`tmp/crystaldlm-tmux.sock`。
- 会话：`flare-full-joint`（GPU0/1联合训练）、`flare-full-ar`（GPU2控制组）、`flare-full-finalize`（CPU汇总等待）。所有会话均使用本轮独立目录。
- 每组：两步真实训练检查→恢复到909updates→28次端到端质量检查→900条验证→700次端到端生成→GRACE评分。联合组700生成分两卡各350，AR组单卡。
- `*-stage.txt`为阶段，`*-exit.txt`出现才表示对应流水线退出；exit0且完整计数核验通过才表示全部完成。
- 原始来源为已有结构条件纯AR检查点，不是8条小试验权重；完整固定7272/900划分不变。

两条流水线都退出后，CPU汇总会自动重算并写入`summary.json`和`RESULTS.md`；未完成或失败的阶段不会被标为完成。`finalize.log`记录汇总是否成功。

## 恢复

先检查当前tmux和GPU进程，确认同一分支没有任务仍在运行。保持源代码、数据、world_size与配置不变。先从`*-stage.txt`和对应日志查清具体失败阶段，再只恢复该阶段及其后续步骤。

- 训练：`run.py --mode train --arm <arm> --out <experiment>/<arm> --resume`；联合组仍需两进程torchrun和GPU0/1，AR组仍单进程GPU2。自动恢复latest.pt的optimizer和所有rank RNG，日志回到保存步。
- 验证：重复`run.py --mode validate --arm <arm> --out <experiment>/<arm>`，自动保留已写入且身份匹配的900逐条记录。
- 生成：重复原`evaluate.py`命令并加`--resume`，保持num-shards/shard-index/limit完全相同；不会覆盖已有有效身份回执。
- 评分：重复既有`grace_score.py --mode score --run-dir <evaluation-directory> --resume`，复用逐条评分记录。
- 阶段重启时，旧exit文件需要保留为历史记录后移出当前exit文件名，避免把上一次退出当作新进程完成。汇总程序只在确认两分支当前进程已退出后运行。

完整启动命令保存在`launch_arm.sh`，当前运行不得重复调用启动脚本。不会修改旧实验检查点，最终训练权重仅在服务器保存。
