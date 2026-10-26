# 结构条件Qwen3-8B纯AR重新训练与评价

**已完成**：2026-10-17 23:59结束，训练、700次生成及GRACE评分均正常退出，GPU已释放。最终80meV条件下新颖达标12/700（1.71%），严格有效95/700（13.57%）。完整比较见 [RESULTS.md](RESULTS.md)。

2026-10-17获用户授权，协议见 [PROTOCOL.md](PROTOCOL.md)。首次启动约21:10（Asia/Shanghai）。现有原始数据与旧结果保留；本轮从固定原始8B基座重新初始化LoRA，不接续旧ehull-only模型。

输入为实际PLaID目标空间群＋覆盖源标签的ehull上限，全部8172条记录的晶体、文字及读取目标保持不变。五项输入/集成/身份/评价计划测试已在远程项目环境通过。这里的“结构条件”首轮具体为“空间群”，没有加入新的tokenizer或diffusion。

双卡预检已完成：最长样本805tokens，梯度有限且新增token路径非零，检查点重载控制logits最大差异0。正式训练初始trainable-state SHA为 `649672e3ab189712903572eacaea646b33153b655835409b9d8ec8f7c0c15862`，与旧试验的初始状态一致，未复用旧训练后的权重。

## 运行

- 服务器：`ssh LiuGangGPU`
- 项目根目录：`/home/liugang/src/crystalDLM`
- tmux socket：`/home/liugang/src/crystalDLM/tmp/crystaldlm-tmux.sock`
- 会话：`sg-ar8b-retrain-20261017`
- 总入口：`launch_pipeline.sh`
- 训练/生成：mamba环境 `envs/crystaldlm`，GPU0、1；评分：已有 `envs/grace`，生成结束后用GPU0。每阶段启动前检查卡占用。

自动顺序为CPU检查、双卡3update预检、909update正式训练、7次生成工程预检、700次正式生成、GRACE评分。遇到程序失败时保留阶段与退出码，不自动改配方或覆盖结果。

## 查看进度和结果

| 路径 | 内容 |
| --- | --- |
| `stage.txt` / `pipeline-exit.txt` | 当前阶段 / 流水线退出码；无退出文件表示尚未退出 |
| `preflight_001/summary.json` | 梯度、长度及保存恢复预检 |
| `run_001/progress.json` | 训练更新进度 |
| `run_001/summary.json` | 训练结束回执 |
| `evaluation_001/shard-*/summary.json` | 700次生成的两个分片进度 |
| `evaluation_001/summary.json` | 生成结束后的合并结果 |
| `evaluation_001/grace_score/summary.json` | 性质与新颖性评分；同时检查status及pending |
| `launch-source-sha256.txt` | 启动时源文件摘要 |

训练中断：确认旧进程已退出和卡空闲后，用原 `train.py --mode train` 命令、相同out，再加 `--resume experiments/structure_conditioned_ar8b_20261017/run_001/checkpoint.pt`。生成中断：对应 `evaluate.py --num-shards 2 --shard-index 0/1` 命令加 `--resume`，种子和已完成回执保留。评分续跑使用原 `grace_score.py --mode score --run-dir ... --resume`。不要从头重启整个无resume流水线覆盖日志。

## 与旧结果比较

评价保持旧模型的7空间群×100次、相同种子、采样温度、长度上限、结构检查与GRACE口径。旧模型最终新颖达标为1/700，本轮为12/700；结构有效率同时下降。训练子集SG143样本为0，SG119仅35条，分组结果须标明覆盖差异。

训练代码复用旧实现并绑定新任务构造；通用评价器仅把两条旧任务范围描述提取为可覆盖的常量，旧默认行为不变，修改前源文件和SHA256已保存在 `reference_source_snapshots/`。生成采样、KV缓存、解析器和物理评分逻辑沿用已验证版本。
