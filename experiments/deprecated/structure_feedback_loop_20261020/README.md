# 结构反馈循环试验

2026-10-20 用户授权。使用冻结的完整 FLARE 检查点，对比独立重采样和结构反馈重试。7 种空间群，每种 10 个配对种子组；每种策略每组最多 3 次，成功即停止。第一次候选共享，后续种子配对。GRACE 不参与反馈，本轮不训练、不做性质评分。

## 当前状态

反馈切换修复版本已完成完整复验：122/122次重试自然进入Diffusion，但新增严格有效晶体为0，最终仍为9/70成功（直接重试19/70）。见[修复验证结果](switch_fix/RESULTS.md)。新入口为 `switch_fix/launch.sh`，见[根因与修复](switch_fix/DIAGNOSIS.md)及[修复验证协议](switch_fix/PROTOCOL.md)。下文原始脚本保留用于复现旧对照。

正式试验已于 2026-10-20 20:27:18（Asia/Shanghai）完成，耗时55分43秒，两个分片及监督脚本均exit 0，GPU已释放。独立重采样19/70成功，反馈循环9/70成功；122次反馈重试中117次未进入晶体段，另5次晶体均未通过严格检查。详见[正式结果](RESULTS.md)。

代码与协议见 [PROTOCOL.md](PROTOCOL.md)。监督脚本在两个工作进程成功退出后自动生成 `run_001/summary.json` 与 `run_001/RESULTS.md`；异常退出会标记为 failed，不会将不完整结果写成完成。

## 预跑验证

- 9 项 CPU 契约检查通过。
- 两张 GPU 上，新旧单段去噪接口 logits 的最大差异和 argmax 差异均为 0。
- 7 个首轮候选的文字与晶体 token 与历史同种子输出完全一致。
- 反馈上下文保留原始目标、上一次实际输出和检查结果；已验证完成组断点恢复直接跳过、不重新生成。
- 7 组均完成，两个分片 exit 0。两种策略均成功 2/7；反馈组 10 次重试均未触发晶体 START，记录为失败。这是小样本观察，正式 70 组结果将用于量化。

预跑结果：[smoke_001/RESULTS.md](smoke_001/RESULTS.md)；工程验收：[smoke_001/acceptance.json](smoke_001/acceptance.json)。第一次启动因新目录误带 `--resume` 而在生成前退出，已修正启动脚本；日志保留，未产生候选或增加实验分母。

## 运行与恢复

从远程项目根目录执行 `bash experiments/structure_feedback_loop_20261020/launch.sh pilot`。长任务在上述独立 tmux 中运行。再次执行前先确认当前会话及 GPU 进程已退出；脚本检查所选 GPU 空闲并限制项目总占用不超过三张卡。

每个分片按已存在的 manifest 自动选择新建或恢复。每次候选完成后原子保存组状态；恢复会核对代码、检查点、请求及种子身份，保留已有失败尝试。运行中的进度可以用以下命令汇总：

```bash
python experiments/structure_feedback_loop_20261020/summarize.py \
  --out experiments/structure_feedback_loop_20261020/run_001 --allow-partial
```

完成前只汇总已结束的组，成功率分母保留全部计划组，状态为 incomplete。最终结构成功指严格检查通过且实际空间群符合原始要求；ehull 仍仅为输入目标。
