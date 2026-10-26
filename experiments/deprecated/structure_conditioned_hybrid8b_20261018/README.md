# 新结构条件共享AR＋diffusion 8B实验

**本轮已完成但生成失败**：2026-10-18 16:03流水线exit 0，GPU释放；700次严格有效0，292次边界错误、408次结构格式/一致性错误，没有候选进入GRACE物理计算。详见 [RESULTS.md](RESULTS.md)。

用户于2026-10-18授权；约13:54在LiuGangGPU启动独立流水线。协议见 [PROTOCOL.md](PROTOCOL.md)。

同一Qwen3-8B核心处理AR文字和掩码扩散晶体，不加载第二个晶体生成模型。输入与刚完成的纯AR逐条一致：空间群要求＋覆盖真实标签的ehull上限；7272/900数据、原始基座初始化、LoRA设置及909更新保持匹配。

晶体段K544、32步、同位置masked-token预测；晶体内部双向，前文文字因果，后续问题/答案不可见。单个位置释放后保持不变，END/PAD由模型预测，原始错误保留。

## 已验证

- 真实8B双卡3update预检通过，重载后前缀和全MASK晶体探针logits最大差异均为0；已进入正式训练。正式初始化trainable-state SHA与匹配纯AR相同（649672e3…）。
- 8172条任务输入与目标逐条等于匹配纯AR，流摘要保存在 `matched-task-audit.json`。
- 远程项目环境12项CPU检查通过：同位置目标、当前块双向注意力、未来答案隔离、完整晶体后的因果读取、MASK/时间/新增token梯度、与纯AR相同的初始适配器状态、检查点恢复，以及32步释放和无修复的边界判定。
- 首轮微型检查点测试错误地重新随机化了冻结基座，已修正为相同基座并要求恢复前后训练状态不同/恢复后logits一致；真实8B预检还会核对前缀及全MASK晶体的重载logits。

## 运行位置

- 项目根目录：`/home/liugang/src/crystalDLM`
- tmux socket：`/home/liugang/src/crystalDLM/tmp/crystaldlm-tmux.sock`
- 会话：`sg-hybrid8b-20261018`
- 总入口：`launch_pipeline.sh`
- 训练及生成环境：`envs/crystaldlm`；GRACE环境：`envs/grace`。
- 训练/生成GPU0和1，评分在生成结束后用GPU0；启动前检查占用。

流水线：CPU检查→双卡3update预检→从基座重新初始化并训练909updates→7次工程生成预检→7空间群各100次生成→既有GRACE/去重/新颖性评分。生成预检的低质量输出不是修改配方或删题的理由。

| 文件 | 用途 |
| --- | --- |
| `stage.txt`、`pipeline-exit.txt` | 当前阶段与流水线退出码 |
| `preflight_001/summary.json` | 真实8B的梯度与前缀/全MASK恢复检查 |
| `run_001/progress.json`、`run_001/summary.json` | 训练进度与完成回执 |
| `evaluation_001/shard-*/summary.json` | 两个生成分片进度 |
| `evaluation_001/summary.json` | 700次生成完成摘要 |
| `evaluation_001/grace_score/summary.json` | 最终代理性质与新颖性成绩 |
| `launch-source-sha256.txt` | 启动时源文件摘要 |

独立目录保存全部输出，旧纯AR代码与检查点没有修改。`trainer_engine.py` 从已验证8B训练器机械复制，改动两处训练mode、完成摘要mode，并添加全MASK重载检查；其余优化器/分布式/保存恢复流程保留。完整代码链纳入检查点身份摘要。

## 中断恢复与结果边界

检查任务退出和GPU空闲后，训练用相同out加 `--resume run_001/checkpoint.pt`；生成用相同分片参数和目录加 `--resume`；GRACE用原命令加 `--resume`。已有回执与种子不变，不从头覆盖流水线日志。

主对照为新输入纯AR的12/700联合新颖达标、95/700严格有效。hybrid的成绩尚未产生，不能用预检梯度/保存恢复通过代表晶体生成成功。ehull统一为GRACE/固定MP2023凸包代理预测，非候选DFT验证。
