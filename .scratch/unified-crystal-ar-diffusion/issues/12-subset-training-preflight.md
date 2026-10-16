# 验证子集数据与共享模型训练链路

Labels: wayfinder:task
Type: task
Mode: AFK
Status: resolved
Assignee: Codex /root
Parent: ../map.md
Blocked by: 01, 02, 11

## Question

已完成的 seed42 子集能否按共享核心模型的文字 AR、PLaID 晶体 diffusion、晶体后的 AR 继续输出，进行实际前后向、保存恢复与原始采样？

## Comments

### User-authorized technical preflight — 2026-10-13

用户要求检查 tickets 并在 Burgundy 按既定架构开展测试。VPN 已恢复，hpclogin02 可连接；A100 分区可用，账号无运行作业。技术预跑沿用已有单视图吸收态去噪机制，使用已缓存的 Qwen2.5-1.5B 原始基座，不复用旧八材料微调权重。

这张执行票不依赖完整性质评分协议冻结，也不关闭“确定共享模型与纯 AR 对照方案”或“确定首轮成功、失败与继续标准”的完整科学契约。其验收限于来源绑定、序列容量、注意力隔离、有限损失/梯度、保存恢复及实际生成回执。

[技术预跑协议](../../../experiments/ehull_interleaved_smoke_20261013/PROTOCOL.md)：train7272、val900；K544、MAXLEN2048；单A100、上限2小时、20次参数更新、3个原始验证生成尝试。实际tokenizer容量检查与显存计时在计算节点执行，不截断超长样本。

当前状态：完成环境/权重缓存核查，正在合并训练三元组和适配独立实验入口，尚未提交作业。

### Submitted — 2026-10-13

数据完成全部8172行来源/标签绑定核查；两条中文/附注异常已定向Flash/max重标并保留原始回执，376条描述移除材料ID，当前格式标记0。代码独立审查未发现核心阻断；远端代码和数据清单哈希匹配，Slurm test-only通过。作业626135已提交gpu_a100，1GPU/4CPU/48GB/2h，当前结果待计算节点回执。

### Resolution — 2026-10-13

作业626135 COMPLETED/0:0，单A100-SXM4-80GB，Elapsed2:38。实际1.5B tokenizer核查所有8172行：晶体最大531token，完整生成/读取序列最大804/814，不截断。4项小模型测试通过，20次更新（20材料各G/Q一对）文字/晶体loss及梯度有限，权重和配置保存恢复逐位一致，峰值显存28.77GiB。

实际生成3次均未在AR端输出CRYSTAL_START，晶体扩散采样和后续AR读取均未进入。本执行票回答为：数据接入和训练/恢复链路可运行；交错生成尚未通过。关闭的是这次技术测试，不是整体模型任务。原始输出与统计见 [技术预跑结果](../../../experiments/ehull_interleaved_smoke_20261013/RESULTS.md)。下一步需要足量短训观察触发、扩散边界及实际结构读取；完整对照与性质达成标准仍留在既有票。
