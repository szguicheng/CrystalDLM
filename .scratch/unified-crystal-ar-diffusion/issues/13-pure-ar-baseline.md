# 先运行匹配纯 AR 基线

Labels: wayfinder:task
Type: task
Mode: AFK
Status: resolved
Assignee: Codex /root
Parent: ../map.md
Blocked by: none

## Question

固定的新 ehull—英文描述—PLaID 子集能否通过纯 AR 学习产生有效晶体并正确读取生成结构？在此基础上，怎样与后续共享 AR＋diffusion 运行保持初始化、数据、目标序列、样本曝光和评估集的匹配？

## Comments

### Sequencing correction — 2026-10-13

用户重申既定顺序：先性质/数据，再纯 AR 基线，然后共享联合训练。当前数据上纯 AR 基线尚未执行。旧生成/编辑任务的 AR 结果不匹配；作业626135是提前运行的hybrid工程预跑，不能替代当前基线。

已优先准备 [纯 AR 配置](../../../experiments/ehull_pure_ar_baseline_20261013/PROTOCOL.md)：沿用train7272/val900、原始1.5B初始化和任务序列，全部目标token因果next-token监督；拟一轮全训练子集，909次更新、有效batch16、固定32个验证ID。后续hybrid须复用相同初始化指纹和样本/评估身份。

当前仅配置/入口准备，未提交Slurm训练，也未产生基线学习结果。完整目标性质达成还依赖独立新晶体评价，不能依据目标值或生成文字判定。此票只在实际训练/评估回执完成后记录运行状态，绝不将准备工作记为基线已完成。

### Prepared, not run — 2026-10-13

纯AR入口已准备：无噪声、全部目标因果next-token；固定1epoch全7272材料、909更新，预先固定32验证ID、训练顺序及初始化/数据/代码指纹。代码与配置位于experiments/ehull_pure_ar_baseline_20261013/。本地小模型测试和语法检查通过，只验证实现机制；没有加载1.5B权重训练，没有生成学习结果，也没有提交Slurm作业。性质评价状态保持not_run。

### Authorized and submitted — 2026-10-13

用户明确回复“开始”，批准本轮纯AR基线执行。VPN脚本确认已连接，Burgundy无账号既有任务。5项本地小模型测试重新通过；远端代码、共享实现、训练/验证文件及清单哈希匹配，data_001和logs路径存在，Slurm test-only通过。纯AR作业627862已提交gpu_a100，1GPU/4CPU/48GB/4小时上限，一轮7272训练材料、909更新、固定32个验证样本。当前只提交纯AR，不开始新的hybrid。运行和结果待Slurm/日志回执。

### Queue transition — 2026-10-13

普通gpu_a100作业627862处于Priority排队，调度估计为次日。检查发现stingy的A100空闲；集群不允许原作业直接改分区，因此确认627862仍为PENDING后取消，按同一脚本以--partition=stingy重新提交作业627890。只保留一个纯AR训练作业，数据/初始化/训练参数及4小时上限不变。stingy分区PreemptMode=CANCEL，存在抢占风险；实际运行状态后续核对。

### Running evidence — 2026-10-13

Slurm确认627890 RUNNING于gpu-a100-03；原627862 CANCELLED、Elapsed0，无重复GPU训练。计算节点5项测试通过，setup.mode=pure_ar，原始1.5B初始化权重指纹31877513dce5b9784be7fbc9802f23306e91257317cb5abcb174796329115355。观察到66/909次更新，最新loss和梯度范数有限，每步约1.33秒。训练、32样本生成与最终性质评价尚未完成，不将RUNNING记为成功。

### Resolution: first learning trial completed — 2026-10-13

纯AR作业627890 COMPLETED/0:0，Elapsed26:10，7272材料一轮909更新，loss/梯度有限，峰值显存32.75GiB。32/32触发晶体段且END/PAD边界通过；原始严格有效结构7/32。25失败首先集中于公式/原子总数/位点多重性不一致（19公式头、19Sites头，计数重叠），另1多重性、1坐标约束错误。

目标0的7请求均产生Cs8Tl4Au4Br24/Fm-3m并通过；25个非零目标均产生Mg4Ti4O10/C2m且失败。不能将此说成无论条件皆同输出，也不能认为目标0对应实际稳定性。7个有效结构后续文字均返回，4基础字段均与实际结构匹配，但它们为同一家族，尚无换结构控制，不能证明真实结构读取依赖。新生成结构ehull未评分。

关闭本票表示首轮运行/审查完成，未宣告整体基线质量通过或性质控制成立。完整结果见 [纯AR基线结果](../../../experiments/ehull_pure_ar_baseline_20261013/RESULTS.md)。先保留此基线并分析计数自洽、解码集中及条件接口，再决定联合训练/后续对照；不扩大全量转换。
