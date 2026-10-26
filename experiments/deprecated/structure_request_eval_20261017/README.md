# 自有 Qwen3-8B 模型：结构请求生成评价

2026-10-17 用户授权执行。只评价自有模型，不运行 PLaID 对照，不启动新训练。

## 本轮范围

- 检查点：`experiments/ehull_pure_ar_qwen3_8b_20261015/run_001/checkpoint.pt`，训练完成909步，trainable state SHA256 `14c8bd031e634fe2587e1dc5685254481dc3b717741b3cb6d6c1f1c16102a9e4`。
- 它是旧 ehull-only 训练的纯 AR 模型。本轮是对新结构要求的迁移评价，不是已按结构条件重新训练的模型，也不是共享 AR＋diffusion。
- 空间群1、15、38、119、143、194、216；每请求100次，共700次。固定seed42生成逐次随机种子，temperature=0.8、top-p=0.95；文字与晶体都随机采样。
- 输入明确指定空间群和 `energy above hull no greater than 0.08 eV/atom`。晶体上限544 tokens、文字上限256 tokens；失败占尝试，不补采。
- 第一阶段仅A组；20个化学式＋空间群的B组本轮尚未运行。

## 已完成的预检

- 7项生成/续跑测试；5项评分测试在远程项目环境通过。
- 缓存与完整前向计算：微型FP32 Qwen3在6处边界、非零时间嵌入下最大差异5.96e-08；8B BF16五处检查最高概率token一致，最大KL约0.00216。BF16舍入不满足最初逐元素严格比较，因此保留诊断并以明示的分布容差门控；不是逐元素完全相同。
- 初次预检暴露的MAXLEN引用和JSON元组/列表续跑比较已修复；首次CIF输出目录创建另有回归测试。失败日志保留。
- `preflight_002` 完成7次（每群1次），严格有效0/7，未据此调整采样配方。正式700次为独立运行目录 `run_001`，其中随机种子预先固定。
- GRACE独立环境：`/home/liugang/src/crystalDLM/envs/grace`，Python3.11、TensorPotential0.6.2、TensorFlow2.20.0；完整版本见 `grace-environment.txt`。
- 固定GRACE权重revision `46ead61dc8a7f90c11b1e59b8489e8e9bf97b888`，模型文件SHA256 `0d7ebce7d3ddbd168d086fbd43c583a4c34fb55b7509f78be3dc89db67651e52`。硅单点能量、力、应力smoke通过。

## 物理评分口径与校准

GRACE-3L-OMAT-large-ft-AM（即GRACE-3L-OAM-L），ASE FIRE＋FrechetCellFilter，fmax=0.05 eV/Å，最多300步。使用MPRelaxSet的统一GGA/U元数据和MP2020兼容修正，与固定2023-02-07 MP参考凸包比较。原始及修正代理能量都保存；主评分采用修正代理分数，门槛0/0.08/0.1 eV/atom，未收敛者不计成功。

12个已知验证材料全部弛豫收敛；修正后代理ehull与同一凸包条目的MAE为0.004806 eV/atom，原始未修正分数MAE为0.176257 eV/atom。80meV分类一致11/12。此检查没有拟合能量修正参数；原2025快照ehull标签没有被用作参考能量。相图条目没有CIF，校准仅核对了材料ID和组成，未验证两版结构完全相同；这不是全域误差保证。

生成晶体须通过原始严格结构检查，再对弛豫后结构检查几何、空间群、收敛与能量。去重采用固定StructureMatcher参数。新颖性相对于自有模型7272条训练结构；仅一个自有生成模型参与，因此不加入PLaID训练集。引用与实现依据另见研究评价设计。

## 运行与结果位置

服务器：LiuGangGPU；全部文件在 `/home/liugang/src/crystalDLM` 内。

tmux socket：`/home/liugang/src/crystalDLM/tmp/crystaldlm-tmux.sock`。

| 会话/文件 | 用途 |
| --- | --- |
| `sg-eval-full-20261017` | GPU0、1各350次生成；`launch_full.sh` |
| `grace-calibration-20261017` | GPU2上的12结构校准，已完成 |
| `grace-scoring-20261017` | GPU2先评分已有候选，退出GPU后等待生成；700次生成结束后自动续跑完整评分 |
| `run_001/shard-*/attempts/` | 每次原始输出、种子、严格审计、可解码CIF；逐条原子写入 |
| `run_001/summary.json` | 生成结束后的合并摘要 |
| `run_001/grace_score/attempts/` | 每次物理评分、失败、弛豫结构、新颖性与去重记录 |
| `run_001/grace_score/summary.json` | 主评分摘要；同时查看status、completed/pending，未结束不当最终成绩 |
| `calibration_001/` | 12结构校准逐例与汇总 |
| `full-exit.json`、`scoring-pipeline-exit.txt` | 两个生成进程与评分流水线的退出状态 |
| `full-source-sha256.txt`、`scoring-source-sha256.txt` | 实际启动源文件摘要 |

生成中断时，在确认对应进程已退出、GPU空闲后，使用原命令加 `--resume`，保留 `--num-shards 2 --shard-index 0/1`、相同run目录与固定种子。评分同样支持 `--resume`。基础设施中断不是新生成机会，已有成功回执不重跑。不要直接重启无resume的启动脚本覆盖日志。

生成和评分耗时分开记录；批量运行尚未全部结束前不得给出最终SUN或ehull达标率。所有ehull数值是指定MLIP/固定参考凸包下的代理预测，不是新候选的DFT验证。
