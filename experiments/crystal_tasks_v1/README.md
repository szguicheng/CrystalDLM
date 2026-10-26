# Crystal tasks v1

三个独立模型：AR-understanding、AR-design、Hybrid-design。所有模型从原始 `Qwen/Qwen3-8B` 初始化；匹配 design 分支共用 `configs/shared.json`。闸门 1 已通过，三卡、global batch 12（每卡 1 条，累积 4 步）；训练前必须通过闸门 2。

## 配置与环境

- `configs/shared.json` 是批准的训练/评价协议；`python -m experiments.crystal_tasks_v1.configuration` 生成两个 AR 的原生 LLaMA-Factory JSON。Hybrid 使用同一共享协议，其训练实现属于 Phase 4–5。
- `ar_understanding.json`：全量 36,362/4,501，3 epoch，BF16，LoRA 32/64，lr 2e-4，seed 42，cutoff 1024，验证每 600 updates。
- `ar_design.json`：相同配方，cutoff 2048 仅为待全量长度审计的上限；Design 数据未发布，`ready=false`。不得截断样本来满足该上限。
- 每 epoch、最低 validation loss、最终 checkpoint 的保留规则写在共享协议中；Phase 3 训练入口需接入 epoch 保存回调，单独运行原生 JSON 的 `save_steps` 不保证恰好在 epoch 边界保存。
- 固定 LLaMA-Factory v0.9.5 / `7af909522a951e3ad9f022ea6f88b6755257eaa5`，项目内补丁见 `setup_20261026/llamafactory-control-tokens.patch`。补丁传递 PEFT `trainable_token_indices`，避免训练完整 embedding/head。
- 远程所有 Python 命令用 `bash experiments/crystal_tasks_v1/python.sh …`。该入口在 mamba 激活后重新设置项目内缓存与临时目录；服务器激活配置会覆盖仅在外层 export 的设置。
- `setup_20261026/` 保存安装前后包清单、依赖解析计划、哈希校验与安装日志。GRACE 仍用原 `envs/grace`。

## CPU 验收

从远程项目根目录执行，输出路径必须是新路径：

```bash
CUDA_VISIBLE_DEVICES='' bash experiments/crystal_tasks_v1/python.sh -m unittest discover -s experiments/crystal_tasks_v1/tests -v
CUDA_VISIBLE_DEVICES='' bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.preflight --dataset-dir datasets/crystal_tasks_v1 --output experiments/crystal_tasks_v1/new-preflight.json
```

`preflight.py` 运行真实 LLaMA-Factory 数据处理/collator，核对全部 Understanding labels、长度、控制 token ID 和一个微型 Qwen 的可训练参数/反向传播。它不加载 8B 权重、不拟合样本。`reference_acceptance.py` 检查完整训练/test 参考库解码，再用少量真实结构和明确标注的合成性质回执验证指标；不是 GRACE 分数。

## 评价入口

具体字段、分母和未完成状态见 [METRICS.md](evaluation/METRICS.md)。所有输出目录禁止覆盖，评价记录含源代码与数据身份。

```bash
# Understanding：predictions 每行含 material_id、response，可附 seed。
CUDA_VISIBLE_DEVICES='' bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.evaluation.cli understanding --dataset-dir datasets/crystal_tasks_v1 --split test --predictions PREDICTIONS.jsonl --model Qwen/Qwen3-8B --checkpoint CHECKPOINT --output NEW_OUTPUT

# Design 数据发布后，先生成所有系统共用的原始尝试计划。
CUDA_VISIBLE_DEVICES='' bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.evaluation.plan --dataset-dir datasets/crystal_tasks_v1 --output NEW_PLAN.jsonl

# Design：每条计划都必须保留一条输出（失败输出可为空字符串）。
CUDA_VISIBLE_DEVICES='' bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.evaluation.cli design --dataset-dir datasets/crystal_tasks_v1 --attempt-plan PLAN.jsonl --attempts ATTEMPTS.jsonl --property-receipts GRACE_RECEIPTS.jsonl --model MODEL --checkpoint CHECKPOINT --output NEW_OUTPUT
```

GRACE 实测及检查点 NLL 均是后续评价阶段命令，执行前检查 GPU 占用并选一张空闲卡（以下 `GPU` 是待替换编号）：

```bash
CUDA_VISIBLE_DEVICES=GPU CRYSTAL_ENV_PREFIX=/home/liugang/src/crystalDLM/envs/grace bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.evaluation.grace --attempts ATTEMPTS.jsonl --dataset-dir datasets/crystal_tasks_v1 --model MODEL --checkpoint CHECKPOINT --output NEW_GRACE_OUTPUT
CUDA_VISIBLE_DEVICES=GPU bash experiments/crystal_tasks_v1/python.sh -m experiments.crystal_tasks_v1.evaluation.nll --task ar_understanding --split test --dataset-dir datasets/crystal_tasks_v1 --checkpoint CHECKPOINT --output NEW_NLL_OUTPUT
```

NLL 入口支持本项目 Qwen/PEFT checkpoint；Hybrid 参数导出为同样格式后可评价其 AR 分支，去噪前向由 Phase 5 的原生移植调用共享 CE 工具。公开 PLaID++ 使用原生采样器，输出送入同一 design 评价器，不能用其论文数字代替运行。
