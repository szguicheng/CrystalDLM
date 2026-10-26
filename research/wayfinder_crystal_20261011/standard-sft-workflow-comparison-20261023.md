# LLaMA-Factory 标准 SFT 用于 CrystalDLM 纯 AR：核查笔记

核查日期：2026-10-23。依据 LLaMA-Factory 官方 `main`/`latest` 文档与源码做静态核对；未安装、运行或训练。

## 结论

**可以用来建立 A：标准 Qwen3 AR SFT 新基线。** 官方模型表列出 Qwen3 8B 和 `qwen3_nothink` 模板；官方 SFT 示例使用 `stage: sft`、LoRA 与 `qwen3_nothink`。这证明框架提供相应训练路径，不证明它会提升 CrystalDLM 的结果，也不等于精确复现现有训练。[官方模型支持表](https://github.com/hiyouga/LlamaFactory#supported-models) · [官方 SFT 配置示例](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)

## 三个问题

**SFT 是什么？** 监督微调用带有标准答案的输入/输出样本训练模型，使因果语言模型根据输入预测目标回答的后续 token。LLaMA-Factory 的 Alpaca/ShareGPT 数据都能表示这类样本；自定义数据要在 `dataset_info.json` 登记。[数据格式说明](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html)

若设 `train_on_prompt: false`，官方处理器把 prompt 的 label 设为 `IGNORE_INDEX`，目标回答保留为监督 token；`mask_history: true` 则只监督当前对话轮，历史 assistant 回答只作上下文。[监督标签处理源码](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/data/processor/supervised.py#L76-L85) · [参数说明](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html)

**和本地做法的差别。** 当前 Qwen3 纯 AR 包装器继承项目自定义 `ARBaseline`、添加四个控制 token，并在训练时调用自定义 loss；本地配置为 LoRA r=32/α=64、目标含注意力/MLP 投影和 `lm_head`，另有可训练的 `time.weight`。[model.py:L17](../../experiments/ehull_pure_ar_qwen3_8b_20261015/model.py#L17) · [model.py:L54](../../experiments/ehull_pure_ar_qwen3_8b_20261015/model.py#L54) · [config.json:L8](../../experiments/ehull_pure_ar_qwen3_8b_20261015/config.json#L8) · [config.json:L48](../../experiments/ehull_pure_ar_qwen3_8b_20261015/config.json#L48)

本地原始 `User:`/`Assistant:` 字符串不是 ShareGPT 消息结构，也没有使用 Qwen chat template；迁移到 `qwen3_nothink` 需转换成 ShareGPT/Alpaca 记录并按 Qwen 模板渲染。G 可表示为 user 指令和 assistant 晶体文本。Q 可表示为旧 G assistant 历史、当前 user 读回问题和 assistant 答案；启用 `mask_history` 后旧 G 只提供上下文，不计入该条 Q 的监督。[本地 G/Q 构造](../../experiments/ehull_interleaved_smoke_20261013/prepare.py#L12) · [官方 ShareGPT 格式](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html#sharegpt)

冻结 train/validation 成员时，把两份既有 split 分别登记为数据集，设 `dataset` 与 `eval_dataset`；不要用 `val_size` 重新切分。官方支持 `logging_steps`、`plot_loss`、`eval_strategy`/`eval_steps` 与 `eval_dataset`，SFT workflow 会记录并保存训练/验证指标。晶体严格解析、性质达成或 GRACE/ehull 评价仍需 CrystalDLM 自己的 evaluator。[SFT 参数示例](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html) · [SFT workflow](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/train/sft/workflow.py#L121-L149) · [本地冻结 split 校验](../../experiments/ehull_pure_ar_qwen3_8b_20261015/train.py#L176)

## 两种迁移目标

- **A｜标准 AR 新基线：可配置，需接受格式与参数集变化。** 可显式设置 LoRA rank、alpha、dropout 和 target modules。官方也提供 `new_special_tokens`、`resize_vocab`，以及 `additional_target` 将 LoRA 以外的模块设为可训练并保存。[LoRA 参数](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#lora) · [新增 token 参数](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#model-arguments)
- **控制 token 的关键差异：** 本地用 PEFT `trainable_token_indices` 只训练四个输入 embedding 行，并对 `lm_head` 加 LoRA；LLaMA-Factory 文档化的 `additional_target` 是模块级，若设为 `embed_tokens`/`lm_head`，训练和保存的是整个匹配模块，不等价于只训练四行，也不等价于当前输出头方案。精确匹配需定制 PEFT/保存逻辑。[本地 token 选择](../../experiments/ehull_pure_ar_qwen3_8b_20261015/model.py#L23) · [官方 `additional_target` 定义](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#lora)
- **B｜精确复现历史自定义 G/Q 权重与 `time.weight`：需要定制，不能承诺零代码。** 历史目标有版本差异：早期 AR 使用分段 CE，后期 AR 使用整体 G CE；不能把它们合并成一个 SFT loss。标准 assistant-token 标签掩码没有同等的 G/Q 独立归一化配置。项目 `time.weight` 也不是 Qwen3 基座模型自带模块；要保持其结构、初始化、优化器纳入方式及 checkpoint，需要定制模型/训练流程。**因此 A 是新基线，不是旧结果的框架迁移复刻。**[本地 G/Q loss 实现](../../experiments/ehull_interleaved_smoke_20261013/model.py#L157) · [纯 AR loss 调用](../../experiments/ehull_pure_ar_qwen3_8b_20261015/train.py#L479) · [本地 time 参数记录](../../experiments/ehull_pure_ar_qwen3_8b_20261015/config.json#L48)

## 未实测

尚未验证 `Qwen/Qwen3-8B` 与 `qwen3_nothink` 的实际 tokenizer/template 组合、四个新增 token 的保存与推理恢复、训练显存/吞吐、生成有效率或性质指标。官方 SFT 示例使用的是 Qwen3 Instruct 4B；Qwen3 8B 家族支持不等于本地 checkpoint/template 配对已验收。

## SFT 的含义与现有实现

SFT（Supervised Fine-Tuning，有监督微调）是在已有预训练模型上，用预先准备的输入和参考输出继续训练。对普通 AR SFT，模型根据输入及正确前文预测下一个参考 token，以交叉熵计算损失并反向传播。LoRA 指只训练一部分适配参数，可以用于 SFT；两者不是互斥选项。

官方概念与监督范围参考：[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer#looking-deeper-into-the-sft-method)。本文没有运行 TRL 或 LLaMA-Factory。

现有纯 AR 的监督内容由 `experiments/ehull_interleaved_smoke_20261013/prepare.py:12` 构造。Qwen 基座和 PEFT LoRA 使用已有库，见 `experiments/ehull_pure_ar_qwen3_8b_20261015/model.py:23`。训练器自行计算 loss、梯度累积、裁剪和 AdamW 更新，见同目录 `train.py:465`。因此它属于 SFT，但没有使用成熟 SFT trainer 的完整流程。

## AR＋Diffusion 与官方 FLARE 的差异

官方资料：[HybridDiffusion README](https://github.com/yuchen-zhu-zyc/HybridDiffusion)、[发布的 2B 训练配置](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/main/torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml)。以下是 2026-10-23 查阅到的发布流程，不能当作所有分支的能力边界。

| 方面 | 官方发布流程 | 当前 CrystalDLM |
|---|---|---|
| 训练主线 | TorchTitan，发布配方采用 Qwen3.5 混合注意力骨干 | Qwen3-8B，Transformers/PEFT 加自定义训练循环 |
| 数据与去噪范围 | 推理、数学、指令语言数据；配置使用 response label 模式 | G 的自然语言描述只用 AR；晶体段做 AR 与去噪，Q 保持 AR |
| 实现 | 官方 clean/noisy 流、目标、内核及推理运行时 | 自写的 shifted-anchor 晶体块实现；有迁移设计 |
| 推理 | 提供 AR-Trust 与 Diffusion-Trust 两套运行流程 | 每个晶体块反复前向，每次提交一个位置 |

本地事实来源：`experiments/flare_crystal_full_20261019/PROTOCOL.md:7`、`experiments/flare_crystal_qwen3_8b_20261019/model.py:50`、`experiments/flare_crystal_full_20261019/evaluate.py:102`。早前固定 commit 的静态核查见 `research/wayfinder_crystal_20261011/flare-code-contract-20261019.md`；该记录没有完成原生全流程复现，不能用它证明官方实现有确定的性能问题。

从这些差异可以判断：当前结果属于 FLARE 思路的晶体任务迁移，不能直接视为官方流程复现。普通 AR SFT 配置也不会自动实现这个联合目标。官方流程可以作为后续实现与复现的起点，但接入晶体数据、仅晶体去噪的规则、Q/R 任务及匹配 AR 对照仍需适配。若变更基座，应重新建立同基座的 AR 对照。

官方 README 说明训练环境在 H100 上验证过；本轮没有在 LiuGangGPU 的 H20 上安装、检查内核或执行原生训练，不能承诺直接运行。也没有以换框架来保证生成效果改善。
