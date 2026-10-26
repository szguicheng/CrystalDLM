# 自定义 loss 与成熟训练库：数值与源码审计

日期：2026-10-23。本轮不修改历史训练实现，不加载8B权重、不使用GPU、不安装依赖；新增审计脚本、结果与本报告。运行环境为 LiuGangGPU 现有项目环境：Python3.11 / PyTorch2.6.0+cu124 / Transformers4.57.6 / PEFT0.18.0；CUDA_VISIBLE_DEVICES 为空。

## 已执行的数值对照

脚本：[check_loss_parity.py](../loss_implementation_audit_20261023/check_loss_parity.py)。输出：[parity-results.json](../loss_implementation_audit_20261023/parity-results.json)。

采用同一随机初始化的两层小Qwen3和明确的G/Q样本。被测一侧调用实际项目loss函数；参考一侧独立构造监督label，调用HuggingFace Qwen3ForCausalLM的原生labels loss，再按该阶段规定的权重组合。辅助时间嵌入第0行设为非零，避免只验证其初始化为零的特殊情况。对所有参数梯度逐项比较。

| 目标 | 项目loss | HF按相同目标组合的loss | 最大loss差异 | 最大梯度差异 |
|---|---:|---:|---:|---:|
| 前置AR的分段目标 | 4.4413771629 | 4.4413771629 | 0 | 2.384e-7 |
| 后续AR的G/Q任务平均 | 2.9759664536 | 2.9759664536 | 0 | 0 |
| 前置AR，实际PEFT注入与控制token参数 | 4.4596652985 | 4.4596652985 | 0 | 1.788e-7 |
| 后续AR，实际PEFT注入与控制token参数 | 2.9877910614 | 2.9877910614 | 0 | 0 |

加入故意错误的同位置预测作为负对照：正确shift loss3.067152、错误同位置loss3.008201，检查能够区分该类错误。远程受测三个loss源码的SHA均与本机文件一致；前置共享model.py的SHA也与原训练setup.json一致。

这个对照证明：在该小模型/样本/FP32 CPU条件下，所测AR实现（包括项目实际PEFT注入路径）符合其定义的加权目标，并没有在这里暴露token错位或梯度计算不一致。它不证明所选任务权重最佳、不覆盖全部真实tokenizer/8B/BF16/DDP行为，也不是一个成熟trainer的完整匹配重训。

## 已重跑的既有测试

详见[existing-tests-results.json](../loss_implementation_audit_20261023/existing-tests-results.json)，四个进程共10项测试全部通过：

- 纯AR训练契约6项：因果mask、控制token输入/输出梯度、未来token不影响过去logit、两rank样本分配、状态指纹、保存恢复及优化器状态。
- FLARE可见性2项：噪声目标不可读取干净答案、互补视图每晶体token恰好覆盖一次。
- 微型Qwen数值1项：clean流匹配原生causal；packed噪声logits匹配逐块推理；改变被遮盖的答案不改变其当前块噪声hidden。
- 完整G/Q目标组合1项：两任务均纳入loss、权重与反向传播可用。

其中两rank测试检查索引划分，不是本轮重新执行真实多GPU通信。去噪测试验证本项目可见性契约，不是对FLARE原版端到端性能的复现。

## 损失定义的实际差异

用 `CE_desc`、`CE_crystal`、`CE_Q` 表示各段或任务内的平均交叉熵：

- 前置AR：`(CE_desc + CE_crystal + CE_Q) / 2`。
- 后续AR：`(CE_G_whole + CE_Q) / 2`，G在整个assistant目标token上平均。
- 联合FLARE式：`(CE_G_whole + D_crystal + CE_Q) / 2`；去噪项只覆盖晶体目标，并按G目标总token数归一化。
- 修正训练另有 `0.5 R_joint + 0.25 G_joint + 0.25 Q_AR`。

因此使用相同交叉熵不等于优化相同目标。按token总体平均、按段平均、按任务平均会赋予不同长度内容不同权重。上表同一小模型在两种AR公式下即有4.4414与2.9760的不同值，不存在参数更新造成的差别。

来源：[早期分段实现](../../experiments/ehull_interleaved_smoke_20261013/model.py#L132)、[后期G目标](../../experiments/flare_crystal_qwen3_8b_20261019/model.py#L50)、[后期G/Q组合](../../experiments/flare_crystal_full_20261019/model.py#L14)、[HF4.57.6因果loss](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py#L25-L63)。

多任务加权可以是研究设计；当前检查不能为本项目具体G/Q/R权重提供实证最优性依据。

## 与使用成熟库的本质区别

PLaID官方也自定义Dataset、晶体串和任务，再交给Transformers Trainer：[官方训练代码](https://github.com/andaero/PLaID/blob/main/llm_finetune.py)。本轮查看的main代码将序列input_ids也用作labels，collator用-100屏蔽padding，因此其prompt监督范围也不能直接等同于只训assistant的SFT设置。

CrystalDLM还自行接管了hidden/attention、预测位置、分段loss归一化、任务合并、训练循环及晶体采样。主要增加的是需独立确认的行为及实验变量；自定义本身不自动构成bug。

本轮没有完成LLaMA-Factory/TRL与现有实现的同数据、同初始化、同可训练参数及同目标的完整对照。若直接更换模板、参数集和loss权重后取得更好结果，不能把改善唯一归因于trainer。


## 独立源码审计与边界复现

完成独立只读审计后，主线程重新检查相关代码并执行[边界复现脚本](../loss_implementation_audit_20261023/check_sampling_boundaries.py)，记录见[sampling-boundaries.json](../loss_implementation_audit_20261023/sampling-boundaries.json)。不修改生产代码。

### 确认的旧AR采样缺陷：PAD可进入正文

`experiments/ehull_interleaved_smoke_20261013/model.py:218` 将PAD与END都排除在禁采集合之外，而 `unpack_block` 拒绝END之前出现PAD。在固定logit的最小例子中，实际采样函数生成PAD→END，随后自身抛出 `invalid crystal PAD boundary`。这是采样约束与格式契约矛盾，不是训练loss错误。

影响边界：Qwen3-8B旧32条评估摘要的 `crystal_pad_boundary_failures=0`；本轮没有证据说该缺陷损坏了这批结果。后续结构条件700次runner和FLARE完整评估已禁止PAD，不走这个旧缺陷路径。

### 确认的FLARE训练/推理边界差异：END后仍填完整块

`experiments/flare_crystal_full_20261019/evaluate.py:102` 每次填满4-token块，`:141` 才截断END后缀。训练 `model.py:56` 则在END终止目标，最终块可不足4个目标token。

受控调用实际采样循环的状态为 `[MASK,MASK,MASK,MASK] → [MASK,END,MASK,MASK] → [MASK,END,MASK,x] → [MASK,END,y,x]`，最后仍会预测END前的位置。小Qwen实验中，仅改变将被丢弃的END后token，该保留位置的logits最大改变0.0751873。因此不应声称最终部分块与训练的边界完全一致。

正式700份FLARE回执中682份以END结束；其中519份的截断后长度不是4的倍数，推理时存在额外块内后缀位置。回执不保存块内提交顺序，不能认定519条全受到了有害影响，也没有量化它对有效率的贡献。该项归类为已经复现的边界不一致与待评估风险，不据此宣称全部训练失效。

### 其余检查与结论

本次没有确认受测AR目标的token shift、错误label掩码、目标答案泄漏、优化器漏纳入参数或DDP平均缩放的具体bug。这里的“没有确认”限于本轮覆盖范围，并不等同于全实现无误。

已确认的目标变动（分段平均→整体G平均）、未论证的G/Q/R任务配比、Q上下文中已有事实和合成R错误分布，属于不同于CE计算错误的设计/证据问题。要判断生成任务是否学好，仍需独立生成评价；要证明具体任务权重或训练方法有收益，仍需匹配对照与消融。

对照文献与源码详见[loss-literature-audit-20261023.md](loss-literature-audit-20261023.md)。
