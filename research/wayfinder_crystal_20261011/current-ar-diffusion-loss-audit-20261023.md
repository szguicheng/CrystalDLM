# 最新匹配 AR / AR＋Diffusion：loss 与实现审计

日期：2026-10-23。用户已将问题限定为 `experiments/flare_crystal_full_20261019/{ar,flare}`；本报告不纳入前置AR的loss变更、旧采样器PAD缺陷或R修正任务。为验证当前Q与参数注入而读取共享实现，不等于评价旧实验。

## 当前目标

令G为描述＋晶体生成，Q为事实读取。当前纯AR对G完整输出计算平均next-token CE，再与Q任务平均：

`L_AR = (G_AR + Q_AR) / 2`

当前联合模型增加只覆盖晶体目标的去噪项；两个互补视图各监督不同的masked targets，去噪总和按G目标总token数归一化：

`L_joint = (G_AR + G_denoise + Q_AR) / 2`

来源：`experiments/flare_crystal_full_20261019/model.py:14` 与 `experiments/flare_crystal_qwen3_8b_20261019/model.py:50`。

普通AR SFT常用选中位置上的next-token CE。[HF4.57.6实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py#L25-L63)。PLaID官方也自写晶体Dataset和collator，再使用Trainer；其当前main将整串input_ids作为labels，只屏蔽padding，不是本项目G/Q任务均值。[官方代码](https://github.com/andaero/PLaID/blob/main/llm_finetune.py)。FLARE对AR和去噪采用token等系数联合监督，为联合目标提供先例；本项目仅晶体去噪、另加Q均值仍是任务迁移设计。[FLARE §3.2](https://arxiv.org/html/2606.01774v2#S3.SS2)。

因此，CE原语相同，不代表监督位置、任务归一化和最终目标相同。G/Q各占一半是本项目设置，其相对G-only或其他任务比例的收益没有在这里被验证。默认按全batch有效token平均与先按任务求均值不同。

## 本轮已执行的当前代码检查

使用服务器原项目环境，仅CPU：PyTorch2.6.0+cu124、Transformers4.57.6、PEFT0.18.0。没有加载8B模型权重、使用GPU或重训。

数值参考不是重新手写CE：独立构造输入labels，调用原生HF Qwen3ForCausalLM loss，再按当前G/Q权重组合，与实际项目目标函数比较loss和全部参数梯度。使用两层小Qwen、非零时间嵌入，并另走实际PEFT注入（LoRA与新增输入token参数）路径：

| 当前纯AR检查 | loss绝对差 | 梯度最大绝对差 |
|---|---:|---:|
| 普通小Qwen共享模型 | 0 | 0 |
| 项目PEFT注入的小Qwen | 0 | 0 |

此外，当前联合实现的4项现有局部测试全部通过：噪声目标不可读取clean答案；互补视图每目标覆盖一次；packed clean匹配原生causal、噪声训练/推理logits匹配及修改masked答案不改变相应hidden；G/Q权重组合与反向传播可用。

证据：[当前结果摘录](../loss_implementation_audit_20261023/latest-pair-results.json)、[数值脚本](../loss_implementation_audit_20261023/check_loss_parity.py)、[原始测试日志](../loss_implementation_audit_20261023/existing-tests-results.json)。该数值脚本保留了本轮最初审计的其他历史目标用例，但本报告只使用continued_ar_task_weighted与peft_continued_ar_task_weighted两项。所测源码哈希已与本机核对一致。

这些检查未发现当前AR目标的CE、shift和反向传播数值错误；不等于对完整8B/BF16/多GPU、真实tokenizer和所有采样路径的全面证明，也不等于官方FLARE端到端复现。

## 当前联合采样的已复现边界不一致

`evaluate.py:102` 在END出现后仍继续填满当前4-token块，`:141` 才截断。训练的参考序列则在END处终止，最后块可能不足4个目标token。

受控调用实际采样循环可出现：`[MASK,MASK,MASK,MASK] → [MASK,END,MASK,MASK] → [MASK,END,MASK,x] → [MASK,END,y,x]`，再预测END前位置。用实际微型Qwen denoise函数，只改变最终被丢弃的后缀，保留位置logits的最大变化为0.0751873。此处确有训练/推理边界差异。

700份正式联合生成回执中682份以END结束，其中519份截断后长度不是4的倍数。回执未记录块内提交顺序，不能认定519份均受有害影响，也没有量化其对严格有效率的贡献。该问题属于当前自写采样器，而不是CE公式错误；本轮未修改它。

证据：[边界复现脚本](../loss_implementation_audit_20261023/check_sampling_boundaries.py)、[当前边界观察与回执统计](../loss_implementation_audit_20261023/latest-pair-results.json)。

## 与成熟库的差异及结果能支持什么

- 如果输入token、监督labels、attention、参数集合、归一化及更新方式全部相同，自写与库执行可以数值等价，不能仅凭自写判定无效。
- 本项目超出了“自定义Dataset后交给Trainer”：自行处理预测位置、attention、G/Q权重和晶体采样。当前AR的数值核心已获得局部对照；联合采样已发现具体边界问题，不能笼统宣布整套实现无误。
- G/Q等权及仅晶体去噪的选择可能有研究意义，但需要任务消融和固定评价；loss下降本身不证明生成晶体有效或性质条件成功。
- 当前试验可以报告该实现下的实际生成表现；尚不足以把AR与联合模型的差异唯一归因于loss或整个AR＋Diffusion范式。
- 本轮未运行成熟SFT trainer的同数据、同初始化、同参数集合、同目标完整训练对照；没有证据承诺换库就会改善结果。

