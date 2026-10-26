# 使用新结构条件训练共享AR＋diffusion 8B模型

Status: resolved
Created: 2026-10-18
Parent: ../map.md

## 用户授权

用户明确要求继续AR＋diffusion训练，输入使用新的结构描述＋ehull。与刚完成的纯AR输入保持匹配：实际空间群要求＋覆盖标签的ehull上限，文字描述仍由AR输出。

## 配方

见 [协议](../../../experiments/structure_conditioned_hybrid8b_20261018/PROTOCOL.md)。同原始8B基座重新初始化LoRA、同7272/900记录和909updates，仅晶体目标/注意力/采样改成32步吸收掩码扩散；共享核心参数，禁止未来答案泄漏。8172条任务输入和目标已与纯AR逐条比较相等。训练后用同700次请求及GRACE口径对照纯AR的12/700新颖联合达标。

当前正在检查混合目标与准备独立代码，尚未声称预检或训练完成。

### 已启动 — 2026-10-18 13:54

远程12项CPU检查通过，输入与目标8172条逐条匹配纯AR。已在LiuGangGPU两张H20上启动独立tmux流水线，进行真实8B的3update预检；后续自动训练、700次生成和GRACE评分。新旧数据/初始化保持匹配，旧代码及检查点不修改。[运行记录](../../../experiments/structure_conditioned_hybrid8b_20261018/README.md)。

### 预检通过，正式训练中

真实8B两卡3update预检完成，梯度有限且新增token路径非零；重载后前缀logits与全MASK晶体探针logits的最大差异均为0。正式初始化trainable-state SHA与匹配纯AR完全一致。当前已记录14/909次更新，损失及梯度有限；尚无训练后生成成绩。

### 运行完成，结果失败 — 2026-10-18

909次训练和700次生成均完成，完整流水线16:03 exit 0，GPU释放。700次均触发晶体生成，但严格有效0：292次边界错误、408次边界通过后的格式/一致性错误。GRACE计算0次，全部候选记录为无效跳过；联合新颖产率0/700，对照纯AR12/700。逐例复核和完整结果见 [RESULTS](../../../experiments/structure_conditioned_hybrid8b_20261018/RESULTS.md)。关闭本票仅表示授权运行完成，不代表方法成功；后续诊断/新配方尚未执行。
