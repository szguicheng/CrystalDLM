# 确定共享模型与纯 AR 对照方案

Labels: wayfinder:grilling
Type: grilling
Mode: HITL
Status: open
Assignee: none
Parent: ../map.md
Blocked by: 02, 04

## Question

依据方法研究和样本原型，选定哪一种基础模型、共享参数范围、性质条件入口、干净/带噪可见性、训练目标及配套生成方式？怎样设置匹配的纯 AR 对照、训练预算和模型评估，使它们共同回答目标条件生成与交错是否成立，而不是只回答记忆或 masked reconstruction？

预算需要覆盖预检查、所有训练臂和模型评估，单张 A100 累计不超过 48 GPU 小时。不能将单次训练收敛或见过样本成功作为性质泛化结论；未获得可靠时间估计则本票不可冻结运行规模。

## Comments

### Technical implementation evidence — 2026-10-13

用户授权Burgundy小测试后，已在独立目录复用共享Hybrid单视图机制，以新1/5三元组构造ehull条件→AR描述→PLaID去噪及参考晶体后的AR读取两类共享训练例子。1.5B原始基座20次更新通过前后向与恢复检查，原始生成3次未触发晶体段。见 [技术预跑](12-subset-training-preflight.md)。这提供接口/运行证据，未冻结完整纯AR对照、科学训练曝光、loss消融或性质评价；本票保持open。

### User correction: pure AR first — 2026-10-13

用户重申原定五步顺序，指出必须先做匹配纯 AR 基线。主智能体核查：新8172条ehull—描述—PLaID子集只有mode=hybrid的20步工程作业626135，没有对应纯AR学习实验。旧8材料生成/编辑AR实验不匹配当前任务，不能代替。提前执行hybrid是执行顺序错误，保留回执，不将其算作第3步或正式第4步。

接下来优先准备同样数据/表示/控制符、Qwen2.5-1.5B原始初始化、固定train/val划分与评估ID的纯AR学习基线；文字与晶体全部采用因果next-token目标。后续联合训练须保持匹配初始化和样本曝光。此处记录执行顺序纠正，不将尚未运行的基线记为完成。新生成结构的性质达成仍需独立评价，不能从文字或目标标签判断。

### Pure AR first-trial evidence — 2026-10-13

同数据/1.5B基座的纯AR首轮已完成，详见 [先运行匹配纯 AR 基线](13-pure-ar-baseline.md)。目前有7/32严格有效结构，但25个非零目标请求均落到同一无效组成/SG家族；尚未测新晶体ehull及结构替换读取对照。本票保持open，不能以training完成或低loss直接判定性质控制、也不自动启动联合训练。

### Base-model evidence and pending choice — 2026-10-13

用户询问Llamole/FLARE是否使用自训练基座，并质疑当前Qwen2.5-1.5B过小、过旧。已核对原论文/官方卡：[基座证据](../../../research/wayfinder_crystal_20261011/base-model-evidence-20261013.md)。Llamole语言基座7B/8B、LoRA4轮，图模块另预训；FLARE受控实验Qwen3-1.7B，最终Qwen3.5-2/4/9B，由已有checkpoint持续训练而非从零预训练LLM。

当前1.5B是低成本流程/小模型基线，不足以证明其适合最终主模型。较新的4B/8B应作为纯AR基座候选对照；候选Qwen3-4B-Instruct-2507与现有dense-attention路径更接近，Qwen3.5混合线性/标准注意力的diffusion迁移需专门实现。模型规模、版本、仅一轮曝光、只有ehull输入与greedy文字解码尚未通过对照分离原因。新型号、精度/LoRA及运行规模仍待用户决定，未下载权重或启动新实验。
