# FLARE式共享Qwen晶体生成迁移

Status: resolved
Created: 2026-10-19
Parent: ../map.md

用户明确要求采用FLARE训练配方，基于Qwen预训练语言能力。以已有结构条件Qwen3-8B纯AR检查点为起点，保持干净序列AR监督并增加晶体互补遮盖块扩散，先8train/8val、64updates小试验，对照同预算纯AR继续训练。使用PLaID表示、既有数据与mamba，最多三卡并行。

完整[协议](../../../experiments/flare_crystal_qwen3_8b_20261019/PROTOCOL.md)。这不是Qwen3.5/GDN原版FLARE全量复现。独立静态审查发现官方当前shift/mask路径可能选择仍可见目标，本实现按论文masked-target契约明确先遮目标再取前位logit，并测试了多层注意力无泄漏与训练/推理数值一致性。

远程五项CPU测试通过；接下来先两步真实训练、恢复到64步，再用固定参考文字评估源模型AR/块扩散、AR控制组AR、FLARE组AR/块扩散。不自动跑700或GRACE，不用train小样本结果声称泛化成功。

## 完成

小试验完成，80输出齐全，exit均0、GPU释放。源模型diffusion为train0/8、val0/8；FLARE式适配后train8/8、val4/8有效且满足SG。源模型、AR控制与联合模型的保留样本AR有效均5/8。完整[结果](../../../experiments/flare_crystal_qwen3_8b_20261019/RESULTS.md)。本票关闭仅表示小样本机制验证完成；完整训练、端到端交错、700请求和性质评分尚未执行。
