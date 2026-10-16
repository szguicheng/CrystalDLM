# 固定 PLaID 的共享 AR/扩散方法证据

研究日期：2026-10-11。范围遵守地图确认：首轮一个性质、简短可核对文字、固定 PLaID、单张 A100 累计不超过 48 GPU 小时；本票只比较方法和本地旧原型，不冻结架构、不改源码、不训练。

## 论文各自实际证明了什么

| 来源 | 原论文方法 | 对本任务的可迁移部分与边界 |
|---|---|---|
| [Efficient-DLM v2，§§2.1–2.3、§3](https://arxiv.org/html/2512.14067v2) | 从预训练 AR 语言模型连续训练为块扩散；块间因果、当前块双向，噪声块以之前已解码的干净上下文为条件。论文在受测语言设置下选用不移位的同位置预测；训练还引入后部位置更容易被遮盖的概率分布，推理按置信度逐步提交 token。 | 支持“干净前缀 + 块内双向去噪”和同位置目标的候选。它处理的是语言 token 的 AR→扩散转换，不是文字 AR 与 PLaID 晶体扩散的多模态交错，也没有证明文本的后置遮盖偏置适用于晶体字段。论文报告 Qwen2.5-1.5B 的 50B-token 训练，不能作为单 A100 小型任务的时长依据。 |
| [FLARE v2，§§3.2–3.3、§4–5](https://arxiv.org/html/2606.01774v2) | 同一 checkpoint 保留 clean causal stream，并增加条件于前置 clean context 的 noisy block stream；每个 noisy block 内双向，不能读后续 noisy block。其扩散损失使用互补遮盖视图 M 与 Mᶜ，使每个 block token 在某个 noisy view 中被监督；总目标另含 clean AR loss。一个 checkpoint 因而能走 AR 和 block-diffusion 两种解码路径。 | 这是共享 checkpoint 的直接先例，但它在同一语言序列上做 AR 与扩散，FLARE 的 hybrid 指 Qwen3.5 等混合注意力骨干，并非文字—晶体两种模态。作者代码的 2B 配置为长度 4096、global batch 256、9000 步、8 GPU 启动配置：[官方仓库及配置](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/main/torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml)。这不等于本地 0.5B/A100 配方或 48 小时预算。 |
| [Llamole v1，§§2.2、3.1–3.3、附录 B.2](https://arxiv.org/html/2410.04223v1) | LLM 通过 trigger/query token 把前文条件汇聚为向量，再调用单独预训练的 Graph DiT 或 GNN；生成图再经 graph encoder 返回 LLM。Graph DiT 收到显式数值/类别性质条件与文本向量。标签来自已有标注及高置信伪标签；生成性质以 oracle 评估。作者实现见 [Llamole 仓库](https://github.com/liugangcode/Llamole)。 | 证明了文本—分子/聚合物结构交错、多条件分子生成和显式性质接口的可行性；结构由独立图网络生成，不是一个共享文本 LM 在 PLaID 字符串上做扩散，也没有直接证据证明可生成晶体。可借鉴数据契约、触发边界、性质标注和外部评价，不可把它说成已验证的共享 PLaID 架构。 |

因此，三篇是互补方法来源，不能合并表述成“已有论文直接实现共享模型 AR 生成文字、diffusion 生成 PLaID 晶体”。

## 本地 PLaID 原型：已有机制与差距

- [model.py](../../experiments/interleaved_ar_diffusion_20261002/model.py) 11–28 行用全局下三角因果 mask，并把同一个晶体 region 改成块内双向；当前噪声晶体可看到前面的干净上下文和本块其他位置。109–155 行对当前晶体块独立加噪，噪声步 k 均匀取 1…STEPS，按 k/STEPS 把目标位置替换成 MASK；只在 MASK 位置计算晶体 loss，并用 STEPS/k 加权。PAD 权重另设为 0.1。该实现不是 FLARE 的 clean/noisy 两流，也没有互补 mask 的第二视图。
- 同文件 31–38 行的默认 hybrid prediction_offset=0 对应同位置预测；offset=1 是可选移位。41–45、212–250 行用每步 1/k 概率释放 MASK，持续 STEPS 步并检查 END/PAD 边界。若保留均匀吸收态遮盖，这与当前离散遮盖率的反向释放日程相配；置信度提交或位置偏置可以另作后续选项，不能假定文本论文的左到右偏置对 PLaID 同样成立。
- [prepare.py](../../experiments/interleaved_ar_diffusion_20261002/prepare.py) 27–39 行的旧文字模板只给化学式和空间群；62–76 行把生成 G、读取 Q1、±5% 编辑 E、再读取 Q2 拆成独立前缀—目标例子；133–135 行只记录 formula、sg 和 PLaID 结构，不含所选性质的标签。它既不是性质条件数据，也没有单个“AR 文字 → 扩散晶体 → AR 文字”完整目标序列。
- 旧试点只能说明原机制和编辑任务的历史表现。[RESULTS.md](../../experiments/interleaved_ar_diffusion_20261002/RESULTS.md) 3–13、19–34 行记录首轮 8 个训练材料、0.5B、每个模型 200 更新，A100 作业 11:02，混合模型条件生成 0/8；[RESULTS_ADAPT_V2.md](../../experiments/interleaved_ar_diffusion_20261002/RESULTS_ADAPT_V2.md) 9–14、22–34 行记录后续增加去噪预适配后旧任务生成 7–8/8，但闭环 3–4/8，仍为已见 8 个材料；[RESULTS_AR_WARMSTART_SHIFT_V9.md](../../experiments/interleaved_ar_diffusion_20261002/RESULTS_AR_WARMSTART_SHIFT_V9.md) 3、9–13 行的 128 项旧验证里 offset=1 未改善生成或闭环。均不能替代新的性质条件任务结果，也不能单独证明预测偏移对所有设置无影响。

## 后续人工决策可比较的两种配方

以下两项是设计候选，不是本票冻结的结论。两者都将属性写在干净前缀里，先用普通文本明确属性名、目标值或范围、单位及方向，例如“target [property] in [range] [unit]”；不引入新 tokenizer 或独立性质头。单条数据至少要把一条经过校验的 PLaID 结构、该结构可追溯的性质标签（注明计算或预测来源）、条件请求和一条确定性生成的短事实描述绑定起来，并按底层材料划分数据。一个便于检验文字—晶体上下文衔接的候选样本是（不是老师已确认的必需流程，最终段落顺序与是否保留尾随文字由样本决策票确定）：

属性条件 → AR 短文字 → <CRYSTAL_START> PLaID结构 <CRYSTAL_END> → 基于该结构的AR短事实描述

扩散位置只能看到属性条件、之前的 clean text 和当前带噪结构块；尾随文字是未来目标，不能作为去噪上下文。推理时尾随事实应读取实际生成且解析通过的结构。属性匹配要用选定性质的外部计算或可信标签评价，模型自己写出的性质句子不能充当属性验证。

1. **单视图 PLaID 扩散（改动最少）**：短文本段用 AR next-token loss；PLaID 当前块单独吸收态遮盖，只在被遮盖的结构位置做同位置去噪 loss。前缀保持 clean、块内双向；使用均匀 mask 比例和相配的 1/k 反向释放作为起点。保留 END/PAD 的边界核验与长度不可见约束。该方案最贴近现有原型，额外引入的变量少；置信度释放和 Efficient-DLM 的位置偏置都留作独立后续消融。
2. **FLARE 启发的 clean 锚定与互补视图**：同一共享骨干另对完整 clean 序列计算 AR loss，保留文字能力并锚定因果路径；对 PLaID 块使用一组互补噪声视图 M/Mᶜ，使所有晶体目标位置都在某个视图得到去噪监督。当前晶体 noisy query 仍只看 clean 前缀和本块；不让去噪查询看到尾随文字。它借用 FLARE 的目标平衡与互补覆盖，但把 diffusion 限定到晶体，是迁移建议，不是 FLARE 原配方；多视图/双流前向的实测额外成本未知。

FLARE 的 noisy logit shift 与旧原型默认同位置预测并非同一配方。选择候选方案时必须一并确定预测位置、注意力和采样接口；加入互补遮盖不等于完整复现 FLARE。

两项的匹配纯 AR 对照应使用同一初始 checkpoint、PLaID 表示和控制符、同一性质样本及材料划分、同一文本—结构序列、相同样本曝光/训练 token 预算和评价器；AR 对照对所有目标 token（文字和 PLaID）使用因果 next-token loss。固定样本曝光并另报实测 GPU 时间，避免把不同前向成本藏进“相同更新数”。生成结构使用同一外部性质评价和结构合法性检查。

## 48 GPU 小时边界与待决定事项

现有本地实测：单张 A100-SXM4-80GB 的 job 600420 用时 42:40，完成的是旧 8 材料任务中两个模式各 600 次预适配、各两个 200 更新分支及诊断；收据见 [RESULTS_ADAPT_V2.md](../../experiments/interleaved_ar_diffusion_20261002/RESULTS_ADAPT_V2.md) 74–81 行及 adaptation_001/verification_receipt.json。该工作负载和新性质数据的样本长度、目标序列、更新量、loss views 与评估流程不同，不能据此外推新方案耗时，也不能承诺 48 小时足够。新方案的 48 小时须覆盖数据预检、纯 AR 对照、选定混合方案及评估；只有取得所选数据形状和配置的 A100 预跑实测后，才能决定预算分配与能否推进。

仍待人工决定：性质名称、单位、目标值/范围、标签来源及生成结构的性质评价器；优先比较单视图方案还是互补视图方案；样本长度、训练曝光与两臂预算分配。无方法研究阻塞；这些决策不能由论文或旧编辑实验代替。

