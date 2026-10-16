# 配位多面体共享字段与 DeepSeek V4 Flash 试标

完成日期：2026-10-12。

## 全量共享字段

在已核对的 45,465 个材料上新增角／边／面共享信息，未修改已有 PLaID 文本、解码结构、核心事实、原始 ehull 标签或数据划分。逐记录恢复旧占位字段后与父数据完整比较，全部一致。

| 增补结果 | 材料数 |
|---|---:|
| 完成可适用的共享关系判定 | 45,128 |
| 不适用：没有 CN ≥ 4 的三维配位多面体候选 | 207 |
| 几何不明确：候选未通过严格凸包条件 | 130 |
| 增补计算错误 | 0 |

原先隔离的 164 条原样保留。可标注队列仍为 train 36,362、val 4,501、test 4,602，总计 45,465。观察到角共享的材料 44,018 个、边共享 38,289 个、面共享 31,058 个；这些类别可以同时出现。周期晶胞中的已验证接触总数分别为 3,888,201、2,350,315、1,401,120。

最终队列位于 [prepared_001](prepared_001/manifest.json)，三个 gzip JSONL 成功队列含新增 local_structure_facts.polyhedra_sharing 字段。共享方法与版本见 [sharing-method.md](sharing-method.md) 和 [sharing-method.json](sharing-method.json)。

使用原有几何模式 CrystalNN 近邻，周期顶点身份为原子索引与晶胞平移。角共享需一个共同顶点；边共享需两个共同顶点构成双方真实边；面共享需共同顶点组成双方同一合并凸包面、中心位于两侧。边和角另检查严格分离以排除额外重叠。没有根据任意三个共同原子直接断言面共享，也没有把 BCC/CP 环境赋予未经验证的多面体名称。几何不明确的关系保留状态与原因，不进入语言中的肯定陈述。

## Flash max 与速度对照

模型使用已列出的 deepseek-v4-flash，Chat Completions，thinking enabled、reasoning_effort max、max_tokens 16384；同一批 30 个训练材料，单并发，每次完成后间隔一秒。

| 30 材料试验 | 平均响应秒 | 中位数秒 | 输入 tokens | 输出 tokens |
|---|---:|---:|---:|---:|
| Pro 历史基准 | 2.128 | 2.144 | 37,427 | 3,712 |
| Flash 同输入对照 | 2.131 | 2.139 | 37,427 | 3,921 |
| Flash 加入共享事实 | 2.296 | 2.286 | 49,340 | 4,349 |

同输入对照的请求内容已逐个核对，与保存的 Pro 请求仅模型名不同。两者平均耗时相差约 0.16%，本次没有测到明显提速。新增共享事实的 Flash 首次文字返回平均约 1.196 秒。

这测量的是当前网关端到端耗时。Pro 使用历史调用，缓存命中也不同，不是同一时段交替请求的模型内核基准。详细结果见 [model-speed-comparison.json](model-speed-comparison.json)。本轮 60 个 Flash 请求共使用 95,037 tokens；完整共享描述队列按样本估计约 8,137 万 tokens，未核实网关实际单价。

## 文字质量与样例

30 个新草稿均包含至少一个有事实支持的共享模式，共享类型没有发现写错。初稿有 1 处将未分类的 W 六配位环境写成 octahedral-like，已依据事实删除该赋名。另修正材料编号、过长句数和几处网络维度归因措辞，保留原始 API 回执及 [修改记录](caption-review-corrections.json)。最终 30/30 的事实、原值标签与段落格式核对通过，平均约 102 个英文词。

最终可读段落见 [pilot30-flash-final.jsonl](pilot30-flash-final.jsonl)；结构—性质—文字三元组见 [pilot30-sharing-final-triples.jsonl](pilot30-sharing-final-triples.jsonl)。结构文字与原始 ehull 仍是独立字段；带标签的段落用于核查，不预设训练段落顺序。语义核查见 [flash-quality-review.md](flash-quality-review.md)。

例如 TlInCl3 的输入事实和段落包含：In 八面体之间的角共享，以及 In 八面体和 Tl 六顶点配位多面体之间的角／边共享。这些是本几何配位方法下已验证的周期接触。

## 验证与资源

共享提取回归 12 项通过，覆盖角／边／面、非真实边／面、重叠、周期平移及原有字段不变；标注客户端检查 6 项通过。Burgundy 作业 624464 在 cpunode-058 使用 32 CPU，耗时 10:19、CPU 时间 5:05:45、峰值 RSS 约 3.99 GiB，COMPLETED/0:0；没有使用 GPU，资源已释放。监控记录见 cpu-monitor.jsonl 和 cpu-monitor-summary.json。

已完成全量结构字段增补与 30 个材料的 Flash 试标；全量 LLM 标注和生成模型训练尚未运行。
