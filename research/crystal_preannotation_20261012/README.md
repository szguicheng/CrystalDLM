# 全量英文标注前准备结果

完成日期：2026-10-12。范围是全量 CPU 结构预处理与 30 个训练材料试标。

## 最终数据队列

源数据 45,629 个材料全部完成预处理和来源核对，45,465 个通过、164 个隔离，通过率约 99.64%。最终使用 **prepared_002**，早期 full_001/prepared_001 仅作诊断记录。

| 划分 | 可标注 | 隔离 |
|---|---:|---:|
| train | 36,362 | 129 |
| val | 4,501 | 16 |
| test | 4,602 | 19 |
| 总计 | 45,465 | 164 |

最终队列及各文件 SHA-256 在 [prepared_002/manifest.json](prepared_002/manifest.json)，失败记录在 [prepared_002/errors.jsonl.gz](prepared_002/errors.jsonl.gz)。三个成功队列为 train-facts.jsonl.gz、val-facts.jsonl.gz 和 test-facts.jsonl.gz。

每条记录包含 PLaID 文本、解码结构、核心事实、局部几何描述符、独立的原始 ehull 标签及来源。全部 45,629 条的材料 ID、源结构哈希、原始标签和单位、划分、源文件哈希均与源数据一致；没有重算或替换标签。

## 方法与失败原因

[feature-method.json](feature-method.json) 固定方法 v3：原版 PLaID 编解码；对称容差 0.01 Å、角度容差 5°；CrystalNN 几何模式近邻与同一近邻集合上的多面体 motif 阈值；Larsen 周期图连接维度。软件版本和原版 utils.py 哈希均受检查。原始结构与解码结构的字段分开保存；理论密度来自解码晶胞质量/体积。

已提取组成、空间群/晶系、晶胞、原子与独立轨道计数、理论密度，以及几何配位数、近邻元素/距离、受支持的局部形状和周期图连接维度。角/边/面共享没有计算，不进入描述输入。解码后的常规晶胞最多 80 个原子；源数据 20 原子上限不等同于解码晶胞上限。

第一次全量核查发现空间群别名解析错误，误拒绝 1,425 条。v3 使用 PLaID 同一 PyXtal 解析器读取编号，再统一规范符号；修复后恢复 1,418 条，另外 7 条未通过真实一致性检查。最终隔离原因：30 条编码原子计数不一致、38 条空间群不一致、54 条空间群和轨道同时不一致、1 条仅轨道不一致、40 条无法匹配回源结构、1 条近邻计算失败。这些是当前表示/分析检查结果，不能直接称为无效材料。

## 英文试标

30 个训练材料覆盖全部七个晶系、28 个空间群，五个 ehull 抽样区间各六个。最终 [pilot30-rendered-final.jsonl](pilot30-rendered-final.jsonl) 经逐条事实、来源和格式核对，30/30 通过，平均约 91 个英文词。

原始 API 回执在 pilot30-annotations-v3.jsonl，模型 deepseek-v4-pro，经 SciencePal Chat Completions 调用，thinking enabled、reasoning_effort max。先由 LLM 生成简短结构文字，再由程序加入原值 ehull 和 MP 来源句；结构文字与性质仍独立保存。

初始试标暴露数量误读、cp 被扩写为特定堆积类型、标签被舍入及缺失字段多余说明。最终输入汇总等价环境，区分每个中心的近邻数量和轨道/原子数量，展开局部 motif 名称，限制额外推断；原始值句由程序呈现，保留 API 原文与整理过程。用于检查的带标签段落和结构文字分别保存在 [pilot30-sample-triples.jsonl](pilot30-sample-triples.jsonl)。本文件不指定后续训练段落顺序。

试标与最终全量 v3 的同一 30 个材料再次核对，PLaID 文本及语言输入的结构语义字段全部一致，见 [pilot30-full-v3-binding.json](pilot30-full-v3-binding.json)。逐材料核查记录见 [pilot30-quality-review.md](pilot30-quality-review.md)。

## 资源与用量

Burgundy 作业 624103：32 CPU、17:44、总 CPU 7:02:51；作业 624223：32 CPU、20:43、总 CPU 10:03:21。两次累计墙钟 38:27、总 CPU 17:06:12，峰值 RSS 约 5.81 GiB。计算分别在 cpunode-038/036，GPU 请求及使用均为零，两个作业均 COMPLETED/0:0，已释放资源。

最终 30 条 API 调用使用 37,427 输入 tokens、3,712 输出 tokens，共 41,139；响应中位数 2.144 秒。按现有样本估计，45,465 条约 6,235 万 tokens；单并发、每次间隔一秒约 39.5 小时。该估计未验证全量吞吐，网关单价未取得。历史诊断的 7+30 条与最终 30 条共观测 111,762 tokens，另有一个被中断请求未取得用量回执。详情见 [pilot30-usage-and-integrity.json](pilot30-usage-and-integrity.json)。

## 验证与当前边界

预处理回归 5 项、标注客户端检查 5 项通过；全量所有记录的来源绑定核对通过。CPU 监控原始记录分别在 cpu-monitor.jsonl 和 cpu-monitor-v3.jsonl，最终资源/状态在相应 summary.json。

已输出全量标注的数据队列和固定的提示词、方法与 30 个试标结果。全量 LLM 标注、共享 AR/diffusion 模型训练和 GRACE 性质评价尚未执行；标注客户端当前限定于 30 材料试标。新生成晶体的 ehull 评价及训练契约仍由 Wayfinder 地图继续决定。
