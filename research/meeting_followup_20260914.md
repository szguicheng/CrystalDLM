# Crystal DLM 2：会后任务补充

2026-09-14。本文件保留详细证据。PPT 已按用户要求精简为新增第 16–22 页：基础模型 2 页、位点解释 2 页、跨空间群比较 1 页、边轨道编码 2 页。原有 15 页（含原有第 15 页空白模板）保留。

## 文献判断

**已有与老师设想直接交叉的基础模型。不能把“多条件统一成文本，混合多种数据，适配不同任务”本身作为未有人做过的创新点。** 以下按方法、训练样本和输出范围区分：

| 工作 | 实际相关之处 | 边界与阅读定位 |
|---|---|---|
| [NatureLM](https://arxiv.org/html/2502.07527v2) | 单域和跨域科学序列混训，材料性质转为文本 | §2.1、§5、§8.3。材料基本序列是组成和空间群；完整 3D 由另行微调的 Mat3D 模块生成 |
| [LLaMat](https://doi.org/10.1038/s42256-026-01199-8) | 材料文献和 CIF 训练，后续多任务适配 | [预印本 §4](https://arxiv.org/html/2412.09560v1) 核对训练方法，期刊页面核对出版元数据。Chat/CIF 有分支，不能合写为一个通用 checkpoint |
| [ALM](https://arxiv.org/html/2606.21395v1) | 结构 soft tokens、LLM、扩散解码器；理解阶段混合结构任务与纯文本 | §2、Appendix B.1/B.2。Core/Edit/Gen 有阶段和变体，不等于任意字段组合均已解决 |
| [MatterChat](https://doi.org/10.1038/s42256-026-01214-y) | 结构编码器 + LLM 做多任务预测 | Methods/Discussion。研究数据以结构为基础，未展示无结构记录混训或晶体生成 |
| [MultiMat](https://arxiv.org/html/2312.00111v3) | 明确用损失掩码处理缺失 DOS / 电荷密度 | §IV.2、[官方实现](https://github.com/vmoro1/multimat)。实验中结构与文本存在于全部样本；表征预测/检索不等于生成 |
| [MatBind](https://arxiv.org/html/2607.08470v1) | 结构锚定对齐 pXRD/DOS/文本，可缺部分模态 | §5。结构和 pXRD 全部存在，DOS/文本不完整；无结构记录不是该实验的缺失形式 |

其他重要交叉：

- [AtomGPT](https://arxiv.org/html/2405.03680v1)：文本化结构做性质预测 / 条件生成，正向 GPT-2 与逆向 Mistral 分开训练。
- [CrystaLLM-π](https://arxiv.org/html/2511.21299v1)：无标签 CIF 预训练，再使用连续性质 / XRD 条件。连续条件进入注意力，并非全做离散文本 token。
- [CLOUD / SCOPE](https://doi.org/10.1038/s41467-026-70467-3)：空间群生成元、WP、组成的对称字符串与 masked 预训练，后续性质预测。
- [Chameleon](https://doi.org/10.1038/s41467-025-59636-y)：Crystal CLIP 加文本条件扩散。
- [Gruver et al.](https://arxiv.org/abs/2402.04379)：较早的晶体文本生成、结构补全和文本条件生成先例。

这些来源不足以支持“完全没人做过”，也不支持“所有人的方法完全等同于我们的”。需要将创新限定到具体的不完整数据监督机制、边轨道表示、非自回归目标与可复用任务的实验收益。没有横比跨论文性能数字，因为数据和评估协议不同。

## 建议训练组织（未实现）

设 O 为观测字段集合，从 O 中随机选择 M 遮盖，仅对 M 的真值计算重建损失。原始缺失字段不产生该字段的监督，不能补成 0 或伪真值。

- 组成 + 性质、无结构：可学组成预测性质，或性质条件的组成补全。
- 有结构、无性质标签：可学结构 token 自监督重建或结构补全。
- 有部分性质的结构记录：随机选择已知字段为条件，其余已知字段为目标。
- 只有文本：可学语言重建 / 问答；仍需配对记录建立文本与结构联系。

`[ABSENT]` 是数据库没有提供，`[MASK]` 是已知值被训练过程遮住。二者均为拟议接口。能量必须注明类型（总能、形成能、凸包能）、单位（eV/cell 或 eV/atom）、来源与条件。同一化学式有不同晶型或实验状态，不能只按化学式强行配对标签。互不关联的数据不会因为都成为文本而自动产生可辨识的跨模态关系。

应隔离四组对照：仅完整样本 / 加不完整样本，CIF/WP/边轨道，AR/非AR，单任务/共享模型。匹配预算，按材料身份、原型和跨来源重复隔离测试集，评价性质误差、合法性、条件命中率、吞吐、少样本迁移与负迁移。

## Wyckoff 示例的核对

定义依据 [IUCr Wyckoff position](https://dictionary.iucr.org/Wyckoff_position) 和 [multiplicity 字典](https://www.iucr.org/__data/iucr/cifdic_html/3_orig/CORE_DIC/Ispace_group_Wyckoff.multiplicity.html)。坐标对照 [PyXtal 官方数据库](https://raw.githubusercontent.com/qzhu2017/PyXtal/master/pyxtal/database/wyckoff_list.csv) 与项目固定版本 `experiments/wyckoff_vocab_gate1/input/wyckoff_list.csv`，PyXtal 1.1.4 的来源和散列在同目录 `provenance.json`。

- NaCl #225：Na 4a 与 Cl 4b 各展开四个位置，满占据常规晶胞 8 个原子，原胞只有一个 NaCl。重数 4 不是配位数 6。
- #62 Pnma 4c：`(x,1/4,z)`，site symmetry `.m.`，两个自由参数。取 x=.10,z=.20 是展开演示值，不是实测材料。
- #139 I4/mmm 4e：标准代表 `(0,0,z)`；mp-30009 的 F 轨道可取 z=.290407，本地编码选用等价代表 `(.5,.5,.790407)`。
- 同名 4a：#225、#216、#62、#19 的 site symmetry 分别为 m−3m、−43m、−1、1，最后一个有三个自由参数。
- 同名 4c：#216 固定在 `(1/4,1/4,1/4)`；#62 为 `(x,1/4,z)`。
- #221 1a 与 #225 4a 可以共享 m−3m 特征，但晶格居中和重数不同；#225 4a 与 4b 同为 m−3m，也必须区分具体 WP。

跨群复用应共享特征 / 参数，同时保留空间群、Hall 设置和 WP 身份；[WyFormer](https://arxiv.org/abs/2503.02407v4) 的 enumeration 也依赖空间群相关映射。PPT 全部采用标准设置，展开坐标取模 1。

## 实际边轨道实例与计数口径

本次复算 `work/edge_orbit_release/examples/mp-30009.prototype.json`、`.geometry.json` 与 `codec/discrete_graph_prototype.py`，没有训练新模型。

全局 SG=139、Hall=424，32 个 Seitz 操作；2 个位点轨道恢复 6 个原子。Kr 的自由度为 0，F 的自由度为 1。晶格约为 `diag(4.383300,4.383300,6.668202)` Å。

实际边 `[0,1,16,-1,-1,-1]` 解码为 `R16 qF + t16 + n = (0,0,.290407)`，距离约 1.93649 Å；只将 g 换成 0 则端点变成 `(-.5,-.5,-.209593)`，距离约 3.40000 Å。g 是当前操作表局部编号，n 是整数晶胞编号。

七条边轨道分别恢复 `4,16,4,8,2,2,8` 条规范化**无向周期边**，合计 44；加上反向边后是 88 条有向邻接。决定性代码证据是 `symmetry_graph_codec.edge_key` 将一条边与其反向归为同一键，`transform` 也返回此规范键。

4.5 Å 是原始结构的建图半径，不代表每条边都是化学键。改变自由几何可以维持抽象连接，但不保证仍是新几何的 4.5 Å 邻居图；特殊退化坐标还可能改变轨道重数。44/7 只比较边记录，未计入全局操作表与位点约束，不能当作整体 token 压缩比。

WP 与仿射稳定子同时出现，是将选定代表的坐标约束显式化以统一解码。设置和代表固定后可查表导出，不是额外独立物理量。当前是编解码记录格式，不是已训练的语言模型词表。

既有结果范围见 `research/edge_orbit_tokenizer_proposal.md`：表示往返、固定图网络计算和单级条件几何去噪已有验证；离散生成、多步反向采样和新材料稳定性尚未完成。ASU、WP 与周期图都有先例，具体编码与任务收益需要独立论证。

## 检索范围与局限

查询覆盖 materials foundation language model、heterogeneous/incomplete data、missing modalities、text space、unified sequence、NatureLM、Atomistic Language Modeling、MultiMat、MatBind、Wyckoff multiplicity，并沿引用链补充直接相关模型。主要核对方法、数据、训练混合与输出范围，未声称逐页通读全部附录或复现实验。本次是针对性检索，不是穷尽性系统综述。论文结论来自原始来源，搜索摘要、博客仅用于发现线索。使用 AI 辅助检索、整理与排版，未进行新的模型训练或付费评测。
