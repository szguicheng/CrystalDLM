# 晶体条件生成模型对比

核对日期：2026-09-07。范围为本地找到的 2026-09-01 报告 `group_meeting_draft.pptx` 中的 7 个相关模型，并补入直接相关的 WyckoffDiff-Adaptor。以下依据论文正文、方法、实验及相关附录整理，不以摘要或旧 PPT 的结论代替原文。我们的模型尚属拟议方案；用户本次确认条件类型尚未确定。

## 模型对比表

AR = autoregressive，自回归；非 AR = non-autoregressive，非自回归。**生成范式与是否条件生成是两个独立维度**。Protostructure 指空间群和元素占据的 Wyckoff 描述，尚缺完整晶格与自由坐标。

| 模型 | 表示 / tokenizer 的基本单元 | 生成范式与解码方式 | 条件生成：原文做了什么 | 模型直接输出 | 与我们拟议方案的主要关系 |
|---|---|---|---|---|---|
| **CrystalFormer** [1] | 人工定义的 Wyckoff letter、元素字段；自由坐标与晶格用连续分布建模，无 BPE 合并 | **AR**；按 Wyckoff 字母序，依次生成位点、元素、坐标，最后晶格 | 空间群条件；已做结构补全，以及外接性质预测器的 **MCMC 性质引导采样**，包括带隙、形成能 | **完整晶体结构** | 条件生成和结构补全已有先例；我们拟更换 tokenizer 和 AR 生成机制 |
| **Matra-Genoa** [2] | 可逆的晶体字段序列：元素、计量、稳定性、空间群、Wyckoff 位点、自由坐标、晶格；离散与连续嵌入混合 | **AR**；条件前缀后逐项预测 | 已做稳定性 LOW/HIGH 前缀、元素化学系统条件；空间群和 Wyckoff 位点可作为前缀约束，但没有同等完整的独立条件基准 | **完整晶体结构** | 已有结构化编码与条件生成；主要需比较 motif/set tokenizer 与非 AR 去噪的作用 |
| **WyFormer / Wyckoff Transformer** [3] | 每个 orbit 用 **元素 + site symmetry + enumeration** 复合单元表示；不含自由坐标；字段分别嵌入，无 BPE 合并 | **AR**；element → symmetry → enumeration 级联；编码器无位置编码，训练打乱 orbit 顺序 | 空间群条件生成；性质预测是另一任务；化学计量条件 CSP 列为未来工作 | **Protostructure**；后接 PyXtal / 几何生成与弛豫 | 与我们的 Wyckoff 集合表示最接近；“输入无序”本身不能作为非 AR 创新，差异在 motif 编码与扩散生成 |
| **WyckoffDiff** [4] | 固定空间群下的 **Wyckoff 位点 × 元素占据/计数** 离散变量；不是文本分词器 | **非 AR，D3PM 离散扩散**；WyckoffGNN 联合去噪 | 结构上以空间群为条件，主流程从经验分布抽取空间群；主实验为 de novo 生成，未做性质或给定化学式的条件任务 | **Protostructure**；PyXtal 等补几何 | 已覆盖 Wyckoff 表示 + 非 AR 离散扩散；我们的 tokenizer 必须有比占据矩阵更具体的区别 |
| **SymmCD** [5] | 不对称单元：orbit 代表原子、位点对称矩阵、元素、连续坐标和晶格；不是文本 BPE | **非 AR，离散 + 连续联合扩散** | 空间群作为输入，主流程按先验采样；未报告化学式、性质或部分结构条件基准，组成条件 CSP 是已述局限 | **完整晶体结构**；投影到合法 Wyckoff 位后展开 | 已有对称压缩表示与非 AR 扩散；我们的潜在区别在离散 motif 编码及选定后的条件任务 |
| **DynaCrys** [6] | **空间群 + orbit 数 + 无序的 Wyckoff row / 元素占据集合**；共享学习的对称 codebook，无 BPE 合并 | **非 AR，两阶段扩散**；动态空间群的符号扩散，再生成几何 | 主实验明确为 **unconditional generation**；第二阶段以自生成 protostructure 为条件，不等于用户指定条件的任务 | **完整晶体结构**；中间为 protostructure | 已有无序集合、对称 codebook 与扩散；我们的条件任务定位不同，tokenizer 仍需与其 codebook 机制区分 |
| **MaskGXT** [7] | **逐原子/逐字段 token**：空间群、晶格 bins、坐标 bins、Wyckoff、元素；附 sub-bin 连续修正 | **非 AR，掩码生成模型**；MaskGIT 式按置信度迭代并行解 mask | 已做 **给定化学组成的 CSP**；原子类型及数量已知，预测晶格、坐标及对称信息 | **完整晶体结构** | 已覆盖 conditional + 非 AR masked generation；若我们选组成条件，它是最直接的比较对象之一，差异不能只写“不是 AR” |
| **WyckoffDiff-Adaptor**（补充）[8] | 沿用 WyckoffDiff 的占据/计数表示，在 WyckoffGNN 中加入条件 Adaptor | **非 AR，条件离散扩散** | 已联合使用 **空间群 + 元素化学系统 + energy above hull** 条件；元素系统不等同于固定化学计量式 | **Protostructure**；PyXtal 补几何，再弛豫 | 已覆盖对称感知条件扩散；若我们选稳定性/元素条件，需要明确与该方法的 tokenizer 和条件处理差别 |
| **我们的模型：CrystalDLM（拟议）** | **新 Wyckoff tokenizer**；PPT 候选为 set-level BPE，将合法共现 orbit 合并为可逆 motif，保留集合/对称信息；具体规则待定 | **非 AR diffusion / DLM**；拟以双向去噪恢复 masked 单元，具体目标与采样算法待定 | **以 conditional generation 为主要任务；条件类型待定**，暂不选定化学式、性质或部分结构 | **待定**：protostructure 或完整结构；PPT 尚未定案 | 拟验证的组合是 **新的集合/motif tokenizer + 非 AR 扩散 + 明确的条件任务**；效果与创新性尚待实验 |

## 表格结论与用词边界

1. **不能把已有模型概括为“AR、无条件生成”。** WyckoffDiff、SymmCD、DynaCrys 已做非 AR 扩散；MaskGXT 已做组成条件的非 AR 掩码生成；WyckoffDiff-Adaptor 已做条件离散扩散。[4–8]
2. **条件生成也不是 AR 模型缺失的通用能力。** CrystalFormer 有结构补全和性质引导 MCMC；Matra-Genoa 有稳定性/化学系统条件。需要比较条件进入方式和实际任务，而不是只勾选“支持条件”。[1–2]
3. **“Tokenizer 不同”需要落到编码单元上。** 已有工作分别采用复合 orbit 单元、占据矩阵、混合连续/离散字段、固定量化 bins 和对称 codebook。PPT 的候选区别是可学习合并的 orbit motif，以及集合等价性下的稳定、可逆编码；这仍是设计假设，不能写成已证明的性能优势。[1–8；本地 PPT 第 8 页]
4. **MaskGXT 保留原文分类。** 表中写 MaskGIT 式掩码生成与迭代并行解码，不将它直接等同于 D3PM，也不武断排除它与 masked diffusion 的数学联系。[7，§2、§4.3–4.5]

“无条件”是没有用户指定目标条件，不代表盲目的均匀随机生成；条件生成同样可以随机采样。表中用 unconditional / conditional，比“随机生成 / 非随机生成”更准确。

可用于汇报的当前定位：**我们拟研究面向条件晶体生成的 Wyckoff 集合/motif tokenizer，并结合非自回归扩散生成；要验证的差异是 token 单元及其对条件任务的作用，而非单独宣称首次使用扩散或条件生成。条件类型与输出层级尚待确定。**

## 原文证据与阅读版本

以下页码均为对应版本 PDF 页码。链接固定到本次所读版本，避免后续改版造成页码漂移。

- **[1] CrystalFormer** — *Space Group Informed Transformer for Crystalline Materials Generation*, [arXiv:2403.15734v3](https://arxiv.org/abs/2403.15734v3)。§II.A、Eq.1：表示与 AR 分解，PDF pp.2–5；§III.B、Fig.8：结构条件元素替换/补全，pp.9–10；§III.C、Fig.9 及 Appendix E：性质后验与 MCMC，pp.10–11、28–29。特别区分外接预测器引导与生成器内部条件 token。
- **[2] Matra-Genoa** — *A generative material transformer using Wyckoff representation*, [arXiv:2501.16051v2](https://arxiv.org/abs/2501.16051v2)。§II.A–B、Fig.1：可逆混合表示、条件前缀与 AR 采样，pp.2–3；§II.C、Fig.3：Al–Ca–Cu 化学系统实验，p.4。稳定性 LOW/HIGH 以 0.075 eV/atom 的 hull distance 为界；该化学系统实验没有给定完整化学式。
- **[3] WyFormer** — *Wyckoff Transformer: Generation of Symmetric Crystals*, [arXiv:2503.02407v4](https://arxiv.org/abs/2503.02407v4)。§2.1–2.4、Fig.4：复合 orbit 单元、无位置编码、AR cascade 与几何后处理，pp.4–5；§3.2 是性质预测；结论 p.9 将 stoichiometry-conditioned CSP 列为未来工作。Appendix R p.29 补充 token 词表分析。
- **[4] WyckoffDiff** — *WyckoffDiff — A Generative Diffusion Model for Crystal Symmetry*, [arXiv:2502.06485v4](https://arxiv.org/abs/2502.06485v4)。§3.1–3.4、Eq.3–6：占据/计数变量、`p(s,z)=p(s)p(z|s)` 与 D3PM，pp.3–5；§5、Fig.3：protostructure 后的 PyXtal/MACE/DFT 发现流程，pp.8–9。后处理稳定性筛选不等于性质条件去噪。
- **[5] SymmCD** — *SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models*, [arXiv:2502.03638v3](https://arxiv.org/abs/2502.03638v3)。§4.1–4.3：不对称单元与混合扩散，pp.4–7；Algorithm 2 p.7：空间群输入、合法位点投影与展开；Conclusion p.10：composition-conditioned CSP 局限。
- **[6] DynaCrys** — *DynaCrys: Crystal Generation with Dynamic Space-Group Diffusion*, [arXiv:2608.07401v1](https://arxiv.org/abs/2608.07401v1)。§4.1–4.4：无序 occupation 集合、codebook、动态空间群、几何阶段，pp.4–6；§5.1 p.7 明确主实验为 unconditional；Appendices C–F 细化 codebook、去噪与几何。学习的 embedding/codebook 不等于 BPE 合并词表。
- **[7] MaskGXT** — *Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist*, [arXiv:2606.22866v1](https://arxiv.org/abs/2606.22866v1)。§4.1、Fig.3：`1+6+5N` 编码，pp.5–7；§4.3–4.5：掩码学习与采样，pp.7–9；§5.3 p.10：composition-conditioned CSP；Appendix A p.16 允许固定空间群的可选接口，但本文主结果不是独立给定 SG 基准。§4.1 说明对称 token 与几何的一致性通过训练鼓励，并非硬约束。
- **[8] WyckoffDiff-Adaptor** — *Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models*, [arXiv:2601.08115v1](https://arxiv.org/abs/2601.08115v1)。§2.1–2.2：条件 Adaptor 注入 WyckoffGNN 与 SG/chemical system/energy above hull 条件，pp.4–5；§3、Figs.2–5：相图、对称保持与几何限制，pp.6–8。PyXtal 是几何后处理；Adaptor 本身参与去噪网络，不能说成仅做后处理。

## 核对范围与限制

- 3 个 Luna 子智能体分别通读论文正文及相关附录，主智能体汇总，并复核了影响比较结论的关键原文：MaskGXT 的条件任务和分类、WyckoffDiff 的 SG 条件分解、CrystalFormer 的补全/性质引导、WyFormer 的未来工作、Adaptor 的条件类型等。
- 8 份本地 PDF 的 SHA-256 均与已有完整性预检匹配，PDF 页数与提取文本页标记一致。`source_manifest.json` 保存具体映射；旧目录的 `dynacrys.pdf` 实际是 WyckoffDiff，`maskgxt.pdf` 实际是 WyFormer，已按内容和 arXiv 编号辨认。
- 这是基于报告范围的比较，不是对整个晶体生成领域的系统综述，也不支持“全球首个”或“尚无人做过”的结论。未运行这些模型，不横比任务和评价协议不同的性能数字。
- 本文由 AI 辅助阅读全文与汇总；不代表用户已人工读完全部论文。逐篇详细核对笔记保存在 `notes/`。
