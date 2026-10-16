# PLaID / PLaID++：任务、指标与主张证据

范围固定为 **PLaID 的 AI4Mat@ICLR 2025 workshop 论文**，以及 **PLaID++ arXiv:2509.07150 v1（2025-09-08）**；PLaID++ 的当前 arXiv 记录后来有更新版本，这里不把后续改动并入 v1 证据。作者代码仓库是 [`andaero/PLaID`](https://github.com/andaero/PLaID)，仓库 README 自称官方实现。

## 实际任务和 I/O

- **早期 PLaID**：LLaMA-2 7B 从 MP-20 学习 Wyckoff 结构文本，再用 MLIP 稳定性偏好做 DPO；论文主实验是无条件生成。作者把空间群或电子、磁性性质条件生成列为 future work。见 [AI4Mat@ICLR 2025 workshop paper](https://openreview.net/pdf?id=7aoP3ZeBfy)，§3.2–3.3、§4。
- **PLaID++ v1**：Qwen-2.5 7B 输出 Wyckoff 编码的晶体文本，编码包含化学式、空间群、晶格参数、元素、分数坐标、Wyckoff 位点与重数；生成文本可解析为 CIF。主评估为无条件生成，以及给定目标**空间群编号**的条件生成。条件示例是“spacegroup number is 15”，输出一份完整新晶体结构，而非复原给定目标结构。见 [arXiv v1](https://arxiv.org/html/2509.07150v1)，§4.1–4.3、§5.1、Appendix 7.2、Table 2。
- SFT 数据里还混有 infilling 任务；prompt 可随机带 1–5 项信息，包括化学式、ehull、每原子形成能、带隙、空间群，论文明确说主要关注空间群条件生成。但结果部分只报告无条件与空间群条件任务；局限性把带隙优化和离子电导率目标列为未来方向。**因此，v1 没有证明输入目标 ehull 数值/范围后，模型会生成达到该目标的结构。**见 [§5.1、§6](https://arxiv.org/html/2509.07150v1)。

## 指标与预算：不要混为一个“性质达成”数字

| 口径 | 论文中的定义与评估 |
|---|---|
| 无条件 S.U.N. | `SUN_count / N_generated`。S=MLIP-relaxed ehull ≤ 0 eV/atom；U=当前生成批次内不与其他输出结构重复；N=不与训练集结构匹配。Novelty/uniqueness 用 Pymatgen `StructureMatcher`。PLaID++ v1 的新颖性参考语料是 MP-20 训练集；其 relaxed ehull 使用 eSEN，并以 Materials Project（2023-02）及 MP2020 energy corrections 作参照。见 Appendix 7.1。 |
| 空间群条件 S.S.U.N. | 除 S.U.N. 外，要求 PyXtal 判定的空间群与请求群相符；稳定条件放宽为 metastable ehull ≤ 0.1 eV/atom。它不是无条件 S.U.N. 的同一阈值。见 §5.2。 |
| 偏好训练 vs. 评估 | DPO 偏好标签用 eqV2：stable ≤ 0、metastable 0 < ehull ≤ 0.08、unstable > 0.08 eV/atom；评估主要改用 eSEN，作者称此举用于降低对同一 MLIP 的过拟合。两处阈值和模型都应与 S.S.U.N. 的 0.1 eV/atom 门槛分开报告。见 §4.3、Appendix 7.2。 |
| 采样与 DFT | 无条件每个模型采样 10,000 个；空间群条件评估选择 7 群、每群 1,000 个：P1 (1)、C2/c (15)、Amm2 (38)、I-4m2 (119)、P3 (143)、P6₃/mmc (194)、F-43m (216)。eSEN 是主评估；另对 1,000 个生成结构做 DFT 验证，报告稳定率约 19.1%、S.U.N. 13%。作者指出 DFT 子集小于 eSEN 的 10,000 样本，unique 比率会更高，因此这两个 SUN 数不能当成同一分母下的直接比较。见 §5.1、§5.3、Appendix 7.2–7.3。 |
| 可解析样本边界 | 采样时 CIF 无法解析会重采，凑足所需结构数；所以 N_generated 是解析后纳入评估的样本数，不是所有原始解码尝试数。结构有效性仍另行计分。见 §5.1。 |

PLaID++ v1 报告 eqV2/eSEN 使用 500 个 relaxation steps、max force 0.02；eSEN 对 DFT near-hull 能量的 R²=0.84，eqV2 为 0.68。这些是代理验证证据，不等价于对全量输出做 DFT。见 Appendix 7.2–7.3。

实现侧细节：作者仓库 `evals/novelty.py` 先按配置的 E_hull 上限筛样，再在该筛选集内计算生成间 uniqueness、与 train split 的 novelty，最后计 S.U.N.；仓库中使用 `StructureMatcher()` 默认参数。复用评估代码时应记录实际 cutoff 与版本，因为论文正文概括为对其他生成结构计算 uniqueness。见 [evaluation implementation](https://github.com/andaero/PLaID/blob/9ac91f07a263594bbaef1574efbf483a98152f0c/evals/novelty.py)。

## 主结果与消融：PLaID++ v1 Table 1

下表摘录 Table 1 的 stability 和 S.U.N. 两列（百分比）；“—”表示原表没有给数。外部基线是论文汇总数；模型自身的评估主线使用 eSEN。

| 模型 / 变体 | Stability | S.U.N. |
|---|---:|---:|
| CDVAE | 1.6 | — |
| DiffCSP | 5.0 | 3.3 |
| FlowMM | 4.6 | 2.8 |
| FlowLLM | 13.9 | 4.7 |
| MatterGen-MP | 13.0 | — |
| CrystalLLM-70B | 5.28 | — |
| Jointly-trained ADiT | 15.4 | 5.3 |
| PLaID++ 非 DPO，3D Coord | 7.20 | 2.81 |
| PLaID++ 非 DPO，Wyckoff | 7.17 | 3.58 |
| Iterative SFT | 9.85 | 4.13 |
| PLaID++ DPO，3D Coord | 13.94 | 1.69 |
| PLaID++ DPO，Wyckoff | 15.59 | 6.25 |
| PLaID++ Stability + Novelty | 16.72 | 6.22 |
| **PLaID++ 完整模型** | **22.27** | **7.74** |

原文称完整模型比最佳先前基线的 S.U.N. 高约 50%；Table 1 中最佳先前 S.U.N. 是 ADiT 5.3%，7.74/5.3−1 约为 46%。3D Coord DPO 的稳定率升至 13.94%，但 S.U.N. 降至 1.69%，支持作者关于坐标表示下多轮偏好优化会坍缩多样性的分析。空间群条件结果在 Figure 4（七群直方图）报告；正文给的是相对增幅：仅用空间群偏好对 Wyckoff base 平均提升 22%，再联合无条件偏好后总提升 47%，不是逐群结果表。见 [§5.3, Table 1, Figures 3–4](https://arxiv.org/pdf/2509.07150v1)。

作为版本对照，早期 workshop PLaID 的 Table 1 报告 LLaMA-2 7B 的 relaxed stability 20.5%、S.U.N. 14.2%；其评估用 CHGNet，并从 10,000 个结构中随机抽 1,000 个评估稳定性。它没有报告空间群条件生成，future work 才提出做条件性质生成。故 14.2% 与 PLaID++ v1 的 7.74% 不应直接解释为性能下降：基座、MLIP 与样本口径不同。见 [早期 PLaID §3.2–3.3, Table 1](https://openreview.net/pdf?id=7aoP3ZeBfy)。

## “novel”有两个含义

1. **方法主张**：PLaID++ v1 把 Wyckoff 文本表示、RLIP/DPO、novelty 偏好、动态采样温度以及无条件和空间群条件联合训练作为贡献。早期 PLaID 已使用 Wyckoff 表示和稳定性 DPO；因此对 PLaID++ 来说，新增重点是条件任务和偏好/训练组合，不宜把“首次提出 Wyckoff 表示”当作相对 PLaID 的增量。论文自己也列出 WyFormer、CrystFormer 等先行 Wyckoff 相关方法；这里只整理两篇主论文，**没有建立领域范围的全局优先权结论**。见 [PLaID++ v1 §2、§4–5](https://arxiv.org/html/2509.07150v1) 与 [PLaID workshop §3–4](https://openreview.net/pdf?id=7aoP3ZeBfy)。
2. **S.U.N. 的 Novel**：指结构与 MP-20 训练集不匹配；Unique 指同一生成集合内去重。它们是按结构匹配与样本集合计算的指标，不表示实验上前所未见或已可合成。论文把“novelty”用于 DPO 偏好，但没有把 uniqueness 直接做成 pairwise 奖励，因为它依赖整批生成结果。见 [PLaID++ v1 §4.3、Appendix 7.1](https://arxiv.org/html/2509.07150v1)。

## 对 CrystalDLM 的证据边界

PLaID++ v1 支持的最具体反向设计任务是“可选空间群条件 + 输出完整晶体结构”，并用结构有效性、S.S.U.N. 与 MLIP/DFT 性质评价分别检查结果。它把 ehull 用作偏好排序和生成后稳定性判据，却没有验证“输入目标 ehull + 结构要求 → 生成结构达到该目标”。把后者设为 CrystalDLM 的研究问题是对论文任务空缺的推论，不是 PLaID++ 已展示的能力；若采用，须把目标条件和生成后性质评价分开写清楚。

### 一手来源

- PLaID++ 固定版本：[arXiv v1 HTML](https://arxiv.org/html/2509.07150v1)、[arXiv v1 PDF](https://arxiv.org/pdf/2509.07150v1)。定位：§4（方法/条件与偏好）、§5.1–5.3（任务/预算/结果）、Table 1、Figures 3–4、Appendix 7.1–7.3。
- 早期 PLaID：[AI4Mat@ICLR 2025 workshop paper](https://openreview.net/pdf?id=7aoP3ZeBfy)。定位：§3.2–3.3、Table 1、§4。
- 作者实现：[andaero/PLaID README](https://github.com/andaero/PLaID)；仅用于确认官方代码入口，不替代论文实验口径。
