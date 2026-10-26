# S.U.N. 口径核对

阅读范围：以下论文的指标方法、结果表和相关附录，以及 FlowMM 官方评估代码；本次没有重读所有章节。

| 来源 | 评估对象和计算方式 | 影响比较的细节 |
|---|---|---|
| SymmCD，ICLR 2025 正式稿 §5.2 / Table 2 | 每模型 10,000 个生成晶体；CHGNet 估计能量并弛豫，与 MP 凸包比较，再判唯一和训练集新颖性 | All-SGs 的 relaxed stable 为 9.34%，relaxed SUN 为 6.86%。§5.3 的另一次 100 个样本 CP2K 验证不能与这张 CHGNet 表混为一谈。 |
| WyFormer，arXiv:2503.02407v4 Table 1 / Appendices G–H | 同一表中同时存在 105 样本的 DFT 估计（阈值 <80 meV/atom）和 10,000 样本的 DFT 评估（阈值 <0 meV/atom） | 1000/105 样本口径以 train+val 判新颖；10,000 样本口径只用 train。不能拿同表不同列当相同指标。附录说明 DFT 不收敛按不稳定计。 |
| FlowMM 官方 scripts_analysis/ehull.py | 同时支持 CHGNet 近似能量及 DFT 能量；CHGNet 用总能量除原子数，再减 MP 2023-02-07 hull 的单位原子能量 | README 明确最准确的 SUN 应使用修正后的 DFT 弛豫能量。我们本轮仅采用其中的 CHGNet proxy 路径。 |
| 本次 paired pilot | 同一组组成条件，每种表示 512 次；已有的预测晶格对称化后结构，经相同 CHGNet 弛豫与相同匹配规则 | conditional、样本规模、额外有效性门槛均不同于论文；可以比较我们自己的两种表示，但不能凭裸分数声称超过这些论文。 |

S.U.N. 的联合分数应按同时满足条件且去重后的数量 / 全部生成次数计算。单独 S、U、N 的边际比例通常不能直接相乘。对 composition-conditioned generation，还应并列报告组成满足率、参考重建率；对于指定已知晶体的恢复任务，新颖性并不总是希望越高越好。

本次 N 仅表示相对于训练集的新颖。CHGNet 自身训练数据与 MP20 重叠；若后续要做新材料发现的强结论，需要更广的结构参考库以及独立 DFT/进一步物理验证。

来源：
- https://openreview.net/pdf?id=xnssGv9rpW
- https://arxiv.org/html/2503.02407v4
- https://github.com/facebookresearch/flowmm/blob/main/scripts_analysis/ehull.py
- https://github.com/facebookresearch/flowmm/blob/main/src/flowmm/chgnet_.py
- https://github.com/facebookresearch/flowmm/blob/main/README.md
- https://figshare.com/articles/dataset/22715158
