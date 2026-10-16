# WyckoffDiff 与 WyckoffDiff-Adaptor 训练集规模核对

核对日期：2026-09-23。目标是与本项目 6,000 个训练样本比较“实际用于训练的结构数”，将全数据集总量、训练 split 和生成/评测样本数分开。

## 结论表

| 论文/实验 | 数据集与实际训练记录 | protostructure 去重 | 能否与 6,000 直接比较 | 依据与边界 |
|---|---:|---:|---|---|
| WyckoffDiff，arXiv:2502.06485v4，主实验 | WBM 全集约 257,000；另外各留出 10,000 验证、10,000 测试。作者 train CSV 有 **236,838 行**，且官方训练 Dataset 原样逐行转为样本，默认不去重，未设置 `pre_filter` 时也不丢行。因此论文默认训练 Dataset 长度为 **236,838 个样本**。 | 不适用：训练集不去重，重复记录若存在仍会作为训练样本保留。 | 可以按训练样本条目比较：236,838 是 6,000 的 **39.5 倍**。 | v4 §4.1 p.8 报告 WBM 257k、留出 10k+10k；官方代码 `dataset.py` 的 `process()` 将预处理 dataframe 每行转图后直接保存，只有显式提供 `pre_filter` 才过滤，而构造 Dataset 默认 `pre_filter=None`；`get()` 只对同条记录多个等价 Wyckoff 编码随机选一种，不改变 Dataset 长度。 |
| WyckoffDiff，Carbon24 附加实验 | 论文写明使用 DiffCSP++ 仓库中的同一 training set；WyckoffDiff 训练 4,000 epochs。统计生成/新 protostructure 时各用 1,000 个，不是训练数。**精确 train 记录数在论文该段未写出。** | 未在本文核实出官方处理文件的实际去重数。 | 可作补充实验背景，不应将 1,000 当成训练集规模。 | v4 Appendix E p.16。作者数据 README 称 repo 实验只用各数据集 train split，但没有列 Carbon24 计数。 |
| WyckoffDiff-Adaptor，arXiv:2601.08115v1 | 论文明确称使用 **MP-20**；称 base model 训练 900 epochs、Adaptor fine-tuning 200 epochs，但没有报告各自的 split、记录条数、去重数或是否过滤。Data Availability 也只说源自 MP-20。故准确结论为：**具体实际训练记录数未披露**。 | 未披露。 | 当前不能做严谨的记录数倍数比较。MP-20 名称代表数据集，不等于训练 split 大小。 | v1 §2.2 p.4、Data Availability p.9。作者官方仓库 README 说明需下载 `data-release/mp-20/mp_20.zip` 再预处理，但没有在 README 写出 split 样本数；Git LFS 数据包当前未能通过免认证媒体 URL读取成 ZIP。 |

## 判读说明

1. WyckoffDiff 论文中“total of 257k materials”是 WBM 总体量，不是训练量；正文紧接着说留出 10k+10k。官方 `httk/wyckoffdiff` 的 `data/README.md` 又明确说虽有 train/val/test splits，论文实验只用了 train split。按作者实际公开处理文件计数后，train CSV 为 236,838 条，约与 257k − 20k（四舍五入后的总数）一致。
2. WyckoffDiff 论文报告的 10,000/1,000 主要是生成或评测统计规模，不是训练样本数。其 WBM 训练规模的 10% 消融也属于子集 ablation，不能替换主模型的全 train split 规模。正文 Appendix A.3 报告批量 256、1,000 epochs；epoch 数不改变样本数。
3. **训练 Dataset 不去重。** 作者官方仓库初始公开 commit `58a5afb432bc38498807fb29357b07a4547275bb`（2025-04-17）的 `wyckoff_generation/datasets/dataset.py` 已显示：`process()` 对 dataframe 每行构造一个 Data，并直接保存；仅当调用方显式传入 `pre_filter` 才过滤。Dataset 构造函数的 `pre_filter` 默认 `None`，训练 loader 调用构造器时也没有传入它。当前代码仍保持相同处理逻辑。`get()` 按样本索引加载后，可能在该条记录可接受的等价 Wyckoff 编码间随机选一个，这改变编码采样、不改变样本条数。因此实际 train Dataset 为全部 236,838 行。
4. WyckoffDiff 的 Carbon24 是额外数据集实验。Appendix E 明确“same training set as in the DiffCSP++ repository”，但该段没有实际样本数。评测中“1,000 protostructures and 1,000 novel protostructures”是生成统计样本数。
5. WyckoffDiff-Adaptor 的 “base model” 在本文上下文是 Adaptor 微调前的 WyckoffDiff 模型阶段；论文说 base 与 fine-tuned model 使用 MP-20，分别 900 / 200 epochs，但并未说两阶段各自具体采用何种 split 或训练记录数。官方仓库操作步骤确认从 MP-20 数据包预处理并支持 train / finetune 两类命令，但公开 README 同样没列 split row counts。不能用 MP-20 全库约 45k 来替代实际训练 split 计数，也不能把 900/200 epoch 或 64/1,280 个生成 trial 当作训练数据数。

## 原文与官方来源

- WyckoffDiff 固定版本全文：[arXiv:2502.06485v4](https://arxiv.org/abs/2502.06485v4)。§4.1 p.8：WBM 来源和 257k 总量、留出 10k+10k；Appendix A.3 p.14：训练超参；Appendix E p.16：Carbon24 附加实验及 1,000 条生成统计。
- WyckoffDiff 官方实现和数据说明：[httk/wyckoffdiff README](https://github.com/httk/wyckoffdiff)（Data 部分列出 WBM 主实验和训练命令）；[官方 data/README.md](https://github.com/httk/wyckoffdiff/blob/main/data/README.md)（行 197–202 说明只用 train split，行 236–239 列出作者处理版 WBM train/val/test 文件）；[初始公开 commit 的 Dataset 实现](https://github.com/httk/wyckoffdiff/blob/58a5afb432bc38498807fb29357b07a4547275bb/wyckoff_generation/datasets/dataset.py#L278-L321)（逐行构造保存、不去重；`pre_filter` 默认空；getitem 随机采样等价编码）。
- 计数文件：[作者公开 WBM train CSV](https://public.openmaterialsdb.se/wyckoffdiff/data/wbm/raw/wbm_train.csv.bz2)。读取该压缩 CSV 的记录数为 236,838，SHA-256 为 `a0898053c5c124484d97030738e25e0567d33896c3be0a65b1bc22e701807ca6`，与官方 Dataset 代码内 train 文件 hash 完全一致。因此这是 dataloader 实际读取的 train split，而非同名的另一份数据。文件响应大小 41,240,030 bytes，ETag `"27545de-632f63da79380"`，Last-Modified 2025-04-17。
- WyckoffDiff-Adaptor 固定版本全文：[arXiv:2601.08115v1](https://arxiv.org/abs/2601.08115v1)。§2.2 p.4（全文 extracted text 行 212–224）：MP-20、base/fine-tuning epoch 数；Data Availability p.9（行 461–469）：来自 Materials Project 并对应 MP-20 内容；两处均未给出训练 split 计数。
- Adaptor 官方实现：[pfnet-research/wyckoffdiff_adapter README](https://github.com/pfnet-research/wyckoffdiff_adapter)（行 180–207：下载 MP-20 数据包、预处理、从头训练和 fine-tune 命令）。可读取到仓库 LFS pointer（ZIP 对象 SHA-256 `5f44d6ad75299b6c08a3679ebcbd166def45c6d7733baacdce3972c2ec452e3a`，声明大小 12,254,159 bytes），但直接媒体 URL 返回空响应；本次没有根据不完整下载去猜 split 数。

## 给 6,000 样本比较用的建议表述

“WyckoffDiff 的 WBM 主实验 train split 有 236,838 条记录；官方训练 dataloader 按行读取且默认不去重，因此实际 Dataset 长度也是 236,838（约为我们的 6,000 条的 39.5×）。WyckoffDiff-Adaptor 论文只明确 MP-20，未披露其实际训练 split 的样本数，因此不纳入精确平均值。”

这里的倍数是对样本数作除法，不代表各工作任务、数据过滤、去重、模型训练目标或训练数据独立性相同。两篇文献也不足以支持声称“过去参考文献平均数据规模”；稳妥的比较是列出可确认的单篇量级并把未报告值留空。
