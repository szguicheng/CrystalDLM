# 基础模型选择证据（2026-10-13）

**判断：**现有方法支持从预训练 AR 模型继续训练；不足以证明本任务必须从头训练，也不足以把晶体失效归因于 1.5B 参数规模。

- [Llamole v1，附录 C.3、B.2.3](https://arxiv.org/html/2410.04223v1)：底座是 Llama-3.1-8B、Mistral-7B、Qwen2-7B，LoRA 微调四轮；Graph DiT、GNN predictor、graph encoder 另行预训练并在微调时冻结。Llama-3-70B 用于离线生成分子描述，不是 Llamole 底座。该工作是分子图证据，不是晶体验证。
- [FLARE v2，§4–5](https://arxiv.org/html/2606.01774v2)：受控迁移从预训练 Qwen3-1.7B 开始，4096 长度、batch 256、9000 步（约 10B token）；另以 post-trained Qwen3.5-2/4/9B 作 AR 起点并进行单阶段 SFT。[作者仓库](https://github.com/yuchen-zhu-zyc/HybridDiffusion)的 model zoo 也将 2/4/9B 对应到 Qwen3.5。它支持“用预训练权重继续训练”，但语言基准不保证晶体效果。
- [Qwen 官方公告](https://qwenlm.github.io/blog/qwen2.5/)标记 Qwen2.5 于 2024-09-19 发布；[模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)为 1.54B。因此它相对较旧且较小，但“太小”尚未证实。[当前纯 AR 一轮基线](../../experiments/ehull_pure_ar_baseline_20261013/RESULTS.md)用 7,272 条、909 更新；32/32 到达 `<CRYSTAL_START>` 且边界完整，严格结构审计 7/32，性质评估未运行。条件只有 ehull，lead 为 greedy；这些结果不能单独检验模型规模是否导致结构无效。
- 可作较新同路线对照的是 [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：官方卡标为 4B、causal、non-thinking；[Qwen3 技术报告](https://arxiv.org/html/2505.09388)称 4B 属 dense 系列且架构类似 Qwen2.5。若后续比较，应固定数据、更新数和生成设置。Qwen3.5 则采用 3:1 Gated DeltaNet 线性注意力与 full attention；结合 FLARE 对混合骨干状态调度和 mask 的专门处理，推断它不是当前 Qwen2.5 双向 mask 路径的直接替换。[Qwen3.5 官方卡](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

本 note 仅做证据核对；未下载模型、启动任务或决定换底座。
