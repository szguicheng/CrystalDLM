# AR 检查点转 MASK diffusion：证据简记

只读查证，2026-10-03；未改训练源码、未提交作业。以下来源事实来自两篇作者原论文；其余为针对本仓库的推断/建议。

## 来源事实

- [DiffuGPT / DiffuLLaMA，Gong et al., ICLR 2025](https://arxiv.org/html/2410.17891)：直接把 GPT-2/LLaMA AR 权重继续训练成离散 MASK diffusion。训练采样噪声输入，只在 MASK 位计算去噪 CE，并将 logits 右移一格，让位置 `i−1` 对齐干净目标 `i`。GPT-2 配方用 10K steps 将因果注意力逐步退火到全注意力；7B DiffuLLaMA 则直接用双向注意力。论文报告 127M/355M 与 7B 规模，仍使用数百亿级 token 继续训练。
- [Dream 7B，Ye et al.](https://arxiv.org/html/2508.15487)：同样以 AR 权重初始化；主模型沿用 Qwen2.5-7B 配置，预测对齐仍是 `h_i → x_{i+1}`，并转入 MASK diffusion 的双向上下文。其 1B 对照显示 AR 初始化在从因果转全注意力时也会经历初期高 loss；该对照训练 200B tokens，Dream 7B 训练语料为 580B tokens。

## 对本任务的含义（推断）

直接先例支持“保留 AR 表征，再用 masked denoising 继续训练”作为可检验假设；不支持它能在 0.5B、1024 个材料上达到晶体语法或物理有效率。DiffuGPT 的 355M 接近目标参数量，但论文训练使用数百亿至千亿级文本 token；Dream 的 Qwen2.5-7B 实验也用 580B tokens。自然语言与 PLaID 结构串、Qwen2.5-0.5B-Instruct 与论文底座都不匹配。Dream 还报告因果转双向会产生初期 loss 跳升，学习率过高会损伤已学 AR 能力。

本地接口也有一个关键错位：[model.py](/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM/experiments/interleaved_ar_diffusion_20261002/model.py:29) 中 AR 目标用前一位置 logits，当前 hybrid 目标与采样都用同一位置 logits；当前晶体块虽双向，块前的 `START` 查询还不能读取该块。若要检验 AR 权重迁移，处理组应让 `h_{i−1}` 预测被遮盖的 `x_i`，采样也用同一对齐，并让首个晶体 token 的前一位置查询可见当前噪声块。否则实验只检验“AR 初始化 + 位置未对齐的现有 H 目标”。

## 最小、可证伪的后续试验（建议，未启动）

等本轮 AR 4096-update 验证关闭后，冻结终点检查点及其哈希，不按验证挑最佳点。复制同一权重为两臂，使用相同 tokenizer/数据清单/1024 训练材料、任务顺序、seed 和新初始化的同一 AdamW/LR 日程，各继续 512 updates（与本地 joint 阶段预算同量）：

1. **控制 A：**继续 AR 文字和晶体 next-token 目标。
2. **处理 B：**文字仍为 AR；晶体换成随机吸收态 MASK 去噪、双向块注意力和上文所述 AR-shift 对齐。

这样两臂的初始化与新增训练预算相同；差异是晶体建模目标/注意力/位置对齐这一整套迁移，不应把结果归因于单个组件。固定同一 128 验证集与生成配置，逐项保留失败分母，256 test 不动。预先将有用信号定为：B 严格有效生成至少 `64/128` 且比 A 多至少 `8/128`，Q-reference、E-reference、闭环均不低于 A；未达到即判该预算下不支持 warm-start 救援假设。此门槛是本试点的操作判据，不是文献结论或统计显著性保证。
