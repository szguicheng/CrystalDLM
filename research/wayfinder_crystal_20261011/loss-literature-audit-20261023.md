# 预训练 LM 继续训练的 loss 对照（2026-10-23）

## 结论

对因果语言模型 SFT，常见核心是对选中目标位置做 next-token 交叉熵；从预训练 checkpoint 继续训练不会自行改变这个定义。实质差别在**哪些位置受监督、每个位置的上下文、token/片段/任务怎样归一化和加权**。训练库负责数据管线、优化循环与保存；相同 logits、labels、mask 和 reduction 下，自定义 CE 与库的 CE 可以是同一目标。此结论仅针对下列因果 SFT 路径，不代表所有 LM 都用同一 loss。

## 一手来源

1. **PLaID 官方 `main`**：[`llm_finetune.py`](https://github.com/andaero/PLaID/blob/main/llm_finetune.py) 的 `tokenize` 令 `input_ids = labels = tokens.input_ids[0]`，collator 只把 padding label 改成 `-100`，然后由 Transformers `Trainer` 训练。因此 prompt 也参与 loss；这直接说明 prompt 是否 mask 是目标设计选择，标准因果 LM SFT 并非一律只训练 assistant。

2. **Hugging Face TRL / Transformers**：TRL 文档把 SFT 定义为 token-level next-token CE；prompt-completion 数据默认可只算 completion，language-modeling 数据则可算全序列，assistant-only 也由标签 mask 控制。[TRL loss 与 label mask](https://huggingface.co/docs/trl/main/en/sft_trainer#computing-the-loss) · [completion-only](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completion-only)。Transformers 的 `ForCausalLMLoss` 将当前位置 logits 与下一位置 label 对齐，以 `-100` 忽略位置，并对有效位置取 CE 均值。[官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L30-L67)。这说明“标准 loss”也取决于数据格式和 mask；不能把 PLaID 的整序列监督说成所有 SFT 的默认行为。

3. **FLARE 论文与官方 HybridDiffusion**：论文 §3.2 定义 clean AR 与两种互补 noisy view 的 masked-denoising 对数似然相加；互补 mask 把块内目标分成不相交两组，因此每个被训练 token 各有一个 AR 项和一个 diffusion 项，系数均为 1。[FLARE §3.2](https://arxiv.org/html/2606.01774v2#S3.SS2)。论文 §4 说明 prompt 只作上下文，loss 作用于 assistant response。[§4 训练设置](https://arxiv.org/html/2606.01774v2#S4)。在官方仓库固定版本 [`HybridDiffusion@6ca547a`](https://github.com/yuchen-zhu-zyc/HybridDiffusion/tree/6ca547aebb72bfe897e80e0a683c776789e5f38c)，训练配置启用 `complementary_mask`、`sft_label_mode="response"`、`causal_x0=true`，设 `ar_loss_weight=1`、`loss_auto_balance=false`、`loss_normalize=false`；模型以 masked-target CE 与 clean AR CE 构成等系数和。[配置](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml#L80-L96) · [互补 mask 与 loss](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/models/qwen3_5/model/dllm_model.py#L614-L636) · [CE 与组合](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/models/qwen3_5/model/dllm_model.py#L693-L748)。

## 对 CrystalDLM 权重的判断

本地 FLARE 式目标是

\[
L_{G} = \frac{\sum_{t\in response}\ell^{AR}_t + \sum_{t\in crystal}\ell^{diff}_t}{N_{response}},
\]

其中互补视图让每个 crystal target 恰好进入一次去噪 CE；文字 lead 只进入 AR 项。[实现](../../experiments/flare_crystal_qwen3_8b_20261019/model.py#L50-L83)。因此这是 FLARE 式晶体段迁移；论文对 assistant response 全段做双流监督，本地 lead 不做去噪。它与 FLARE 的 CE 原语及 `AR + diffusion` 单位系数相符，说明有方法先例；不验证本地代码执行正确，也不证明晶体数据上的效果。

repair 训练再按任务均值组合 `0.5 R_joint + 0.25 G_joint + 0.25 Q_AR`。[实现](../../experiments/crystal_repair_training_20261021/model.py#L15-L33) 这在数学上合理，表示把一半任务级权重给 repair、其余各四分之一给生成和读回；它不是全 batch 的 token 均值。上述三组来源没有给这个**具体任务配比**提供证据。FLARE 的单位权重是 AR/denoise 的 token 级关系，不能推出本项目的任务级配比。要声称该比例有效，需要同预算任务权重消融并分别看 repair、G、Q 指标。

本地早期分段 AR 与后期整体 G-AR 定义不同，但已分别与对应的 HF 加权 token-CE 参考式做数值对照，loss 与 gradient 一致。也就是说，各自的自定义实现都匹配各自对应的 HF 加权参考目标；早期段均值和后期整段均值仍是不同的权重定义。

合成数值核对（非模型实验）：若 R/G/Q 的 token loss 分别是 `[2,2]`、`[4,4,4,4]`、八个 `1`，先各自取均值再按本项目权重组合为 `2.25`；把 14 个 token 直接合并求均值则为 `2.00`。因此任务权重确实改变目标。互补例 `M={1,3}`、`Mᶜ={2,4}` 覆盖 4 个目标各一次，验证互补求和没有重复或遗漏。

本记录是官方文本与源码的静态对照；早期/后期 AR 的 loss-gradient 等价已做数值对照。未运行官方或本地训练；该对照不验证训练路径执行正确性，也不验证 `0.5/0.25/0.25` 的学习效果。
