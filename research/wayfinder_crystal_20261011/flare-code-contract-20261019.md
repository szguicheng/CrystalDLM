# FLARE 代码契约核查（2026-10-19）

固定官方 commit `6ca547aebb72bfe897e80e0a683c776789e5f38c`。本记录为静态代码核查，不是对官方已发布模型性能的重新评价。

- [论文 §3.2](https://arxiv.org/html/2606.01774v2#S3.SS2)：clean token-causal AR loss；noisy block 内双向、仅见前置clean context；M/M补集各监督不同的masked target。
- [数据代码](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/hf_datasets/text_datasets.py#L704)：DLLM labels与input位置对齐。
- [Qwen3.5配置](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml#L86)：互补mask、logit_shift、fusedCE、causalclean、bidirectionalnoisy；注释N=4，block_size=3，不可把此数字直接解释为论文B=4。
- [Qwen3损失](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/models/qwen3/model/model_dllm.py#L807)与[Qwen3.5损失](https://github.com/yuchen-zhu-zyc/HybridDiffusion/blob/6ca547aebb72bfe897e80e0a683c776789e5f38c/torchtitan/models/qwen3_5/model/dllm_model.py#L764)：先shift label，再按未shift的mask过滤。若i被mask、i+1未mask且同块，i位置的loss目标是仍可见的i+1，存在答案可见风险。其他helper函数有先mask再shift的不同路径。尚未运行完整官方网络验证影响。

本项目采用显式目标遮盖后前位预测的迁移实现：每个4目标token块附前缀末token副本作anchor；clean保持因果；noise只看过去clean和本块。互补视图确保每个晶体token只得到一次去噪监督。独立多层微型Qwen测试已确认cleanstream与原causal输出一致，改变被遮盖答案不影响当前块噪声hidden，packed训练与逐块推理logits一致。

本地协议：[FLARE晶体小试验](../../experiments/flare_crystal_qwen3_8b_20261019/PROTOCOL.md)。这是按论文目标契约修正索引的FLARE式迁移，不声称逐行复现当前仓库，也不声称与原论文基座、数据量或解码实现相同。
