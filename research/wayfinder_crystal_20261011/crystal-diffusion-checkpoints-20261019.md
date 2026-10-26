# 晶体表示与 diffusion 公开检查点调查

核查日期：2026-10-19。只核查论文、作者代码、官方模型卡及文件元数据；没有下载大权重、安装环境或运行推理。公开文件存在不等于已验证可在本项目恢复，更不等于可以直接加载进 Qwen3-8B。

## 核心候选

| 方法 | 表示及生成方式 | 网络／预训练来源 | 生成器 checkpoint 核查 |
|---|---|---|---|
| [WyckoffDiff，ICML 2025](https://proceedings.mlr.press/v267/ekstrom-kelvinius25a.html) | 在离散 Wyckoff 占位／原型结构上做 D3PM；不直接输出完整坐标 | 自建 WyckoffGNN，在 WBM 等晶体数据上训练；未见使用语言模型权重 | **未找到官方生成器下载资产。** [代码](https://github.com/httk/wyckoffdiff)里的 `evaluation_checkpoint/checkpoint.pth` 是 Wrenformer 评价器，不能当生成器。 |
| [WyFormer，ICML 2025](https://github.com/SymmetryAdvantage/WyckoffTransformer) | **AR** 生成无坐标的 Wyckoff 表示；下游可以接 DiffCSP++ 恢复几何 | 晶体专用 Transformer，MP20／Alex-MP20 训练；不是预训练 LLM 改造 | **有 AR 权重，不是 diffusion 权重。** [官方 HF](https://huggingface.co/SymmetryAdvantage/WyFormer-Alex-MP20/tree/main)有 `best_model_params.pt`、配置和表示处理文件。 |
| [DiffCSP++，ICLR 2024](https://github.com/jiaor17/DiffCSP-PP) | 在空间群／Wyckoff 约束下扩散晶格和分数坐标；另有含元素生成的配置 | 晶体专用 CSPNet/GNN，继承 DiffCSP 方法与代码，非 Qwen/Llama 权重；提供 MP20 等数据集模型 | **有官方预训练下载入口。** [Drive](https://drive.google.com/drive/folders/1FQ_b6CE09KtyGaU_r6uO8_I5JhrQmUFB)目录响应包含 `mp_csp`、`mp_gen` 等条目。WyFormer 链接的修改版另见下述来源边界。 |
| [ADiT，ICML 2025](https://github.com/facebookresearch/all-atom-diffusion-transformer) | VAE 将全原子结构变成连续潜在表示，DiT 以 latent diffusion／flow matching 生成；VAE 解码回晶体 | 先训练领域 VAE，再冻结它训练自建 DiT；公开模型联合学习 QM9＋MP20，非 LLM 初始化 | **有配套权重。** [HF](https://huggingface.co/chaitjo/all-atom-diffusion-transformer/tree/main)中的 `ldm.ckpt` 为 2,484,501,443 bytes，`vae.ckpt` 为 613,436,560 bytes。官方注明为作者独立复现，不保证与论文最终数字对应。 |
| [CrystalDiT，AAAI 2026](https://github.com/hanyi2021/CrystalDiT) | 对原子特征、坐标和晶格进行联合扩散，非 Wyckoff 文本扩散 | 自建 DiT；公开推理配置为 hidden=512、18 层、8 heads；晶体数据训练 | **有。** [HF `best_model.pt`](https://huggingface.co/xiaohan-yi/CrystalDiT/blob/main/best_model.pt)，文件元数据为 346,184,101 bytes。不能误用部分旧说明中的不同文件名。 |
| [Crystalite，2026](https://github.com/joshrosie/crystalite) | 原子化学描述符压缩成 PCA-16 连续 token，联合扩散原子类型、坐标和晶格；EDM | 自建几何增强 Transformer；公开 DNG 模型为 MP20 训练，14 层、hidden=512、16 heads；不是文本 tokenizer | **有。** [HF `best.pt`](https://huggingface.co/datasets/joshrosie/crystalite-datasets/blob/main/best.pt)，541,206,094 bytes；另有 `csp_mp20_best.pt` 和 `csp_mpts52_best.pt`。代码标注早期版本，尚未本地验证。 |
| [MatterGen，Nature 2025](https://github.com/microsoft/mattergen) | 联合生成元素、分数坐标和晶格；几何 diffusion，非文本 tokenizer | GemNet 系去噪网络；`mattergen_base` 在 Alex-MP-20 训练，性质模型再从这个晶体基座微调；另有 `mp_20_base` | **有。** [HF 基座权重目录](https://huggingface.co/microsoft/mattergen/tree/main/checkpoints/mattergen_base/checkpoints)有 `last.ckpt`，461,370,350 bytes；另有空间群／性质条件权重。官方说明公开权重是代码重训版，不与论文权重完全相同。 |
| [UniGenX，2025/2026](https://github.com/UniGenX-Team/UniGenX) | 符号 token 用 AR，连续坐标用 diffusion head，同一模型处理；晶体包含晶格及分数坐标 | Llama 风格自建 100M／400M 科学模型，先领域预训再任务微调；单独还有 NatureLM 约 1B 的自然语言实验，不能混作所有权重的初始化来源 | **有。** [HF](https://huggingface.co/unigenx/UniGenX/tree/main)列出 `csp_pretrain.pt`（1.89GB）、`unified_pretrain.pt`（678MB）、`csp_mp20.pt`（5.66GB）等。CSP 预训练文件本身不是开箱即用的评测检查点。 |

“预训练来源”区分网络架构、训练数据和加载的权重。采用 Transformer、LlamaConfig 或 GemNet 架构，不等于继承同名预训练模型参数。上述领域模型也不具有天然可供自然语言 LLM 使用的同一词表、隐层维度和生成接口。

## 其他查到但未确认生成器权重的工作

- [SymmCD，ICLR 2025](https://github.com/sibasmarak/SymmCD)：对不对称单元坐标／晶格做连续扩散，对元素／位点对称性做离散扩散；去噪器为基于 DiffCSP 的消息传递 GNN。官方 README 没有生成器下载链接，Releases API 返回空；库内 `prop_models` 的两个 checkpoint 是性质评价模型。已有[请求生成器权重的 issue](https://github.com/sibasmarak/SymmCD/issues/11)，本次未找到下载资产。结论仅为“未找到”，不等于证明作者从未对外分享。
- [DynaCrys，2026](https://arxiv.org/abs/2608.07401)：空间群、Wyckoff 占位和元素的符号扩散，再接几何扩散；使用由对称性描述符预训练的冻结 codebook。未核实到官方代码或生成器 checkpoint 下载文件。见 [token diffusion 证据笔记](crystal-token-checkpoint-evidence-20261019.md)。论文中描述了预训练 codebook，不等于公开了完整生成器。

## 与当前项目的关系

- 保留 Wyckoff 路线时，WyckoffDiff 的表示／离散扩散方法和 DiffCSP++ 的几何恢复阶段最相关；WyFormer 本身是 AR。
- 若寻找已训练好的结构编码器／解码器及 latent generator，ADiT 提供了具体的配套文件；它的 latent 是连续的，并非 PLaID 文本 token。
- 若研究共享 AR＋diffusion 机制，UniGenX 比“只把独立生成器接到 LLM”更接近该问题，但它主要扩散连续几何，仍需单独设计自然语言和晶体交错的任务接口。
- MatterGen、CrystalDiT、Crystalite 可以作为已有几何生成器候选。直接把它们作为外部模块调用，不等于已经完成共享核心参数的架构目标。

以上为基于接口与表示的研究判断，不是已测得的性能排序。本次未改变项目架构、tokenizer 或实验。

## 证据与来源边界

- [公开文件元数据](crystal-diffusion-checkpoint-assets-20261019.json)：HF API 文件路径、字节数及可用 LFS 元数据；SymmCD Releases 查询。
- [Wyckoff 专项核查](wyckoff-checkpoint-evidence-20261019.md)：WyFormer 链接的 `kazeevn/DiffCSPNew` 现重定向到 GeoCSP，包含 `data/mp-20/test_ckpt.pt`；不能确认该当前文件就是历史 ICML 实验的相同权重。
- [UniGenX 专项核查](unigenx-checkpoint-evidence-20261019.md)：权重用途、文件访问、预训练来源及精确词表配对。
- [ADiT 论文](https://arxiv.org/html/2503.03965v2)、[模型卡](https://huggingface.co/chaitjo/all-atom-diffusion-transformer)：两阶段 latent 训练和作者复现权重声明。
- [Crystalite 论文](https://arxiv.org/html/2604.02270v1)：PCA 化学原子 token 与 EDM；公开 checkpoint 细节来自作者仓库。
- [MatterGen 模型卡](https://huggingface.co/microsoft/mattergen)：GemNet 架构；数据与条件基座细节来自官方仓库。

没有执行 pickle 加载、权重 checksum 下载验证或生成，因此本表的“有”指公开资产／官方入口已核查，不代表推理复现通过。
