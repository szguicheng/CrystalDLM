# Group–subgroup / Wyckoff-splitting 生成先例筛查

## 候选逻辑与筛查边界

候选设想以高对称母相/原型为条件，生成群–子群选择、标准 Wyckoff orbit splitting，以及由对称性降低释放的位移自由参数；目标可以是条件结构变化或降对称生成。本次只筛查是否已有模型把这些对象直接作为 token、扩散状态或层次动作，最多列四项相关原始论文/官方实现。群–子群关系、Bärnighausen 树和 WYCKSPLIT 本身是既有晶体学基础，不属于候选可声称的新概念。

## 1. DynaCrys：直接的 group–subgroup 结构化扩散先例

### 来源与实际阅读范围

- 本地全文：`/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/work/papers/dynacrys-2608.07401.txt`；`source_manifest.json` 记录 arXiv v1、26 页、文本页标记完整。
- 官方来源：[arXiv:2608.07401](https://arxiv.org/abs/2608.07401)。
- 实际阅读：全文第 1 页摘要/引言，第 2–6 页 §2–§4.4（空间群–Wyckoff orbit、group–subgroup、symbolic diffusion、合法采样和几何 diffusion），并核对了 Figure 2 的具体 splitting 例子以及 Eq. (1)–(10)。未把未阅读的附录实验细节当作证据。

### 精确重合

DynaCrys 明确把空间群 `G` 放进扩散状态，使 `G` 在反向过程中与 Wyckoff occupations 和元素共同演化，而不是预先固定。§3（PDF pp. 3–4）说明 group–subgroup 关系改变可用的 Wyckoff vocabulary、multiplicity 和自由坐标参数化；限制到子群时，一个母群 orbit 可以裂成多个独立参数化的子群 orbit。Figure 2(b) 给出具体 `Pm → P1` 例子：`Pm 1b` 变成 `P1 1a`，而 `Pm 2c` split 成两个 `P1 1a`，去掉 mirror constraint 后三个代表坐标可以独立移动。

§4.1–§4.3（PDF pp. 4–6）定义 protostructure `P=(G,K,{(Z_i,r_i)})`，其中每个 occupation 是当前空间群的 Wyckoff row 和元素；symbolic diffusion 对 `G`、orbit slots、row 和 element 联合去噪。空间群前向 kernel `R_t` 在 maximal translationengleiche/klassengleiche group–subgroup 图上扩散，并明确称其结构来自 Bärnighausen trees。row vocabulary 随当前 `G_t` 改变，shared codebook 为不同空间群的 row 提供对齐。

§4.4（PDF p. 6）再以最终 protostructure 为条件生成 lattice 和每个 occupied orbit 的自由 fractional coordinates，投影到由 `P` 指定的允许 DoF。故它同时覆盖候选中的：

- 群–子群选择作为显式 symbolic diffusion 状态/转移图；
- space-group change 后 row vocabulary 的重新解释；
- orbit splitting 后独立 occupation 和自由坐标；
- symbolic symmetry 阶段到连续 geometry 阶段的层次接口。

### 与候选的差异

DynaCrys 并未把“母 WP → 子群 WP 列表”做成一个独立的可逆 splitting token，也没有以指定母相原型为条件执行单向降对称编辑；它在 group–subgroup 图上对 `G` 做双向结构化 corruption，并联合重采样当前合法 protostructure。候选若只说“使用群–子群关系指导生成”，已被 DynaCrys 直接覆盖；可辩护差异只能是母相条件、显式 split action/branch identity、或专门的位移模式 parameterization，并须与 DynaCrys 的动态 `G` diffusion 对比。

## 2. LEGO-xtal：subgroup symmetry augmentation + 生成模型

### 来源与实际阅读范围

- 原始论文全文：[AI-assisted rapid crystal structure generation towards a target local environment](https://www.nature.com/articles/s41524-025-01931-9)，npj Computational Materials（2025）。
- 官方代码：[MaterSim/LEGO-xtal](https://github.com/MaterSim/LEGO-xtal)。
- 实际阅读：官方 HTML 的 §§Crystal representations / Challenges / Solutions / LEGO-xtal framework（lines 86–157）、Discussion（lines 241–251）、Methods 的 representation transformation（lines 252–258）以及 code/data availability（lines 284–290）。

### 精确重合

论文 lines 130–133 以 diamond 为例：高对称 `Fd-3m` 的 `8a` 可通过 subgroup symmetry 表示成较低对称 `F-43m` 的两个 WP `4a+4d`，并指出这种 Wyckoff split 在相变/结构转变中常见。作者调用 PyXtal 的 subgroup symmetry module 生成丰富的 subgroup representations，作为数据增强，使 VAE/GAN 能学习并生成与训练结构 closely related 的 subgroup alternatives。lines 152–156 说明对 140 个碳 allotropes 用该模块产生有效 subgroup representations，再把 space-group number、Wyckoff site indices 作为离散特征，将 cell parameters/Wyckoff position values 作为连续特征训练生成模型。Discussion lines 241–245 将“subgroup symmetry augmentation”列为框架核心，并称 supplementary §IV-C 提供通用 subgroup 搜索接口。

### 与候选的差异

这是很强的“母结构 → 子群表示/标准 WP split → 生成”先例，但 split 主要是**预生成数据增强**，不是模型反向过程中的显式动作 token；论文使用 VAE/GAN（lines 176–180、252–268），没有 conditional diffusion，也没有报告逐步生成 subgroup choice、split branch 或 symmetry-lowering displacement mode。它直接削弱“首次用 Wyckoff splitting 扩展生成空间”的说法，但仍留出“把 split 作为可学习条件动作并联合生成释放的 DoF”的空间。

## 3. Bilbao Crystallographic Server WYCKSPLIT：公开的精确 orbit-splitting 表/程序

### 来源与阅读范围

- 官方工具：[WYCKSPLIT – Wyckoff Positions Splitting](https://cryst.ehu.eu/cryst/wpsplit.html)。
- 官方服务器总览：[Bilbao Crystallographic Server](https://cryst.ehu.es/)，其中将 WYCKSPLIT、SUBGROUPGRAPH、MAXSUB、GROUP–SUBGROUP RELATIONS 等列为正式工具。
- 阅读了工具说明和输入/输出要求；没有把它当作机器学习论文，也没有运行外部服务。

### 精确能力

WYCKSPLIT 的官方说明直接定义任务：给定 supergroup `G`、subgroup `H` 及连接两者基底的 transformation matrix，计算 Wyckoff position 在 `G → H` 对称性降低时如何 splitting。官方相关页面还说明结构关系由 occupied orbits 的 relations 决定，且同一 WP 中所有 orbits 遵循相同 splitting 行为。服务器同时提供 subgroup graph、maximal subgroup、Wyckoff positions 和 coordinate transformations，因此可以作为候选训练/采样器的精确 orbit-splitting oracle 或离线表来源。

### 与候选的差异

它是确定性晶体学工具和数据源，不是 tokenized generator、diffusion model 或学习到的母相条件编辑器。故“有公开 splitting 表/代码”这一基础能力已经存在；新颖性不能放在重新定义 splitting 规则，而应放在如何将其编码为动作、如何采样合法分支，以及如何对释放的自由参数做条件生成。

## 4. WyckoffDiff-Adapter：条件扩散相关，但不是 subgroup-action 生成

### 来源与实际阅读范围

- 原始基础模型：[WyckoffDiff, arXiv:2502.06485](https://arxiv.org/abs/2502.06485)；官方实现：[httk/wyckoffdiff](https://github.com/httk/wyckoffdiff)。本地全文已在本项目 source manifest 中，但本次只核对其已有 symmetry-based discrete diffusion 表示，不重复全文分析。
- 条件扩展官方/论文页面：[Symmetry-aware conditional generation of crystal structures using diffusion models](https://www.sciencedirect.com/science/article/abs/pii/S0927025626003083)。本次使用搜索结果和页面摘要/方法片段作边界核对，未将其视为已完整阅读全文。

### 已核对到的事实

该扩展以 WyckoffDiff 的 protostructure 为条件生成基础：输入 space group 和 element-by-Wyckoff occupancy，离散 diffusion 输出 `z^∞`（元素×一般位置计数）与 `z^0`（固定位置元素/空位），再用 PyXtal 实例化。页面明确说输入 space group 在 protostructure 阶段保持不变；生成后 relaxation 可能出现向 P1 或其他低对称群的下降，这被解释为后处理 relaxation 的 symmetry descent，而不是 symbolic diffusion 中的 group–subgroup action。

### 与候选的差异

WyckoffDiff-Adapter 是 symmetry-conditioned diffusion 的相关先例，但不是母群选择子群、显式 Wyckoff splitting 或 Bärnighausen-tree transition 的先例。它可以作为候选的静态-WP diffusion baseline；不能用它支持“已有 subgroup-action diffusion”，也不能把 relaxation 后的 subgroup descent 当作模型生成了 subgroup token。

## 综合判断

在本次最多四项筛查中，直接重合强度排序为：

1. **DynaCrys：** 最直接，已把 group–subgroup 图/Bärnighausen 关系放进空间群 symbolic diffusion，并在当前 `G` 下联合更新 Wyckoff rows、元素和 orbit count；其 Figure 2 明确展示 WP splitting 和自由坐标释放。
2. **LEGO-xtal：** 已将 subgroup symmetry augmentation 和具体 `8a → 4a+4d` split 用作训练数据/生成空间扩展，但使用 VAE/GAN，不是 diffusion action token。
3. **WYCKSPLIT：** 公开且精确的 `G,H,transformation → WP splitting` 工具/表，是任何候选实现可直接调用的基础设施。
4. **WyckoffDiff-Adapter：** 条件 diffusion 先例，但 space group 固定；没有显式 subgroup/splitting state，relaxation descent 不等于生成动作。

因此不能声称“首次将 group–subgroup relations 用于晶体生成”“首次使用 Wyckoff splitting 做生成增强”或“首次在 symmetry-aware diffusion 中改变空间群”。较窄、可检验的差异可以是：给定母相/高对称原型，显式预测一条合法 subgroup/splitting action（而非 DynaCrys 的无向图 corruption 或 LEGO-xtal 的离线 augmentation），再条件生成由该 split 释放的独立位移模式和连续自由参数。该主张仍需与 DynaCrys、LEGO-xtal 和静态 WyckoffDiff 做动作级、合法性、条件结构恢复及 OOD subgroup 评估。

本筛查未训练、未运行生成实验；“未找到独立 token”仅限上述实际阅读范围，不能推广为全球文献不存在。
