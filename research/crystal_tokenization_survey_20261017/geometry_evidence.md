# 4.3.5 Crystals：几何表示证据

范围：仅核对 CGCNN、CDVAE、DiffCSP 三项；依据原始论文与作者/官方代码。以下为证据摘录与对当前稿件三行分类的只读核对，不修改稿件。

## CGCNN

**准确引文**：Tian Xie and Jeffrey C. Grossman. “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.” *Physical Review Letters* 120(14), 145301 (2018). DOI: [10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301). [arXiv:1710.10324](https://arxiv.org/abs/1710.10324).

- **单位与几何**：晶体表示为周期多重图：节点是晶胞内原子，边表示原子及其周期像之间的邻接；节点/边带元素与键特征。边距离参与特征构造，但送入网络的是原子特征、邻居特征和邻居索引；晶格矩阵与分数坐标没有作为可逆字段保留。原文 Figure 1、晶体图构造段及补充材料 §I.1；官方实现从 CIF 建立 Pymatgen Structure、取周期邻居和距离，再返回图张量：[论文 HTML](https://arxiv.org/html/1710.10324v3)，[data.py，CIFData 与邻居特征](https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py)。
- **训练与边界**：图卷积后对原子特征做晶体级均值池化，再输出目标性质；这是性质预测器，没有单独的 encode/decode 或结构重建流程。原文 Figure 1；[model.py，forward 与 pooling](https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py)；[作者仓库 README](https://github.com/txie-93/cgcnn)。
- **空间群/重建**：没有显式空间群输入；论文和官方模型接口都以原子与邻居图为输入。任务是性质预测，不重建完整输入晶体。
- **不变性来源**：刚体平移/旋转不改变距离特征；原子重标号由图消息聚合及晶体级均值池化处理。补充材料 §I.1 用 NaCl/KCl 示例说明其晶胞选择不变性。它不是显式 E(3)-等变模型。
- **D 分类注意**：论文写明特征属性按类别 one-hot，连续属性先分成 10 个区间；但作者代码使用浮点原子特征向量和 GaussianDistance 展开。见论文补充材料 §I.1，以及 [data.py，浮点元素嵌入](https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py)、[GaussianDistance 与邻居距离展开](https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py)、[距离特征调用](https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py)。因此，若 D 按论文文字的“所有属性离散化”作字面判定，CGCNN 的 D 可能应为是；若 D 指完整晶体的离散 token/符号串，或依官方实现的浮点图张量判定，则为否。当前稿件要保留“CGCNN 全否”，建议把 D 的判据明确限定为结构 token 字母表，而不是笼统的“表示全离散”。

## CDVAE

**准确引文**：Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi Jaakkola. “Crystal Diffusion Variational Autoencoder for Periodic Material Generation.” *International Conference on Learning Representations (ICLR)* (2022). [arXiv:2110.06197](https://arxiv.org/abs/2110.06197); arXiv-issued DOI: [10.48550/arXiv.2110.06197](https://doi.org/10.48550/arXiv.2110.06197).

- **单位与几何**：原文 §3.1 将完整晶体写作 M=(A,X,L)：元素类型、原子坐标、周期晶格；用周期多重图表达跨晶胞相互作用。编码器输出连续随机潜变量 z∈R^D；晶格和坐标不是 z 中可直接读取的显式字段。[原文 §3.1、§4，PDF pp. 3–5](https://arxiv.org/pdf/2110.06197)。
- **训练与边界**：VAE 编码器产生 μ、log variance 并重参数化得到 z；由 z 预测原子数、晶格参数和组成；周期图去噪解码器通过 Langevin dynamics 生成/重建坐标与元素类型。三部分连同 KL 与去噪目标联合训练。[作者代码 model.py：encode](https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/model.py)、[decode_stats](https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/model.py)、[langevin_dynamics](https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/model.py)、[联合训练 forward](https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/model.py)。
- **空间群/重建**：未把空间群作为显式条件或结构字段。论文明确报告输入晶体重建、无条件生成和性质优化；因此是完整晶体的近似重建型连续自编码器，不能称作离散 tokenizer。实现的 recon 路径对原始晶体 encode 后再解码；原子数和元素类型可由可选开关强制使用真值，故比较重建结果时要报告该设置。[evaluate.py，reconstruction 与可选强制真值](https://github.com/txie-93/cdvae/blob/main/scripts/evaluate.py)、[命令行开关](https://github.com/txie-93/cdvae/blob/main/scripts/evaluate.py)。
- **不变性来源**：周期多重图编码周期边界；作者将 SE(3) 不变图网络用于编码器、SE(3) 等变 PGNN 用于坐标去噪解码器，并报告原子置换、平移、旋转和周期不变性。应把这些归于表示构图与模型的组合，而不只归于“invariant graph encoder”。原文 §4，PDF pp. 4–5。

## DiffCSP

**准确引文**：Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, and Yang Liu. “Crystal Structure Prediction by Joint Equivariant Diffusion.” *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, Main Conference Track. DOI: [10.52202/075280-0767](https://doi.org/10.52202/075280-0767). [arXiv:2309.04475](https://arxiv.org/abs/2309.04475).

- **单位与几何**：晶体状态由元素/组成 A、分数坐标 F、晶格 L 表示；任务定义为给定化学组成预测 p(L,F|A)。L 与每个原子的 F 都是显式几何变量，因此 G=是。原文 §3、§4.1–4.2，PDF pp. 3–5；[NeurIPS 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract.html)。
- **训练与边界**：直接对 lattice 和 fractional coordinates 加噪，周期 E(3) 去噪网络学习 score；采样从随机 lattice/坐标开始执行反向扩散。没有晶体压缩 encoder 或 latent-token decoder。代码中的 self.decoder 是去噪网络，不表示 encode→decode tokenizer 边界。[官方 diffusion.py：训练输入、扰动与 score loss](https://github.com/jiaor17/DiffCSP/blob/main/diffcsp/pl_modules/diffusion.py)、[随机初始化与反向采样](https://github.com/jiaor17/DiffCSP/blob/main/diffcsp/pl_modules/diffusion.py)。
- **空间群/重建**：论文形式化的对称性是置换、O(3) 与周期平移，没有显式空间群标签或 Wyckoff 约束；实验是稳定结构预测与从头生成，不是把输入晶体编码后重建。原文 §4.1–4.3、§5.1、§5.4；官方 [DiffCSP README](https://github.com/jiaor17/DiffCSP)。
- **不变性来源**：周期性通过分数坐标及取模表示；lattice 的 O(3) 变换和坐标的周期平移由联合扩散/周期等变去噪模型处理；原子置换由图网络处理。不是由离散 tokenizer 消除几何自由度。

## 对当前稿件三行的只读核对

按稿件当前定义（N 只计显式神经 encoder latent，G 计显式保留晶格与充分位置变量，S 计显式空间群/Wyckoff）：

- **CGCNN：N=否、G=否、S=否**有原文/接口支持；D=否取决于 D 是否限定为完整结构 token 字母表。按“所有属性是否离散”字面理解，原论文的分箱描述会产生 D=是的解释；官方代码又使用浮点图特征。需收窄 D 定义来稳定“全否”结论。
- **CDVAE：N=是、G=否、S=否**成立。它有显式连续 VAE latent 和完整结构重建任务；不是全离散表示。表中“不变性来自 invariant graph encoder”只覆盖编码端，若该列描述整个系统，应补上周期多重图与等变去噪解码器。
- **DiffCSP：N=否、G=是、S=否**成立。它是直接在显式晶格/分数坐标上训练和采样的生成模型，不是 tokenizer；空间群没有显式约束字段。
