# 交错 AR 与晶体扩散实施计划

本计划在当前聊天内逐项执行，使用独立实验目录，不创建新的聊天或复制旧模型结果。本工作区没有 Git 仓库，保留文件、配置哈希和运行收据作为版本证据。

**目标：** 实施 PROTOCOL.md 的数据、评分、训练与采样闭环，并先运行8材料机制检查。

**架构：** `score.py` 负责结构与事实核验；`prepare.py` 冻结材料、编辑目标与任务；`model.py` 负责注意力、损失和反向采样；`run.py` 训练和评估。共用一个 `test_checks.py` 检查关键行为。

**技术：** stdlib、现有PLaID codec、PyXtal、pymatgen、PyTorch、Transformers。规范见同目录 PROTOCOL.md。

## 约束

- PLaID格式，晶体块256token，时间步32，完整样本最多1536token。
- 所有失败留证；不提供真实生成长度；不从参考结构补齐输出。
- 计算节点执行训练，不在HPC登录节点计算；先测吞吐量。
- 阶段0最多200更新，阶段1每模型800更新；阶段0未通过不进入阶段1。
- 原有实验环境和输出保持；本目录建立专用环境和新输出目录。

## 1 数据事实和评分器

- [x] 在 `test_checks.py` 用真实Ti2O4文本构造正确结构、矛盾头部、错误位点、错误编辑和同义回答；先运行并确认新评分功能缺失。
- [x] 实现 `score.audit(text, condition=None)`、`score.edit(text, scale)`、`score.answer(text)`、`score.grade_answer(response, text)` 和 `score.grade_edit(output, source, scale)`。
- [x] 运行 `rtk proxy .venv/bin/python test_checks.py`，验证正确样例通过、错误样例失败。

## 2 冻结数据

- [x] 实现 `prepare.py` 读取官方CSV，按ID哈希选择、执行codec资格审计、按结构去重、检查token长度，写JSONL和manifest。
- [x] 先构建8材料数据，审计每个方向的编辑目标；然后构建协议规定的train/val/test队列。
- [x] 独立读回JSONL，验证材料和任务划分、源SHA256、长度、编辑目标和评分器。

## 3 混合模型

- [x] 在同一检查文件增加真实小型Transformer的注意力无未来泄漏检查、AR shift与diffusion同位置标签检查、吸收态反向采样与边界检查，确认失败后实施。
- [x] 实现共享预训练模型、自定义块mask、时间embedding、固定容量噪声和32步反向采样。
- [x] 验证全MASK输入不读取END位置；完成后重算干净上下文；两模型同种子初始化。

## 4 阶段0执行

- [x] 实现 `run.py` 的匹配任务顺序、训练日志/checkpoint和独立G/Q/E/闭环评估。
- [x] 验证资源连接和计算节点CUDA；固定模型revision与依赖版本，测吞吐量。
- [x] 在同一8材料队列成对训练最多200更新，保存原始输出并按协议检查门槛。
- [x] 未过门槛写失败定位；通过才冻结正式预算并启动阶段1。所有完成项以实际收据为准。
