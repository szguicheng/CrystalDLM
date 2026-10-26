# Design / Understanding 全量数据

2026-10-25 用户确认：两个任务单独准备；Understanding 只给 PLaID 晶体，仍回答化学式、空间群编号、晶胞原子数和独立 Wyckoff 轨道数；使用完整可用数据，取消 1/5 抽样。此目录只准备数据，没有启动训练。

## 输入和输出

| 任务 | 模型输入 | 监督输出 |
| --- | --- | --- |
| Design | 生成要求：空间群编号＋Ehull 上限 | 英文结构描述＋`<CRYSTAL_START>` PLaID `<CRYSTAL_END>` |
| Understanding | 固定的四字段提问＋`<CRYSTAL_START>` PLaID `<CRYSTAL_END>` | 化学式、空间群编号、晶胞原子数、独立轨道数 |

Understanding 没有历史对话、生成要求、英文描述或 Ehull 条件。化学式使用 PLaID 展开晶胞的组成，不改成约化化学式。四字段回答保持之前的英文句式，移除数据中手写的 `<|im_end|>`，由训练框架的模型模板添加结束标记。

Design 保留之前的要求构造规则：按源 Ehull 标签选择包含它的最小上限，候选上限为 0.08、0.1、0.2、0.3 eV/atom。源标签与结构均不重算。

## 数据量和划分

| 每个任务 | training | validation | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 记录数 | 36,362 | 4,501 | 4,602 | 45,465 |

两个任务逐行使用相同材料顺序与原始划分，但训练时独立选择任务。原始采集 45,629 条；之前的预处理已排除 164 条转换/一致性检查失败记录（train 129、val 16、test 19），不是本轮再次采样。

源事实：`research/crystal_sharing_20261012/prepared_001/{train,val,test}-facts.jsonl.gz`。该目录的 `manifest.json` 和 `base-errors.jsonl.gz` 保留原失败记录与哈希。原始采集数据也同步到服务器的 `research/mp_ehull030_20261011/collection_001/data_001/`。

## LLaMA-Factory 格式

采用官方 [Alpaca SFT 格式](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md#alpaca-format)，每行 JSON 对象仅有三个字符串字段：`instruction`、`input`、`output`。框架将 `instruction` 与 `input` 拼成用户消息。没有预先拼接 `User:`、`Assistant:` 或模型聊天控制符。

最终目录：

```text
data/
  dataset_info.json
  design/{training,validation,test}.jsonl
  understanding/{training,validation,test}.jsonl
  metadata/{training,validation,test}.jsonl
  manifest.json
  validation-report.json
```

`metadata` 按行记录 material_id、原划分、原结构哈希、精确标签和标准答案来源，不作为模型输入。注册名为 `crystal_design_training`、`crystal_design_validation`、`crystal_design_test` 和对应的 `crystal_understanding_*`。

训练时：`dataset_dir` 指向本目录下 `data`；`dataset` 选择一个任务的 `*_training`；`eval_dataset` 选择同任务 `*_validation`；测试集仅用于最终测试。不要再随机分割验证集，不设置 `max_samples`，不将两个任务混合。SFT 使用 `train_on_prompt: false`。模型模板与 tokenizer 在正式训练配置阶段固定；晶体边界标记必须保留，不能把 `<CRYSTAL_START>` 设为生成停止标记。

## 英文描述补齐与可复现性

- 复用旧 train/val 的 8,172 条已整理描述（含既有清理/修正）。
- 另外 37,293 条需要补描述：train 29,090、val 3,601、test 4,602。
- 沿用已有 `research/crystal_bulk_annotation_20261013/batch.py`，DeepSeek V4 Flash、thinking enabled、reasoning effort max、原英文提示词、8 并发。语料构造期间调用；训练时不调用。
- `annotation_inputs/` 只包含缺描述的材料。`annotation_run/annotation.sqlite3` 持久化进度并支持续跑；`successes.jsonl` 保留模型回执和事实哈希。已有 8,172 条不重复请求。
- 标注在已有凭据的本机运行；未将 API 密钥上传到共享服务器。完成后自动将回执和最终数据同步到 LiuGangGPU，并在服务器再次验证。
- 本地 tmux 会话：`crystal-full-data-20261025`。`finish.sh` 顺序完成补描述、导出、验证、同步及服务器复核；任一步失败会停止，日志在 `pipeline.log`，退出码在 `pipeline.exit`。

在本地项目根目录运行 `python3 experiments/design_understanding_full_20261025/annotate.py` 可续跑已有标注；运行 `prepare.py finalize` 需要全部 37,293 条成功。标注开始后不能重新运行 `prepare.py prepare`，避免更改已冻结输入。

## 验收

`validate.py` 检查全部记录：文件行数、注册映射、同材料对齐、材料 ID/原结构哈希无交叉、Understanding 输入仅含 PLaID、四字段答案、化学式与 Wyckoff 多重度/原子数/轨道数一致、Design 条件包含真实标签、无聊天结束符重复写入。

状态以 `data/manifest.json` 为准。Design 尚未补齐时，只注册已准备好的 Understanding，状态为 `understanding_ready_design_awaiting_captions`；两任务全部导出后为 `both_tasks_exported_validation_pending`，全量校验通过后才改为 `both_tasks_ready`，同时生成 `validation-report.json`。

Understanding 已通过服务器全量检查。现有 Qwen3-8B tokenizer（关闭 thinking）的聊天序列长度最大为 training 622、validation 613、test 614 tokens，均低于 2048。结果在 `data/token-length-audit.json`；这是 tokenizer 检查，没有安装或实际运行 LLaMA-Factory 的训练流程。

格式检查和已有结构一致性检查不等同于逐条人工审查全部 LLM 描述的科学准确性。原结构哈希互斥也不是跨所有晶胞表示的完整等价结构去重。
