# Phase 3：AR-design

当前在 `main` 基准 `624f9f72` 上继续，使用原始 Qwen3-8B，未从 Understanding 或过拟合 checkpoint 继续正式训练。

## 数据与配置

- 完整固定划分：36,362 train / 4,501 validation / 4,602 test。原 `datasets/crystal_tasks_v1` 的内容和 manifest 保持不变。
- 本轮输入版本：`/home/liugang/src/crystalDLM/data_preparation/crystal_tasks_v1/exports/design_full_20261026`；manifest SHA-256 为 `360c68297db7e14e9db38815c68a6b18e2f38d61d17063133694c20d2f6e3969`。
- 本地与远程9个任务/metadata JSONL哈希一致；源ID、划分、性质标签和晶体对应关系全量校验通过。实际模板长度中位数371、最长799，cutoff2048无需截断。
- 三卡、每卡1条、累积4步，global batch12；BF16、LoRA rank32/alpha64、lr2e-4、seed42；正式训练固定3epoch。AR-design与Hybrid-design引用同一数据路径/哈希和训练配方，Hybrid仍等待后续END闸门。
- 输入为SG＋ehull上限，输出英文描述＋PLaID，均用AR。描述CE和晶体CE分别记录；每600updates及epoch边界评价保存，保留best/final。

## 过拟合检查

训练集中有542个不同请求。原seed42随机32条只有25个不同输入，不能要求贪心对同一输入复现不同参考输出；因此32条诊断按seed42选择不同请求，ID保存在run_meta，全部训练数据不作筛选。

- 首轮60个诊断epoch：30/32完整复现，32/32严格有效且SG正确，NLL0.0006213。两条返回同SG、另一ehull上限对应的训练样本；首分歧token检查显示条件区分仍不充分。该失败运行保存在 `phase3_design_20261026/`，全量训练被正确阻止。
- 第二轮从原始基座、相同32条和相同超参重新开始，仅把诊断长度改为120epoch：完整输出、晶体正文、严格有效且SG正确均为 **32/32**，NLL **3.56555e-5**。
- 第二轮正式训练与诊断的初始可训练参数SHA一致（`a5139c3415ff8bd26b6a91c606213be9653598cea9ed971977aeaa63d049bcca`）；正式训练仍是3epoch。

## 运行与评价

活动tmux：`crystal-design-phase3-run2`；目录：`experiments/crystal_tasks_v1/phase3_design_20261026_run2/`。检查时已进入全量训练更新，GPU0/1/2工作，第四张卡保留。

后台入口：[launch_design.sh](launch_design.sh)。后续顺序：完整validation/test参考答案NLL → 固定test条件生成 → GRACE → 共享design评价。若best与final权重不同，两版都运行完整评价。

生成计划是359个不同test请求 × 每请求10次 × 3个seed，共 **10,770次原始尝试/每checkpoint**，计划SHA `0cff01e36b0052900591aec10390ce01983a0462bb04ecb77f19fb248e9cf6cc`。解码固定temperature0.7/top-p0.8/top-k20，实际attempt seed由计划身份派生；原始失败计入预算，运行错误会停止worker，不伪装成模型输出失败。三个分片必须完整覆盖计划，GRACE回执按ID和输出/请求哈希关联。

共享指标包括格式、strict validity、SG、代理ehull、S.U.N.、SM-COV和每seed均值/标准差。原有GRACE模型、MP2020修正与MP2023凸包保持固定。只有这些评价完成后才能提交闸门3；目前尚无正式生成成绩。

## 验证

新路径的49项CPU测试、未定义变量检查、真实LLaMA-Factory prompt/label对齐均通过。额外的诊断epoch参数检查验证其不改变正式3epoch设置。32条实际gold结构全部通过既有严格审计。运行回执和逐条生成记录保留在各独立目录。

服务器激活脚本还设置了项目外的HF数据集/hub缓存；本轮已在`python.sh`中补齐激活后的项目内覆盖，并实查实际路径。未更改共享环境配置或清理他人的缓存。
