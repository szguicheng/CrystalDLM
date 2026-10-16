# LiuGangGPU 上的 Qwen3-8B 纯 AR 实验准备

- 远端现有项目根目录：`/home/liugang/src/crystalDLM`。
- 环境：`envs/crystaldlm`，Python 3.11.17；创建命令见 `bootstrap_env.sh`。
- 原生 mamba 2.9.0 的环境登记开关未能核实，因此创建环境使用同套 Miniforge 的 `conda --solver libmamba`，并设置 `CONDA_REGISTER_ENVS=false`。执行后检查共享 `~/.conda/environments.txt` 中没有项目环境条目。后续包操作和训练仍由 `mamba run --prefix` 执行。
- `runtime.sh` 将包缓存、模型缓存、临时目录等放在项目内部；共享 base 环境不安装项目依赖。
- 固定模型 revision：`b968826d9c46dd6066d109eabc6255188de91218`。`download_model.py` 从可访问的 HF 镜像取得该 revision 的 Git/LFS 元数据；权重从 ModelScope 下载前比对 LFS SHA-256，下载后重新校验。模型目录内的 `download-manifest.json`、`VERIFIED.json` 记录结果。
- PyTorch/CUDA 包下载先使用官方 cu124 源；保留已完成 wheel 后从项目 wheelhouse 安装。公共 PyPI 下载缓慢，后续依赖改用阿里云镜像；最终版本由 `pip-freeze.txt` 记录，安装日志保留各次尝试。
- 最终依赖由官方 PyPI 解析为 `dependency-plan.json`，`offline_dependencies.py` 用 curl 并行下载 51 个 wheel，逐一核对计划中的 SHA-256 后离线安装，`pip check` 通过。训练代码中的精确版本断言亦在实际预检中通过。
- `check_scorer.py` 用新环境重新审计旧基线的 32 个原始结构，要求与原 7/32 及失败原因一致，避免评分环境变化混入模型比较。
- 实验入口见 `../../experiments/ehull_pure_ar_qwen3_8b_20261015/PROTOCOL.md`，训练参数和状态见对应运行目录。
- 依赖安装完成、代码检查通过后，`launch_experiment.sh` 顺序执行 CPU 测试、评分回归、两卡预检、两卡纯 AR 训练和独立单卡生成评估。阶段失败立即保留日志退出，训练检查点可用于显式恢复。
- 实现 review 保留共享模型原有的可训练 `time.weight`，在新实验配置和身份中显式记录。实验改变了基座、精度、微调方式和学习率，不作为纯模型大小的单因素比较。

## 排查记录

- 环境排查子智能体曾短暂在远端 `/tmp` 创建两个探针文件，随后已移除；后续探针与全部持久实验资产均限制在项目内。
- 环境子智能体在创建完成后的汇报阶段遭遇上游 429，主智能体已通过实际 Python/pip 版本、创建日志和环境登记检查核验创建结果。
- 训练实际启动、显存与结果只以运行日志及 JSON 回执为准；准备完成不表示已获得学习结果。
- 启动验收已完成：六项小模型测试、32 条旧结果评分回归、8B 两卡实际预检及重载均通过；正式训练已越过第 50 次更新并保存首个周期检查点。tmux 会话 `crystal-qwen3-ar`，socket `tmp/tmux.socket`，总日志 `logs/experiment.log`，实际训练与评估输出见实验目录。
