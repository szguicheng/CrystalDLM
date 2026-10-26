# CrystalDLM 项目约定

本文件适用于本仓库及其子目录，补充全局 `AGENTS.md`。

## 版本控制

- 后续工作默认直接在 `main` 分支按顺序修改；仅在用户明确要求时创建其他分支或 worktree。
- 以已合并的最新基准继续工作，保留历史实验、冻结数据和运行回执。

## 研究与实验

- 项目术语和表示边界见 `CONTEXT.md`。讨论或修改表示、条件、训练目标和评价时先阅读该文件。
- 当前任务与实验协议见 `.scratch/unified-crystal-ar-diffusion/map.md` 及其关联 tickets；实验状态以对应运行记录为准。
- 主线固定使用 PLaID 类 Wyckoff 晶体表示：文字使用 AR，晶体使用 diffusion，共享核心模型参数并在同一上下文中交错生成。新 tokenizer 方向暂时搁置。
- 首轮性质目标为 ehull，文字使用简短、可核对的英文描述，只表达结构事实和已有标签能支持的信息。
- 用户请求应包含结构特征或其他有标签支持的设计要求，ehull 是另行给定的性质条件；不能把仅含 ehull 的请求当作完整任务。2026-10-13/15 已跑的两轮属于 ehull-only 简化试验，保留作历史对照；结构性输入缺失需先修正训练和评估样本契约。
- 实验顺序为数据准备、匹配的纯 AR 基线、基线分析、共享 AR＋diffusion 联合训练。比较两种训练方式时保持基座、数据划分、微调设置和评估集匹配。
- 沿用已固定的数据子集、随机种子和 manifest；修改数据配方或划分时保留原版本并记录差异。
- 分开报告工程链路、晶体结构有效性、文字与结构一致性、生成结构的性质评分。输入的目标 ehull 和生成文字不能代替性质评价。

## 训练服务器与文件范围

- 后续训练默认使用 `ssh LiuGangGPU`；Burgundy 作为备用。
- LiuGangGPU 上使用已有项目目录 `/home/liugang/src/crystalDLM`（注意大小写），不另建 `CrystalDLM` 目录。
- 用户于 2026-10-21 授权将整个项目迁至 `/data1/users/liugang/crystalDLM`；原路径 `/home/liugang/src/crystalDLM` 使用符号链接指向该目录。继续从原路径启动任务，环境前缀和脚本调用路径保持不变。
- 代码、数据、模型权重、检查点、日志、独立环境、缓存及临时文件均保存在该项目内，实际落盘到 `/data1`。
- 文件修改范围为上述工作区与其数据盘目标；保留其他人的目录、环境和进程。
- 大文件传输、模型下载和保存检查点前，使用 `findmnt -T /home/liugang/src/crystalDLM` 与 `df -h /home/liugang/src/crystalDLM` 确认目标确实在 `/data1` 且空间足够；挂载缺失时先恢复挂载，不在系统盘创建替代目录。

## GPU 使用

- 根据任务需要使用 1–3 张 GPU；独立训练或评估可并行安排，无需为使用第三张卡另行确认。我们所有任务合计同时占用的 GPU 不得超过 3 张。
- 每次启动前使用 `nvidia-smi` 检查占用，并检查已有项目任务，选择空闲卡，避免重复启动。
- 显式设置 `CUDA_VISIBLE_DEVICES`，分布式进程数与选用的 GPU 数量一致。
- 不终止或调整其他人的任务，不为腾显存清理他人的进程。

## 独立环境与缓存

- 使用 `/home/liugang/software/miniforge3/condabin/mamba` 管理环境。
- 项目环境前缀固定为 `/home/liugang/src/crystalDLM/envs/crystaldlm`，初始使用 Python 3.11；安装训练依赖时记录版本。
- 已有项目环境直接复用。首次创建为避免修改共享环境登记列表，使用同套 Miniforge 的 `conda --solver libmamba` 并设置 `CONDA_REGISTER_ENVS=false`；准备命令和依赖版本见 `research/liugang_setup_20261015/README.md`。
- 使用 `mamba run --prefix /home/liugang/src/crystalDLM/envs/crystaldlm <命令>` 执行项目程序。
- 保持共享 base 环境、全局配置和其他人的环境不变；创建环境及安装依赖前，显式将工具目录设置到项目内部：

| 设置 | 相对于远程项目根目录的路径 |
| --- | --- |
| `MAMBA_ROOT_PREFIX` | `.mamba` |
| `CONDA_ENVS_PATH` | `envs` |
| `CONDA_PKGS_DIRS` | `.cache/mamba/pkgs` |
| `PIP_CACHE_DIR` | `.cache/pip` |
| `HF_HOME` | `.cache/huggingface` |
| `TORCH_HOME` | `.cache/torch` |
| `XDG_CACHE_HOME` | `.cache` |
| `TMPDIR` | `tmp` |

- 上表中的设置均使用绝对路径，并在使用前创建相应目录；环境中的包默认安装到项目环境内。

## 长任务与运行记录

- 使用独立命名的 tmux 会话运行长任务。
- 日志和检查点使用独立的运行目录，记录启动命令、GPU 编号、环境版本、数据版本和配置。
- 原始数据、历史结果和失败记录保留；新试验使用新目录，避免覆盖已有运行。
