# Crystal tasks v1：Phase 0 勘察报告

日期：2026-10-25 PDT / 2026-10-26 UTC。状态：勘察完成，存在已列明的资产与决策缺口；停在闸门 1。Phase 1–8 尚未执行。

执行依据：[交接文档](../../docs/CrystalDLM%20交接文档：design%20understanding%20基准与%20AR–Hybrid%20对照.md)、[项目约定](../../AGENTS.md)、[术语](../../CONTEXT.md)。本地 HEAD 为 `0bf04e1158fcedeca8472221a544bdd90c230f85`，main 有既有未提交修改；本轮仅新增本报告及勘察回执，保留原文件。

## 1. 本轮执行与边界

- 只读连接 LiuGangGPU，核查磁盘、GPU、项目进程、环境包、数据登记、元数据、实例、模型配置与 tokenizer。
- 用现有 Qwen3 tokenizer 对 Understanding 全部 45,465 条重新统计长度；重新核对三个任务文件和三个 metadata 文件的 SHA-256；检查材料 ID、源结构哈希的唯一性及划分交集。
- 只读查询补标 SQLite 的成功条目数量；没有恢复补标，没有访问或转移 API 凭据。
- 核查本地历史实现，并比对远程归档中的关键源文件哈希；查阅官方 LLaMA-Factory/Qwen/HybridDiffusion 公开源码。
- 没有安装依赖、下载模型、加载 8B 权重、启动训练或生成/GRACE 评价，没有修改 `datasets/` 或历史实验。

本轮负责人已经确认：Phase 0 仅检查并给出安装方案；COV 先给具体方案，由负责人确认。交接文档四个原有闸门继续有效。

## 2. 当前目录与资产身份

服务器逻辑根目录：`/home/liugang/src/crystalDLM`；物理目录：`/data1/users/liugang/crystalDLM`。`findmnt -T` 返回 `/data1 /dev/vdc ext4`，`df -h` 为总量 2.9 TiB、可用约 1.4 TiB。

服务器已于 2026-10-25 调整布局，本地旧路径尚保留。后续不能把本地整个工作区覆盖同步到服务器。

| 用途 | 服务器当前项目相对路径 | 本地旧路径/说明 |
| --- | --- | --- |
| 正式任务数据 | `datasets/crystal_tasks_v1/` | `experiments/design_understanding_full_20261025/data/` |
| 数据准备、补标 DB/日志 | `data_preparation/crystal_tasks_v1/` | `experiments/design_understanding_full_20261025/` |
| 原始结构事实 | `datasets/sources/crystal_facts_v1/` | 原 `research/crystal_sharing_20261012/prepared_001` 有远程兼容链接 |
| 旧训练和评价实现 | `experiments/deprecated/<原目录名>/` | 本地仍是 `experiments/<原目录名>/` |
| 迁移记录 | `data_preparation/_migrations/20261025_layout/migration.json` | 服务器权威记录 |

远程归档 README 明确说明：旧脚本内容未改、保留旧绝对路径/相对根目录假设，且旧实验入口没有兼容链接。例如旧 `grace_score.py` 的 `parents[2]` 在归档后已不再是项目根目录。复用应在新实验/评价目录显式接入，不能直接运行旧入口。

## 3. 数据就绪情况、字段与长度

权威 manifest：`datasets/crystal_tasks_v1/manifest.json`。

SHA-256：`dce98892174a02a113a2de13200472d5997054f2b6a4306212be2fd9189c2d6c`。

当前状态为 `understanding_ready_design_awaiting_captions`；`sampling_fraction=1.0`；沿用源结构哈希划分，没有重新抽样或切分。原始采集 45,629 条中的 164 条已有预处理错误，保留在来源错误记录中。

| 任务 | training | validation | test | 就绪情况 |
| --- | ---: | ---: | ---: | --- |
| Understanding | 36,362 | 4,501 | 4,602 | 全部 JSONL 已发布，登记与文件哈希一致 |
| Design | 目标 36,362 | 目标 4,501 | 目标 4,602 | 正式 JSONL 未发布，不能开始训练或参考答案 NLL |

实际数据登记只有 `crystal_understanding_training`、`crystal_understanding_validation`、`crystal_understanding_test`。格式为 Alpaca JSONL，三个字符串字段 `instruction/input/output`；列映射分别为 `prompt/query/response`。数据内不手写模型聊天结束符，模板负责 EOS。

Understanding 输入为固定四字段问题加 `<CRYSTAL_START>…<CRYSTAL_END>`，不含生成请求、英文描述、ehull 或历史对话。答案为化学式、空间群编号、晶胞原子数、独立轨道数。化学式标签采用展开晶胞组成。例如 `mp-1210092` 的答案为 `Na2 Bi2 Mo4 O16`、82、24、6。属性标签及来源信息只在 metadata 中。

这一 PLaID-only 设置已在[数据准备票](../../.scratch/unified-crystal-ar-diffusion/issues/24-full-design-understanding-data.md)确认。交接文档第 11.4 节仍把类似调整列为待确认，当前按其“得到答复前沿用现有数据”执行，不改任务，也不新增推导问题。它仍主要衡量事实读取。

长度重新统计使用冻结 tokenizer、原生 chat template、`enable_thinking=False`，包含输入、答案及聊天模板，不截断、不加载模型权重：

| 划分 | 最小 | 平均 | P50 | P95 | P99 | 最大 | 超过 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 149 | 268.83 | 240 | 459 | 584 | 622 | 0 |
| validation | 150 | 270.35 | 241 | 444 | 583 | 613 | 0 |
| test | 150 | 268.40 | 240 | 451.90 | 582.99 | 614 | 0 |

答案正文自身为 27–42 tokens。建议 Understanding `cutoff_len=1024`、贪心评价 `max_new_tokens=128`；安装框架后还需核验真实 LLaMA-Factory 预处理和 loss labels。本次不是框架预处理/训练通过证明。

重新验证了 45,465 个唯一材料 ID、45,465 个唯一源结构哈希，三个划分两类集合均无交集。这不等同于 StructureMatcher 意义的晶体等价去重。

### Design 缺口

已复用历史描述 8,172 条，SQLite `successes` 当前有新描述 1,312 条（全在 train），合计 9,484 条；仍缺 **35,981 条**：train 27,778、validation 3,601、test 4,602。

`progress-snapshot.json` 和数据准备 README 标明 `stopped_at_user_request` / `resume_requires_user_instruction=true`，进程已停止，`pipeline.exit=130`。本轮复查没有项目训练/补标进程，两个已知项目 tmux socket 均无运行会话。服务器准备目录还明确记录尚未配置补标 API 密钥。

当前只有一个 Design 可读示例，原生模板长度为 376 tokens；不能据此报告全量 Design 长度分布或固定其 cutoff。描述补齐后才能完成对应长度审计。该任务的正式输入契约为“空间群编号＋ehull 上限”，输出为英文事实描述＋PLaID；没有用现有 9,484 条替代全量数据。

`prepare.py` 的既有条件规则将源 ehull 映射到能容纳它的最小上限 `0.08/0.1/0.2/0.3 eV/atom`。按 test metadata 和这一规则只读统计，4,602 条来源覆盖 135 个空间群、359 个不同的 `(空间群, ehull 上限)` 组合；四档来源条数分别为 2,663/348/1,017/574。这是未来条件列表的规模估算，正式列表须与发布后的 Design JSONL 一致。

## 4. 基座、模板与监督范围

基座目录：`models/Qwen3-8B-b968826d9c46/`。

- 准确名称 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- `config.json`：`Qwen3ForCausalLM`，`model_type=qwen3`，36 层、hidden size 4096、vocab size 151936、BF16。
- `VERIFIED.json` 与下载清单记录该原始基座及 13 个文件的既有校验。本轮读取身份清单、config 和 tokenizer，未重新计算全部 8B 权重文件哈希。
- `tokenizer_config.json` 的 chat template 有 `enable_thinking` 开关；不是 `Qwen3-8B-Base`。官方 [Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)也说明同一模型支持 thinking/non-thinking。

建议使用 **LLaMA-Factory `template: qwen3`、`enable_thinking: false`、`train_on_prompt: false`**。v0.9.5 的 `ReasoningTemplate` 在关闭 thinking 时把空思考前缀加入 prompt、不计 loss，答案加 EOS；训练和推理均使用同一模板。不要把原生 HF 非思考模板与另一套省略思考前缀的 `qwen3_nothink` 模板混用。依据：[固定版本模板源码](https://github.com/hiyouga/LlamaFactory/blob/7af909522a951e3ad9f022ea6f88b6755257eaa5/src/llamafactory/data/template.py)、[参数定义](https://github.com/hiyouga/LlamaFactory/blob/7af909522a951e3ad9f022ea6f88b6755257eaa5/src/llamafactory/hparams/data_args.py)。

Phase 1/2 必须打印真实 collator 产生的非 `-100` labels；Understanding 只能包含答案和约定的结束符。Design 的描述/晶体标签边界、四个控制 token 的注册及可训练行也需在两个 design 分支一致处理。此处未声称这一步已通过。

## 5. 环境与具体安装建议（尚未执行）

| 项目 | 本轮实查 |
| --- | --- |
| GPU | 4 × NVIDIA H20，每张 97,871 MiB；检查时均 0 MiB / 0%，无 compute processes |
| 驱动 / Torch CUDA 构建 | 580.126.20 / 12.4 |
| 项目 Python | 3.11.17 |
| PyTorch | 2.6.0+cu124 |
| Transformers / tokenizers | 4.57.6 / 0.22.2 |
| Accelerate / PEFT | 1.15.0 / 0.18.0 |
| pymatgen / PyXtal | 2025.10.7 / 1.0.8 |
| LLaMA-Factory / datasets / DeepSpeed | 未安装 |
| TensorBoard / wandb | 未安装 |
| torch.distributed / NCCL / FSDP | API 可用，FSDP 可导入；本轮未执行多卡通信 |
| 当前项目环境依赖一致性 | `python -m pip check` 通过 |

独立 `envs/grace` 已存在：Python 3.11.17、tensorpotential 0.6.2、TensorFlow 2.20.0、pymatgen 2025.10.7、ASE 3.29.0。它不需要重复安装到训练环境。

推荐安装方案：

1. 复用固定的 `envs/crystaldlm`；安装前保存完整包清单与 conda explicit 回执，所有缓存/临时文件按项目约定设置到 `/home/liugang/src/crystalDLM` 内。当前 `.mamba` 根目录不存在，安装时按约定创建；共享 base 和全局配置不动。
2. 固定 LLaMA-Factory **v0.9.5**，Git commit **`7af909522a951e3ad9f022ea6f88b6755257eaa5`**。先生成依赖解析计划，锁定现有 torch 2.6.0+cu124、transformers 4.57.6、peft 0.18.0；补装 datasets、TRL、torchdata、TensorBoard 和框架要求的依赖。新增传递依赖以解析回执逐项记录。
3. 该版本声明 `accelerate>=1.3.0,<=1.11.0`，现有 1.15.0 不兼容。建议明确降到 **1.11.0**；torchvision/torchaudio 如需要，配套 torch 2.6 使用 0.21.0/2.6.0。不能让 pip 无记录地升级 PyTorch。此环境变更等待闸门 1 确认。
4. 初始使用两卡 DDP + BF16 + LoRA，先不安装 DeepSpeed；FSDP 已随 PyTorch 存在，不为“可用性检查”另装软件。实际显存和吞吐由后续小样本检查确定。
5. 安装后执行 `pip check`、框架导入/CLI、模板和 labels 检查；再运行小批量检查，不能把依赖声明兼容当成训练验证。

依赖边界依据：[v0.9.5 官方 pyproject](https://github.com/hiyouga/LlamaFactory/blob/7af909522a951e3ad9f022ea6f88b6755257eaa5/pyproject.toml)。本轮只读取元数据，没有执行解析安装或环境更改。

## 6. 实现、参考代码和评价资产

以下旧代码本地在 `experiments/`，远程在 `experiments/deprecated/`。两处的 full/pilot `run.py`、严格 `score.py`、共享 `readback.py` SHA-256 已逐一一致核对。

| 组件 | 已定位位置与状态 |
| --- | --- |
| 旧 Hybrid 完整训练 | `flare_crystal_full_20261019/{run.py,model.py,evaluate.py}`；当前 loader 指向旧 AR checkpoint，不能作为新训练初始化 |
| 旧 FLARE 模型实现 | `flare_crystal_qwen3_8b_20261019/{model.py,run.py}`；可参考掩码与目标构造 |
| END 复现证据 | 本地 `research/loss_implementation_audit_20261023/{check_sampling_boundaries.py,sampling-boundaries.json}`；见下文 |
| 严格晶体审计 | `interleaved_ar_diffusion_20261002/score.py` 的 `audit()`；PLaID 解码依赖 `work/PLaID/utils.py` |
| 四字段解析 | `evaluation_common/readback.py`；新任务化学式“约化等价”规则仍需与交接文档一致核查 |
| GRACE 评分 | `structure_request_eval_20261017/grace_score.py`；另有 `test_grace_score.py`，本轮未重跑 |
| GRACE 模型 | 远程归档 `structure_request_eval_20261017/assets/grace-model/GRACE-3L-OMAT-large-ft-AM/`，saved_model.pb 存在 |
| GRACE 身份 | 仓库 `AMS-ICAMS-RUB/grace-foundation-models`，revision `46ead61dc8a7f90c11b1e59b8489e8e9bf97b888`；归档包 191,371,901 bytes，清单 SHA-256 `0d7ebce7d3ddbd168d086fbd43c583a4c34fb55b7509f78be3dc89db67651e52` |
| 固定凸包 | 同 assets 下 `2023-02-07-ppd-mp.pkl.gz`，220,011,896 bytes；旧评分使用 MP2020 correction + MP2023 hull |
| HybridDiffusion 原生代码 | 官方仓库 [yuchen-zhu-zyc/HybridDiffusion](https://github.com/yuchen-zhu-zyc/HybridDiffusion/tree/6ca547aebb72bfe897e80e0a683c776789e5f38c)，历史核对 commit `6ca547aebb72bfe897e80e0a683c776789e5f38c`；本轮未在服务器 work 中发现完整 checkout，需后续固定版本取得 |
| PLaID++ 代码 | 本地 `work/PLaID/` 为完整 checkout，HEAD `9ac91f07a263594bbaef1574efbf483a98152f0c`；服务器该目录只有 `utils.py` 及缓存，不是完整采样仓库 |
| PLaID++ 权重 | 本轮读取官方 snapshot `91f3acacc43bbcc5256f50f1f92badfdcc323094` 的模型卡与 `train_dpo/adapter_config.json`，确认为 LoRA r8/alpha32、q_proj/v_proj，基座配置为 `qwen-local/Qwen2.5-7b`；未下载权重，尚无已验证的完整加载链 |

原生 HybridDiffusion 的源码定位与 shift/mask 迁移差异见[此前源码核查](../../research/wayfinder_crystal_20261011/flare-code-contract-20261019.md)。不能把项目已有 FLARE 迁移实现当作“已经完成原生代码行为对照”。

PLaID++ 身份来源：[固定版本 adapter 配置](https://huggingface.co/HOPE-Lab-HMC/PLaID/blob/91f3acacc43bbcc5256f50f1f92badfdcc323094/train_dpo/adapter_config.json)、[固定模型卡](https://huggingface.co/HOPE-Lab-HMC/PLaID/blob/91f3acacc43bbcc5256f50f1f92badfdcc323094/README.md)。本轮通过 resolve 文本端点成功读取；模型卡称其为旗舰模型，但没有核实对应论文每项实验的权重身份。

END 已有证据：旧训练在 END 截断，最终块可短于 4；旧 `evaluate.py` 先填满 4-token 块再截断。已有 CPU 复现显示 END 后的待丢弃 token 能影响 END 前保留位置的 logits。Phase 4 应复用这一复现验证新路径，本轮没有修复或重跑；不能由旧回执推断所有样本都受到有害影响。

Phase 2 必须处理的实际接入差异：

- 旧 GRACE 入口写死 `EXPECTED_TRAIN_RECORDS=7272` 并读取旧子集，新颖性参考必须换成冻结的完整 **36,362 条训练结构**（是否另纳 validation 在协议中说明）。
- 旧阈值汇总只列 `0/0.08/0.1`；新 test 请求还有 `0.2/0.3`，主达标指标必须读取每条请求自己的上限，不能固定用 0.08 代替。
- 所有模型共用严格审计、GRACE 权重/弛豫/能量修正/凸包、去重及新颖性实现；旧路径与小样本计数不能沿用。
- PLaID++ 官方采样含解析失败重采逻辑；共同实验必须记录每次原始尝试并让失败占分母。
- 新结果中的 loss、NLL、validity、文字事实一致性、性质达标分别报告。

## 7. 闸门 1 的具体决策建议

以下为提交审核的方案，尚未固定为配置或启动命令。

| 待确认项 | 推荐方案 | 原因 / 边界 |
| --- | --- | --- |
| 微调与超参 | LoRA rank 32、alpha 64、dropout 0；q/k/v/o/gate/up/down/lm_head；学习率 2e-4，AdamW，weight decay .01，warmup .05 + cosine，clip 1，seed 42，BF16，global batch 8；各 3 epoch | 沿用原始 8B AR 配方，而非旧嵌套 checkpoint 继续训练的 5e-5；两个 design 分支严格一致。LLaMA-Factory 对控制 token 可训练行的接入需验证 |
| 验证与保存 | 全量 validation；每 epoch 至少 5 次，保存每 epoch、最低 validation loss 与最终 checkpoint | 固定 3 epoch，不早停；周期步数随最终 batch 固定，不能套旧 909 updates |
| Understanding | 保持已经发布的 PLaID-only 四字段任务；cutoff 1024，贪心 max_new_tokens 128 | 已有明确确认和完整数据；不扩展成新的推导题 |
| design 条件与预算 | 待 JSONL 发布后，冻结 test 中不同的 `(SG, ehull 上限)`；当前推导为 359 组，每条件每 seed 10 次，3 个采样 seed（42/43/44） | 每个 design 模型约 10,770 次尝试；3 个系统约 32,310 次。是采样随机性重复，不是 3 次独立训练；不能据此声称覆盖训练方差 |
| ehull 上限 | 按每条请求现有的 0.08/0.1/0.2/0.3 判定；保留 0.08 子集结果便于低 ehull 对比 | 不更改训练数据或把所有请求变成 0.08 |
| COV | 采用 StructureMatcher，`ltol=.3, stol=.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=False`；同一实现用于全部系统 | 复用历史去重设置。明确命名为 StructureMatcher coverage，不声称等于 CDVAE 指纹版 COV |
| COV 参考集与分母 | test 全部 4,602 条真实结构；匹配生成的 strict-valid 原始结构，弛豫前计算。R=匹配到的参考数/4,602；P=有参考匹配的有效生成数/有效生成数，同时单列有效生成数/全部尝试 | 避免把 S.U.N. 筛选后的集合用于覆盖评价。有效生成数为 0 时 P 按预先固定规则记 0；全部失败仍完整反映在 validity/S.U.N. |
| 新颖性参考 | AR/Hybrid 使用全量 36,362 条训练结构；加入 PLaID++ 后主对照使用本项目训练结构与可核验的 PLaID 公开训练库并集，另保留相对本项目训练库的数值 | 所有系统使用同一参考库；公开 adapter 的全部训练来源无法核实的部分需标明，不能称为对其全部训练数据的新颖性 |
| 额外分布指标 | 暂不增加 Wasserstein 等可选指标 | 本轮覆盖文档要求即可 |
| PLaID++ 对齐 | 原生空间群条件＋公开接口的标量 ehull=请求上限，结果按共同的“不大于上限”判据评价，并明确它接收的是标量代理条件 | 不是语义严格等价的区间条件；共同结论主要依据匹配 AR/Hybrid。若要求输入完全等价，则 PLaID++ 的 ehull 部分需另定，不能默认为已解决 |
| GPU / 时长 | 三卡训练，小检查一张；所有项目任务总占用最多三张。各阶段先测吞吐再给完整运行时长估计 | 项目 GPU 上限已明确，无需重新询问第三张卡；本轮没有得到新的总 GPU 小时预算 |

另一个文字歧义需确认：第 7.2 节说“三个模型使用同一条件列表”，但三种训练模型包括不生成晶体的 AR-understanding。建议解释为 **AR-design、Hybrid-design、PLaID++** 共享 design 请求；AR-understanding 只参加四字段评价及对应 Test NLL，不强行放进 design 生成表。

Formula 比较建议采用 `Composition(...).reduced_composition` 等价，另外独立比较晶胞原子数；明确元素顺序/约化形式不产生假错误，不能让公式的倍率差异污染原子数指标。具体实现与负例测试在 Phase 2 提交闸门 2。

## 8. 后续依赖与停点

1. 闸门 1 确认安装与实验/评价方案后，执行 Phase 1–2：安装到项目环境、固定配置、建立共享评价工具及规定测试；到闸门 2 停止，**不因闸门 1 通过就开始训练**。
2. 闸门 2 通过后，按 Phase 3 先运行 AR-understanding 的 32 条过拟合检查，再跑全量 3 epoch 与评价；Design 数据未就绪时其训练保持未启动。
3. 要完成整个文档，必须恢复或另行解决 Design 缺失描述。该任务此前被负责人暂停；本报告不恢复、不自行补造描述，也不改数据版本。负责人需明确是否恢复，以及执行位置和凭据来源。补标完成后另行发布/确认数据版本，遵守禁止改动现有 `datasets/` 的要求。
4. END 修复可按文档与 Phase 3 并行，但 Hybrid-design 仍必须等待闸门 4。最终还需 Phase 6 多种子评价、Phase 7 PLaID++ 实测和 Phase 8 报告。

尚未验证：真实 LLaMA-Factory labels/多卡训练；Design 全量 token 分布；Hybrid 原生行为对照；PLaID++ 完整加载链与条件适配；GRACE 本轮新数据的运行。以上缺口均不能记为已完成实验。
