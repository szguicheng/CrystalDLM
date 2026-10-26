# 活动目标进度

当前目标未完成：尽可能跑通共享LLM交错AR＋晶体Diffusion，用学习式改进，不硬编码回答或替换生成值。

alignment_002完成：混合生成8/8、实际闭环7/8、旧留出29/32、3%留出30/32、新问法115/128，.95编辑6/8。纯AR新问法128/128，但旧配对漏组成0/32。只有8已见结构，不能完成goal。

已实施真正新材料阶段cohort.py/PROTOCOL_COHORT_V5.md，冻结data_002 train1024、val128、test256。两模型各2048更新，从alignment_002继续。方向、编辑句式、结构版本、历史引导和回答风格独立随机抽取，32768样本曝光已验证：每种方向×句式都有310–371例，.95实际出现；最大长度787，目标均可解析。

先在val按ID哈希前64真正未见材料上跑生成、参考读取/编辑、实际闭环、冲突读取，再做旧8回归。测试256保持未使用，验证有信号后执行最终测试，不把64验证当完成目标。

下一动作：确认新cohort作业ID并检查队列与logs/cohort-ID.out/err、cohort_001/hybrid/status.json。按真实验证错误调整。源码和数据有快照，失败分母不删，不补齐模型值。

新材料作业ID601205，输出cohort_001。下一次须先重新检查Slurm状态和logs/cohort-601205.out/err，不凭文件或意图断定活着。首跑hybrid，2048更新后64真正未见验证，再旧8回归，随后AR同预算。最终256测试未触及。

最新实查：601205 RUNNING于gpu-a100-03，hybrid完成2048更新及全部验证，AR已进入同预算训练。远端训练源码保持提交时版本。

验证分层已核对并保存cohort_exposure_001/validation_strata.json：64验证材料与训练ID无交集，旧8均属于训练集；64组成全部未见、63空间群已见，联合组成/空间群条件全部未见。独立轨道数2–7，常规胞原子数4–56。该队列同时考察新组成泛化，不能把生成失败直接等同于交错机制失败；结合参考读取、参考编辑、实际闭环和AR控制定位。未读取测试256的内容。

本地rescore.py抽出通用score_evaluation(folder,materials)，用于新64队列的统一v2核对；旧score_branch仍要求8材料与64冲突案例。既有test_checks.py完整通过，通用函数精确复现adaptation_001四分支已保存v2计数。只复用先前命名轴短语规则，不改数值容差/缺失判定，不改原输出。该本地评分重构未同步进正在运行的远端训练；后续对完整64原始输出另存v2结果及源码/原输出哈希。

cohort_001元数据、源码快照和训练曲线已同步本地。provenance_verification.json核对通过：源码快照逐文件匹配训练receipt，远端曝光审计与本地完整曝光审计相同，冻结64验证ID与分层记录相同，源data_002清单哈希相同。同步时训练状态1000更新；未下载checkpoint，不声称已有模型验证结果。

完整混合结果见RESULTS_COHORT_V5_PARTIAL.md：64验证G0、Qref64、Eref49、CL0、swap63；旧8 G0/Q8/E7/CL0，生成能力相对alignment_002退化。新问法49/64，15失败集中在population问法缺原子数/轨道数，另外3问法各16/16。此轮未跑通，goal保持活动。

已实现零更新诊断diagnostics/cohort_probe.py及Slurm入口，已有单一test_checks全通过。stingy额外提交因该分区仅允许1作业而拒绝；查询配置后gpu_a100成功提交601264，实查PENDING(Priority)。不把排队当执行或结果。下一动作：重新检查601205和601264；同步最终AR与诊断raw/summary，分离问法、材料、遮盖、温度因素，再据证据改进学习或通用采样，不修补晶体答案。probe选择的16训练材料已核对包含旧8全部。

后续推进：squeue --start预计601264次日08:47才启动；现有分配A100仅用13840/81920MiB，已取消未执行的601264，改用srun --jobid=601205 --overlap启动同一诊断。scontrol实查601205.1 State=RUNNING、gpu-a100-03；首个训练材料输出已经落盘。SSH执行句柄70251，输出logs/cohort-probe-step-601205.out/err，结果仍cohort_probe_001。主AR训练仍运行，最近记录825/2048。共享GPU期间训练耗时受争用影响，不作公平训练速度比较；两模型更新/曝光不变。后续先查该步骤、主作业和raw/summary，勿重复提交已取消排队任务。

最新实查：601205.1已COMPLETED/0:0（10:08），SSH70251完成；704条诊断已同步核验，结果见RESULTS_PROBE_V6.md。训练16材料两问法三温度共288尝试全失败；仅验证材料mp-980669在训练问法低温下有4重复尝试成功，总576中4。全MASK正文训练问法train68.1%/val63.4%，验证问法均约55.6%。降温不能恢复，学习不足/退化与条件措辞敏感性需要干预。

已实现recover.py/recover.slurm/PROTOCOL_RECOVERY_V6.md并通过单一test_checks全部检查。下一轮两模型从cohort最终checkpoint各继续G专项2048更新（每训练材料32曝光）再联合512更新；G条件6措辞，首个条件在整条8样本轨迹同步替换，不泄漏长度或结构答案。冻结本地recovery_exposure_001审计通过：G32768例、所有1024各32次、最大303token；联合8192例、所有材料各8条件曝光、最大778token。尚未提交recovery，等待601205完整AR结束后同步当前新增源码，不能在AR未完成时复制中间初始化或修改运行源码。

601205最近实查RUNNING，AR已完成2048更新，正进入验证；诊断已结束，不与AR评估争用GPU。后续动作：同步完整AR结果并做统一v2核对，确认Slurm终态/summary complete，再同步recover.py、recover.slurm、rescore.py、test_checks.py、PROTOCOL_RECOVERY_V6.md，提交recovery_001并核对新作业。仍保持128验证（后64此前未预测）与256最终测试完整范围，不因失败缩减目标。

纯AR验证当前已评13/64：G5、Qref13、Eref13、实际闭环5、swap13。仅为部分结果，完整总结待跑完。持久SSH监视句柄93237已确认RUNNING并有新评估输出；下一次可先write_stdin(93237)复用连接，直到601205终态，再做完整结果核对。不要凭部分计数提交或判定目标完成。

最新终态：601205 COMPLETED/0:0，1:23:44。完整AR64 G17/Qref63/Eref61/CL17/swap63；额外问法63/64、旧8G4/CL4/Q8/E8。混合此前完整结果不变。两模型v2计数与初版一致，paired/Wilson/原哈希已保存verified_report_v2.json，报告RESULTS_COHORT_V5.md。goal未完成。

主作业结束后已同步新增源码并检查远端CLI，提交生成恢复601362。实查RUNNING于gpu-a100-03，结果recovery_001；从cohort最终两个checkpoint继续，先H再AR，各G2048＋联合512，不用测试256。记录submission_recovery_001.json。旧监视93237的queue空判定被HPC的“Last update”行干扰而未退出；已识别其独有远端bash PID1589973并仅终止旧监视，不修改Slurm作业。下一次先重新查601362和recover-601362.out/err、recovery_001/hybrid/generation/status.json，确认真实训练进度，再跟进专项64和联合128（含此前未评的后64）的结果。

601362已实际训练，同步时混合G专项250/2048更新；日志task_counts G=16、text_loss=null，暂无错误。startup_verification.json核对通过：初始化指纹等于cohort混合最终指纹，完整源码快照匹配训练receipt，远端两阶段曝光审计与本地冻结审计完全一致，数据清单哈希匹配，fresh64与已诊断64无交集。暂无新验证结果，不能因损失或训练启动宣告恢复。旧93237确认已退出；HPC squeue会附带更新行且失效ID时可能显示其他作业缓存，未来须明确匹配请求job ID，并结合sacct/训练receipt确认。

最新实查601362 RUNNING，混合G专项600/2048，尚无验证结果。已建立复用SSH监视句柄78554：精确匹配sacct的JobID/State，不用squeue非空判活；遇混合generation/validation_64/evaluation_summary.json或明确终态退出。下一次先write_stdin(78554)取得新输出并确认句柄状态，若已结束则查601362终态和暖身64输出。监视只是读取状态，不重启或改变主训练。

独立只读审查recovery_review完成，见RECOVERY_REVIEW_V6.md：全32768G＋8192联合样本重建、读取/编辑目标和哈希核对通过，所有test_checks通过，无Critical/Important实现问题。原64/新64需在结果后分开汇总，原始ID和两组清单已正确保留；这属于后续报告事项，不改活跃训练。审查没有HPC访问，不代表验证成功。监视最新实查601362 RUNNING、混合专项1375/2048更新，尚未验证；后续继续读78554精确状态。

最新：监视78554因完整warm64结果出现而正常退出0；主作业未结束。混合专项2048已完成，中间验证G1/Qref7/Eref1/CL0/swap0，见RESULTS_RECOVERY_WARM_V6.md和verified_analysis.json；生成触发失败12、END/PAD30、解析18、结构3、通过1，读取29未正常终止/异常。原版与v2计数一致，不修模型答案。中间阶段出现严重能力退化，不能把低训练损失当生成恢复。

601362最新sacct RUNNING（41:52时），已进入混合联合阶段，最近日志175/512更新。下一次须重新查job601362及recover-601362.out/err和hybrid/joint/status.json，78554已结束不再wait。等待joint/validation_128完整结果，分开汇总原64与新64；后续还有旧8及同预算AR恢复。最终256仍未使用，goal保持活动。

更新实查：601362仍RUNNING（54:21），联合512更新已结束、正在混合128验证；目前已评85/128，G1/Qref85/Eref57/CL1/swap84。这是部分结果：读取已明显恢复，生成仍很弱；不能替代全部128、新64子组或最终256结论。下一次先重新检查并同步hybrid/joint/validation_128，结果完整后再分析，仍有AR恢复等待执行。

最新完整：601362 COMPLETED/0:0，2:29:30。混合128 G3/Q128/E76/CL2/swap127、额外读取127；AR G56/Q128/E109/CL51/swap126、额外125。旧8 H G0/CL0/E6，AR G5/CL5/E7。报告RESULTS_RECOVERY_V6.md。

诊断order_probe_001在601362.0完成0:0（5:13），最终混合joint512固定权重，48材料×3种子×随机/置信度共288。所有step数量一致、前后指纹一致；有效数随机18/144 vs置信度5/144（不能视为144独立材料）。模型概率排序来源LLaDA官方主代码，但本方法并非其完整复现，不采用置信度变体。

修复了一项评分bug：wp.ops[0]只接受固定代表点，等价Ti2a(.5,.5,.5)被错误拒；min over全wp.ops现与实际decoder轨道成员一致。测试通过，独立344点/10SG审查通过，不改坐标/容差/字段/权重。统一v3重算所有已完成cohort/recovery/order001成功数不变；见ORBIT_SCORER_FIX_V3.md和orbit_rescore_v3_report.json。ROOTscore新版本仅在主作业结束后同步到远端。新通过G但缺流程不能算CL。

从左到右释放，仅顺序变化、同一counts/温度/检查点/病例/种子：standalone601603已COMPLETED/0:0（2:28），out order_probe_002，共144尝试，训练0/48、oldVal4/48、后Val3/48。已同步本地，fingerprint、数量和与order001病例/seed/counts核验已启动；v3统一重审尚待执行。旧代码snapshot保存在order_probe_001/source_snapshot.py，新order_probe.py支持--policies，source_snapshot随输出保存。不要重启已完成作业。

下一动作：完成order002验证归档，核对完整AR新旧64分组与初始化/源码，再设计最小同批G/E/Q重配比（从最终joint检查点，全部噪声级别，保留AR触发监督和QA/E，避免纯G遗忘）。尚未实现或提交该下一轮；不要凭计划认定有活跃训练。test256仍未读，目标未完成，不能用AR代替混合达成。

order002的verification_receipt.json已核验144例、初末权重指纹与order001一致、检查点SHA与病例/种子/每步count一致，每个已执行生成的release位置严格0..255一次性，未回写已释放token。进一步v3汇总已发起，本地exec新句柄见后续工具结果；不把启动当已完成。主作业601362及诊断601603均终态，无活跃训练。后续必须做联合配比入口并实际提交，或者据真实失败证据选择其他必要改进，不能靠重复已完成作业状态维持目标活动。

v3对order002也完整重审，成功数仍0/48、4/48、3/48，旧所有cohort/recovery/order001计数不变。所存原始输出不改，温度/置信度/左序变体不采用。

已实现rebalance.py/rebalance.slurm、协议PROTOCOL_REBALANCE_V8.md。G/E/Q每优化器步8/4/4，4096更新，joint噪声覆盖1..32并保留G/E触发监督；两模型顺序执行，从recovery/<mode>/joint最终checkpoint继续。全65536曝光冻结：各条件材料32G/16E/16Q，最大793token，哈希2d0773845491266b0d71bd9ff76b40416ac8b91c9ecdebfb7f55038544ee64c0。test_checks全部通过；独立审查核验全部4096批次及targets/独立dict，见REBALANCE_REVIEW_V8.md，无需修复问题。

已同步新入口和协议，远端CLI成功，sbatch已接受新job601653，out rebalance_001。尚需确认其RUNNING/具体进度和startup源/初始化指纹/远端曝光与本地相同；不能仅凭提交认定已训练。下一次检查sacct601653与logs/rebalance-601653.out/err、rebalance_001/hybrid/status.json。完整128/旧8/额外读取/原64后来64分组均保留，test256仍未读。当前goal未完成。

最新实查601653 RUNNING，日志确认混合joint任务8G/4E/4Q，无错误；同步时200/4096更新。startup_verification核验实际初始指纹等于recovery最终、源码snapshot逐文件匹配receipt、完整远端65536曝光与本地冻结相同、data002清单哈希相同。尚无新验证结果。后续先fresh sacct/日志确认此job活跃，再观察训练完成后的128验证，不因低损失宣称成功。

再次实查601653 RUNNING，混合400/4096，配比8G/4E/4Q正确，暂无验证。建立精确sacct job ID的复用SSH监视句柄87954，遇H validation128完整summary或明确终态退出。下一次先write_stdin(87954)确认新输出/运行状态；如果监视完成，再同步结果及确认主job。不会凭文件或旧状态判活，也不启动其他训练来覆盖当前实验。

继续实查601653 RUNNING，监视87954最新混合1625/4096，仍8G/4E/4Q，无新验证结果。等待当前真实训练，不修改活跃源码/模型/数据。补齐前轮AR原64 G27/Q64/E56/CL24、后来64 G29/Q64/E53/CL27，全部128总计与v3审计核对一致；stratified_summary含原输出与manifest哈希，RESULTS_RECOVERY_V6.md已更新。两组现均已使用验证；test256未使用。下一次先write_stdin87954；若监视完成，重新查主job601653并同步H完整128结果，再按失败类别分析，AR同预算仍需完成。

复用87954精确sacct监视，最新601653 RUNNING、混合2850/4096更新；持续正常推进，8G/4E/4Q不变，H完整验证尚未出现。下一次继续读87954；如完成则先核对主作业、同步全部H128原输出/summary/strata，再分析。当前属于已核实的训练等待，目标活动，不重启任务。

实查601653 RUNNING/00:55:02，H4096训练已结束并保存checkpoint。status trained、最终指纹5e33596b7296806a7bca26a0b4d2843a2dd95927e269e633f42c1aa5a0ecf755；本地training_verification核对源码/起点/全部噪声级别/49152晶体样本与末条曲线。128验证已实际启动，最初4例G2/Q4/E4/CL2/swap4（部分结果，不作完整结论）。87954仍为活跃精确监视，等待H128完整summary后同步统一分析；主作业后续额外读取/旧8/AR同预算仍需完成。

H完整128已完成并同步核验：G14/Q128/E111/CL12/flow13/swap127/noCrystal0；前64 G8/E57/CL7、后来64 G6/E54/CL5。相较恢复G净增11且无丢失、CL净增10且无丢失，参考编辑39增/4失。生成失败36边界/49解析/29结构或解码，报告RESULTS_REBALANCE_V8_PARTIAL.md、verified_analysis.json保存哈希/配对。G10.9%和CL9.4%仍不达门槛，goal未完成。
旧监视87954因H完整summary出现正常退出0，不再poll该句柄。fresh sacct601653 RUNNING/01:09:07，主作业继续额外读取/旧8/AR。已建新复用监视27802，精确sacct job ID，遇AR128核心summary或明确终态退出；最新输出H unseen_reading96/128。下一次先读27802，必要时同步小输出（exclude checkpoint.pt）核对H额外/旧8及AR起点/源码，等AR完整再分析下一干预，不重启601653。

H status已validated，旧8完整G3/Q8/E7/CL3/swap8（已同步），AR同批联合训练实际启动25/4096，监视27802仍活跃。AR初始/暖身指纹等于recovery AR最终，源码snapshot核验通过；不把AR启动当完成验证。

最新27802监视确认601653 RUNNING、AR275/4096。只读核对model.crystal/reverse_step：32级吸收过程、每级1/k释放、已释放值不回写；k32首步期望256/32=8 token，存在早期承诺的采样近似假说但未证实为根因。时间编码nn.Embedding(STEPS+1)，直接增加循环上限会越界或引入未训练时间，不能当普通配置修复。未修改model.py/活跃训练/采样。前轮置信度与左序均失败，暂不无证据重试。下一次继续27802等AR同预算结果，H报告/原输出已完整保存。

最新复用27802精确监视：601653 RUNNING，纯AR1075/4096，8G/4E/4Q不变；H全部结果已核验保存，AR本轮尚无验证，goal保持活动。下一次继续27802，若监视退出再重新查主job及同步AR完整核心结果，不能wait已关闭的87954。

最新27802仍活跃，精确sacct601653 RUNNING，AR2250/4096；H完整结果与verified_analysis不变，AR本轮尚未验证。继续当前作业，不重启/不改模型或采样，goal活动。下一次先读27802，遇AR128summary正常退出后核对主job并同步完整原输出；最终256仍未使用。

2026-10-03最终核验：601653 `COMPLETED/02:41:16/0:0`，rebalance_001远端与本地summary均complete。H与AR各4096更新，冻结65,536曝光按每16例核验全部4096个优化器批次为8G/4E/4Q；两模式源snapshot 13项SHA匹配，初始指纹分别等于对应recovery终点。128核心、额外读取和old8输出均已同步，ID、原64/later64、local score_evaluation重评分、原始hash及配对gain/loss核验通过，详见RESULTS_REBALANCE_V8.md。AR为G62/128、closed_loop59/128；G仍低于50%门槛64/128，混合H为G14/128、closed_loop12/128，目标未达成。两模型各一个训练seed，验证材料已重复使用，test256未读。相较前轮同时增加联合预算与改变配比，不作单一配比因果结论。

后续诊断方向为AR warm-start + predictor alignment三分支1024更新方案。现有曝光辅助器核验首512批仅覆盖512/1024材料；首1024批覆盖全部1024材料且每个各一次，因此采用每臂1024更新与16384冻结样本。候选predictor_align.py、协议及小检查已在本地准备，尚无GPU任务；候选diagnostics/rebalance_probe.py的完整probe从未运行，暂缓且不纳入本轮证据。

额外读取的自动reader_v2为AR111/128，较recovery前轮125少14；17例失败全部集中在READ_TEST第1问法，自动评分字段标记15例atoms+orbits、1例sg、1例sg+atoms+orbits；其余三问法各32/32且无调用异常。H127/128，仅同一问法有1个自动sg失败。自动字段标记不等于人工确认的事实错误，详见下方语义补充；不修输出、不改reader或把READ_TEST问法加进训练。

修复offset恢复根因：新增`Hybrid.restore_checkpoint`并接入当前实验树所有活动Hybrid checkpoint loader及手写offset诊断路径。仅已知V8同位置`model.py` SHA `d7abb6...` 且config缺少offset的旧checkpoint按0恢复；offset字段必须为整数0/1，其他未知缺字段来源拒绝。V9 AR→H先恢复源AR offset0再显式切换目标arm，数据启动SHA固定为V8 `data_002` manifest的`e3a5a86e...`。test_checks通过legacy0、unknown legacy拒绝、offset1恢复及默认0回归；GPU未提交。

语义人工核对补充：17个AR自动reader_v2失败中，9例缺orbit数、6例orbit数明确错误（其中1例兼SG错误）、1例仅SG错误、mp-567545一例事实完整正确但parser未识别。另16例`cell population is N`里的atom count正确，v2没有识别此写法；mp-567545的“independent orbits are 5”也未被解析。原自动111/128分数和raw记录保持不变；该人工补充不冒充统一新评分，也不把17例全算作模型错误。V9 primary reader仍冻结named_axis_v2，本轮不改scorer。

2026-10-03执行纠正：Slurm 601839 / `predictor_align_1024_001`在gpu-a100-03实际以CPU训练；sacct终态`CANCELLED by 90790433/00:11:07/0:0`。原输出目录与log未覆盖，本地已排除大checkpoint同步，保留6条慢速AR曲线、状态、source snapshot、16384行exposure及hash；无checkpoint/summary/validation。这是执行配置错误，模型性能无结论。原因是AR权重restore后未将新Hybrid移到CUDA，训练循环自身也没检查device。修复后的runner每个分支恢复与offset设定后执行`.cuda().train()`并打印设备/fingerprint，`train_phase`增加CUDA参数断言并在训练receipt写真实parameter device/CUDA index/同步计时口径。现有`test_checks.py`仍为20项PASS，14项源码pin与`bash -n`通过；独立差异审查已通过。新run脚本为`predictor_align_1024_002.slurm`，固定旧AR指纹、数据、16384曝光和全部评估设置不变。修复源码SHA：`predictor_align.py=d6b6338e...`、`adapt.py=6ebd2b15...`、脚本`fa14c7f3...`。远端14项pin/data/V8 AR最终指纹/002空闲复核通过后提交job 601858，初次fresh状态RUNNING/00:00:02/gpu-a100-03；首个AR启动与运行CUDA证据见`startup_verification_predictor_align_1024_002.json`，submission receipt见`submission_predictor_align_1024_002.json`。当前AR control实际进度见fresh Slurm/status/curve；仍需三臂全验证和终态。
2026-10-03执行错误归档：Slurm 601858 / `predictor_align_1024_002`于`FAILED/01:11:15/1:0`终止。AR control完整validated；hybrid_same完成1024更新后核心validation写出66/128行，再因`score.py`长度匹配捕获`7.02.`、`float()`抛ValueError而中断；该正在计算的第67条未写入raw JSONL。旧日志、66条H raw、checkpoint、full exposure、源码快照和partial summary保留，详情见`execution_exception_predictor_align_1024_002.json`。此run没有完整Hsame结果，不作算法结论。`score.py`最小修复为仅在晶胞长度字段float转换失败时记false继续（raw和阈值不变）；加入尾随句点测试，先行测试复现原异常，改后`test_checks.py`20 PASS。修复SHA：`score.py=c1dfe1b8...`、`test_checks.py=ca0bf491...`；全臂重跑候选使用全新`predictor_align_1024_003`目录，尚未提交，待快速差异复核。

2026-10-03固定接续运行已由独立审查放行并提交：Slurm 601979 / `predictor_align_1024_003`初次fresh `RUNNING/00:00:09/0:0`，节点gpu-a100-04。该作业不重训已完成的AR/Hsame训练：AR继承601858全128/extra/old8原raw并用修复scorer另存计数；Hsame从601858 offset0 final checkpoint严格恢复后全128重放、原66行科学payload必须匹配（seconds/forward_calls另记录）；仅Hshift从V8 AR final fingerprint `7567d5dd...` offset1训练1024更新。冻结16384曝光、data SHA、三分组、reader/physics与test256边界不变。submission receipt见`submission_predictor_align_1024_003.json`，startup device/前驱checkpoint/prefix对照核验待计算节点产物。

601979 latest实查`RUNNING/00:27:20/0:0`。AR继承后修复评分为G51/Q128/E113/CL47/flow51/swap126；extra reading97/128，old8 G6/CL6。Hsame已验证：core128 G0/Q128/E0/CL0/flow0/swap122；extra118/128；old8 G0/Q8/E0/CL0/swap7。002原前66行与003 replay逐行科学payload完全一致（validation IDs相等、first_mismatch null、forward_calls无差异；raw仅含新的timing观测变化）；receipt写入`startup_verification_predictor_align_1024_003.json`和Hsame prefix comparison。Hshift当前625/1024，mode hybrid/offset1，parameter_device cuda:0，initial/warm fingerprint均为V8 AR `7567d5dd...`；每更新计时约0.8秒，GPU job继续RUNNING。prefix未择优，test256未读取。
601979后续fresh`RUNNING/00:34:45/0:0`：Hshift完成1024更新，offset1/cuda:0，V8共同初始指纹不变，final FP `f784f004...360b8c1c`，median0.81071s/update。offset1 checkpoint strict reload与final fingerprint核验通过，进入128核心验证（20/128时G0/Q20/E0/CL0/flow0/swap20/noCrystal0，仅部分计数）。Hsame extra128与old8均已完整，root/warmstart可读Hsame core receipts/raw本地归档；作业继续到Hshift core、extra、old8和summary complete。

2026-10-03：按最终独立审查结果提交V9三臂运行 `predictor_align_1024_001`，Slurm job 601839。提交前核对远端14个源码SHA、data_002 manifest SHA `e3a5a86e...`、V8 AR终点4096更新/最终指纹 `7567d5dd...`，并确认本任务输出目录与submission receipt均空闲。初次fresh状态为PENDING/0:0；提交receipt见`submission_predictor_align_1024_001.json`。现已在gpu-a100-03/A100启动AR control：status=training、mode=ar、offset=0，初始指纹等于V8 AR最终指纹，初始source snapshot核验通过。冻结曝光SHA `d53173eb...`：16,384行、1,024批、1,024材料各一批且每批8G/4E/4Q。首阶段startup receipt见`startup_verification_predictor_align_1024_001.json`；两个Hybrid臂待串行启动时核验各自offset与同一初始指纹。继续监控三臂全验证和主job终态，test256仍未使用。

2026-10-03 最终终态补记（覆盖上文提交时的临时状态）：601839 / `predictor_align_1024_001`因CPU设备错配取消并单独归档；601858 / `_002`因Hsame验证评分器异常`7.02.`失败，但AR与Hsame各自1024更新检查点保留。固定续跑601979 / `predictor_align_1024_003`在gpu-a100-04终态`COMPLETED/00:49:57/0:0`，summary为`complete`。AR沿用601858完整raw并用修复评分器重评分；Hsame从offset0检查点完整重放128例，旧66例科学payload全部相同；仅Hshift从V8共同AR终点`7567d5dd...b21c85`以offset1训练1024更新，参数设备cuda:0，严格重载与最终指纹核验通过（final `f784f004...b8c1c`）。冻结16384曝光仍为1024材料各一批、每批8G/4E/4Q；data manifest SHA `e3a5a86e...c0004`。三臂core128、extra128、old8及原64/later64均完成，完整计数、失败阶段分类、Hshift对Hsame配对增益/损失见`RESULTS_AR_WARMSTART_SHIFT_V9.md`及`predictor_align_1024_003/summary.json`。offset1相对offset0的G与closed-loop配对均为0 gain/0 loss；两个Hybrid均G0/128、closed-loop0/128，目标未达成。单训练seed，验证重复使用；test256未使用。601839/601858执行故障不算模型结果，不重训已完成臂。
