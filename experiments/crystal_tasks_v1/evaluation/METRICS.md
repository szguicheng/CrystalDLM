# Crystal tasks v1 评价口径（闸门 2）

适用范围：AR-design、Hybrid-design、PLaID++ 共用 design 评价；AR-understanding 单独评价四字段和 AR Test NLL。固定方案见 `../configs/shared.json`。

| 指标 | 输入与计算 | 分母 / 未完成状态 |
| --- | --- | --- |
| 四字段准确率 | test 真实晶体对应的模型回答；化学式按 pymatgen reduced composition 比较，空间群编号、晶胞原子数、轨道数按整数比较。重复且矛盾的字段判错，缺失字段判错 | 每字段及 all-four 均为正确条数/全部 test 回答；必须与全部材料 ID 一一对应 |
| AR Test NLL | 参考答案 teacher forcing；logits 向后错一位预测 labels，忽略 `-100`；汇总 CE 总和/目标 token 总数 | 不是逐批平均的平均。design 分别报告描述/边界、晶体正文＋END，Understanding 只报告答案及 EOS |
| 去噪 CE | 固定 seed 和材料 ID 产生互补 mask，每个晶体 token 恰被覆盖一次；对被遮位置求 CE 和 S | `L_crystal_denoise=S/T_crystal`；`L_denoise_current=S/(T_text+T_crystal)`。均不称为 NLL，不与 AR NLL 直接比较 |
| 格式合法率 | 提取唯一的晶体段；PLaID++ 可提供原始 PLaID；用既有解码器得到 Structure | 可解码尝试数/全部原始尝试 |
| strict valid | 原样调用冻结的严格 audit：组成、计数、晶格、Wyckoff、解码结构、重叠与空间群自洽 | 严格有效数/全部尝试；重复 START/END 不接受 |
| 空间群命中率 | 从实际解码结构分析 SG（symprec .01、angle tolerance 5），与请求编号比较 | 命中数/全部尝试；与 strict-valid 单独报告 |
| ehull 达标率 | 固定 GRACE 弛豫＋MP2020 修正＋MP2023 凸包的 corrected proxy ehull，与本条请求的 .08/.1/.2/.3 上限比较 | 达标数/全部尝试；收敛、弛豫后有效、弛豫后 SG 另报。不是 DFT |
| Stable | strict-valid、弛豫前 SG 命中、GRACE 收敛、弛豫后有效且 SG 命中、有限 corrected ehull 达标同时成立 | 满足数/全部尝试 |
| Unique / Novel / S.U.N. | 在 Stable 集合中对弛豫后结构去重，每个等价类取一个代表；与完整训练库匹配判新颖 | Unique=类数/Stable 数；Novel=新颖类数/类数；S.U.N.=新颖 Stable 类数/全部尝试 |
| COV-R / COV-P | 所有 strict-valid **弛豫前**输出与 test 真实结构比较，不先按性质筛选 | R=被匹配的 test 结构数/4,602；P=有 test 匹配的有效输出数/有效输出数；有效输出为空时 P=0 |
| 多种子 | 每个 seed 单独计算去重与各指标，再报告均值、样本标准差和原始 k/N | 固定采样 seeds 42/43/44；仅表示生成随机性，不是训练随机性 |

结构匹配统一使用 `StructureMatcher(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True, scale=True, attempt_supercell=False)`。这是 StructureMatcher coverage，不是 CDVAE 指纹版 COV。AR/Hybrid 新颖性参考为完整 36,362 条训练结构；PLaID++ 外部比较还须传入已核验的公开训练库并集。

每次原始尝试都必须出现在冻结 attempt plan 和输出中；解析失败占分母，不补采。评价器按 attempt_id 校验完整性，并校验 seed、请求、GRACE 回执绑定的输出/请求哈希；不能用只保留成功样本的文件计分。参考库任一结构解码失败即报错，不缩小分母。

缺少有效候选的 GRACE 回执或发生评分基础设施错误时，性质及 S.U.N. 等相关 rate 标为 `null`，状态为 `not_run`、`partial` 或 `failed`，不写作测得的 0%；结构和 COV 指标仍可报告。每次评价保存配置、命令、Git/源文件身份、模型/检查点、数据哈希、种子、环境、起止时间及逐条回执。

Design 自由英文描述的事实一致性未定义自动评分契约，当前不以 Understanding 正确率替代；描述 CE 也不代表事实正确。该项在最终报告单独标明未评价。
