"""Assemble reports and a reproducible research package from verified receipts."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
OUT=WORKSPACE/'outputs/explicit_symmetry_program_20260917'


def read(path): return json.loads((ROOT/path).read_text())


def main():
    corpus=read('corpus_512/summary.json');records=read('corpus_512/records.json')
    hall=read('hall_audit/summary.json');wyckoff=read('wyckoff_audit/summary.json');frames=read('unseen_frames/summary.json')
    pilot=read('pilot/records.json')
    assert corpus['passed']==512 and not corpus['failures']
    assert hall['passed']==530 and wyckoff['passed']==1731 and frames['passed']==96
    for name,digest in corpus['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert frames['core_sha256']==corpus['code_sha256']['program_codec.py']
    for r in records:
        assert len((ROOT/'corpus_512'/r['id']/'encoded.sym').read_text().split())==r['program_tokens']
        assert (ROOT/'corpus_512'/r['id']/'decoded.cif').exists()
    for run in ['010-final-audits','011-unseen-frames']:
        receipt=read(f'hpc_receipts/{run}/receipt.json')
        assert receipt['state']=='completed' and receipt['exit_code']==0
    assert wyckoff['source_sha256']==hashlib.sha256((ROOT/'fixtures/wyckoff_list.csv').read_bytes()).hexdigest()
    old=json.loads((WORKSPACE/'experiments/cif_roundtrip_20260917/fixed_dictionary_results/records.json').read_text())
    old={r['id']:r for r in old}
    total_plain=sum(r['plain_structure_fields'] for r in records)
    lines=['# 显式位点约束程序：方法与验证结果','',
      '日期：2026-09-17。结论：已得到一个符合本次目标的可逆编码候选。它用可执行的坐标关系和空间群生成操作替代不透明的 Wyckoff/位点规则编号，不保存逐原子邻接边列表。当前验证覆盖结构还原、规则表达与长度；尚未训练 LLM 或证明论文级新颖性。','',
      '## 如何拆开 Wyckoff 位置','',
      '每个独立原子轨道直接写出三坐标的固定值、独立变量与变量关系。给定全局对称操作后，这些式子定义一族允许的代表位置，实际自由参数选出其中一个具体结构。','',
      '| 对象 | 可见的规则 | 自由度 |','|---|---|---:|',
      '| 固定位点 | (0,0,0) 或 (1/2,1/2,5/6) | 0 |',
      '| 轴上位置族 | (0,0,z)，z 独立可变 | 1 |',
      '| 关联坐标位置族 | (1−y,y,5/6)，只生成 y | 1 |',
      '| 一般位置族 | (x,y,z)，三个独立变量 | 3 |','',
      '这让模型直接接触“坐标固定”“两个坐标相等/互补”“某个参数同时影响两个坐标”等关系。它不能保证模型已经学会这些知识，但提供了可组合、可执行的语义，而不是要求模型从类别编号中隐式记忆查表结果。','',
      '## YbBPt₂ 实际表示','',
      '原 CIF 的 12 个原子合并为三组：Yb=(1/2,1/2,5/6)，B=(1/2,1/2,1/3)，Pt=(1−y,y,5/6)，y=0.852342。两条生成操作分别是 (y,y−x,z+2/3) 和 (y,x,2/3−z)。闭包给出完整操作集，展开为 3 个 Yb、3 个 B、6 个 Pt。','',
      '下面是完整的 49 单元程序，不依赖任何 Wyckoff 字典：','',
      '```text',(ROOT/'pilot/mp-1095422/encoded.sym').read_text().strip(),'```','',
      '`ADD NEG y 1` 就是 1−y；`FREE 0.852342` 指定 y 为独立参数。`ROUND 6` 还原原 CIF 的有限小数精度。没有逐原子边列表。','',
      '## 重复约束直接共享，不换成新编号','',
      '同一种坐标关系在多个元素/位点出现时，程序把公式写一次，随后列每个成员自己的元素和自由参数。例如 TmNiGe₂：','',
      '```text','FAMILY 4 ADD NEG y 1 PARAM 1/4','Tm 0.107937','Ni 0.322749','Ge 0.459473','Ge 0.746923','```','',
      '四个成员都采用 (1−y,y,1/4)，但每个成员的 y 独立。`PARAM` 是变量声明，规则就在同一个输入中展开，没有外部 rule ID，也没有绑定不同成员的参数。','',
      '## Encoder 与 decoder','',
      'Encoder 从输入结构识别近似对称性，把固定位点的操作写成线性方程 (R−I)q=n−t，经有理行最简化得到独立参数和最小仿射约束。然后从完整空间群操作中选择少量生成元，并检验其闭包与完整操作集相同。相同的显式模板可分组共享。','',
      'Decoder 解析有限算术原语 x/y/z、ADD、NEG、MUL、FREE/PARAM 和有理常数，以算术闭包展开对称操作，再生成各轨道的等价原子。解码路径不识别空间群、不查询 Wyckoff/空间群数据库、不读取原 CIF。审计时禁用了 spglib 识别和数据库查询接口，仍通过全部检查。','',
      '生成元选择是确定性的贪心选择与冗余剔除，不宣称全局最小生成元数。坐标规则在当前参考系中经行最简化，不宣称所有等价原点/基底都产生相同 token 序列。','',
      '## 数值信息如何保留','',
      '原始 CIF 往往只是近似对称，例如 0.833333 与精确 5/6。不能为了严格匹配这些小数就把 #180 强行降成 #3，也不能静默理想化并丢失原数值。','',
      '本方案把精确符号骨架与数值载荷分开：保留输入晶格参数，记录输出小数精度；非标准原点通过显式 ORIGIN 平移进入有理参考系；舍入仍不能恢复的差异用 RESIDUALS 保存。所有原点/精度/残差字段都计入长度。','',
      f"512 个材料中 {sum(r['origin_tokens']>0 for r in records)} 个使用 ORIGIN，{512-corpus['residual_free']} 个使用残差，共修正 {corpus['total_residual_atoms']} 个展开原子。残差可能使最终数值结构不再具有符号骨架的严格数学对称性，因此不能把‘保留原始数值’写成‘最终数值完全对称’。",
      '信息保留范围为晶格、元素、满占位原子数和周期坐标。CIF 注释、标签、热参数、磁矩、实验元数据不做文本级还原。','',
      '## 验证范围与证据','',
      '| 检查 | 结果 | 证明的范围 |','|---|---|---|',
      '| 固定的真实 MP-20 结构 | 512/512 通过 | 识别、编码、解码、写回 CIF，元素/晶格/坐标保持 |',
      '| 标准 Hall 设置 | 530/530 通过 | 覆盖 230 空间群的生成元闭包和一般位置解码 |',
      '| Wyckoff 位置族 | 1731/1731 通过，每种 3 个参数实例 | 5193 次特殊/一般位置族展开，覆盖默认设置 |',
      '| 新材料与坐标参考系变化 | 32×3=96/96 通过 | 未参与开发的验证集材料，原始、平移原点、换基底 |',
      '| 语义干预 | 通过 | 改自由参数确实移动对应坐标，共享模板的成员参数独立，拒绝非线性生成操作 |','',
      '晶体学表只用于生成独立验证样本，不是 decoder 的数据源。真实 CIF 结构坐标误差阈值为 1e-7 Å，晶格矩阵分量阈值为 1e-8 Å；512 个原始样本写回再读后的最大坐标差记录为 0.0 Å。不同参考系验证仅证明结构信息保留，不等于 token 不变性。','',
      '## Token 长度','',
      '计数沿用用户要求：每个算术原语、元素、完整数值、控制标记分别计一个单元，完整表达式不合并成一个巨型词。包括晶格、生成元、模板、自由值、原点、小数精度和残差。连续数值未量化，不是某个现成 LLM 的 BPE 数量。','',
      f"512 个材料总计：原 CIF **{corpus['total_cif_tokens']:,}**，新程序 **{corpus['total_program_tokens']:,}**，减少 **{100*(1-corpus['total_program_tokens']/corpus['total_cif_tokens']):.1f}%**。",
      f"单材料长度比中位数 **{corpus['ratio_median']:.3f}**，95 分位数 **{corpus['ratio_p95']:.3f}**，最大 **{corpus['ratio_max']:.3f}**。**{corpus['shorter_or_equal']}/512** 不长于 CIF，其余最长增加约 **{100*(corpus['ratio_max']-1):.1f}%**。",'',
      '| 材料 | 原 CIF | 旧边轨道完整序列 | 新显式程序 |','|---|---:|---:|---:|']
    for r in pilot: lines.append(f"| {r['formula']} ({r['id']}) | {r['cif_tokens']} | {old[r['id']]['actual_stream_tokens']} | {r['program_tokens']} |")
    lines+=['',
      f"公平性边界：这些源 CIF 以 P1 展开原子并包含文件元数据。只数元素＋xyz＋6 个晶格数的裸结构参数，共 {total_plain:,} 个，比新程序更短。这一裸参数数目不包含对称规则与格式开销。因此新程序相对 CIF 的缩短，不等于比所有几何表示都更短。",
      '旧边轨道与新程序之间同时改变了表示目标（去掉显式近邻图）、数值对称处理和语法，表中的改善不是单因素消融。','',
      '## 先例与新颖性边界','',
      '已有工作分别覆盖了这些组件。CLOUD/SCOPE 使用紧凑生成元字符串，WyFormer 使用位点对称＋枚举，Matra-Genoa 使用 Wyckoff 身份＋自由坐标，DiffCSP++ 使用仿射 Wyckoff 约束。CrysTune 的官方 PDF 索引摘录也显示坐标模板和仿射展开；该 PDF 当前受访问挑战阻挡，不能声称已阅读全文。','',
      '本候选可陈述的区别是：把坐标约束本身作为可执行的原语序列，与同语法的全局生成操作组合，并在输入内共享显式位置族，不依赖 Wyckoff 标签或隐藏规则字典完成解码。仿射代数、群闭包、坐标模板本身不是新晶体学。没有证据支持全球首创或已证明论文贡献。','',
      '原文核查见 prior_art.md，主要来源包括 [WyFormer](https://arxiv.org/abs/2503.02407)、[Matra-Genoa](https://arxiv.org/abs/2501.16051)、[DiffCSP++](https://arxiv.org/abs/2402.03992)、[CLOUD](https://www.nature.com/articles/s41467-026-70467-3)、[CrysTune](https://openreview.net/pdf?id=Oe5iihLiiV)。原点约定按 [spglib 官方文档](https://spglib.readthedocs.io/en/stable/dataset.html#transformation-matrix-and-origin-shift)。','',
      '## 仍需后续研究','',
      '- 实际 LLM/非自回归扩散的学习与生成收益，需与 Wyckoff 字母、WyFormer 等表示做同模型训练对照。当前未训练新生成模型。',
      '- 等价晶胞、原点、原子排列下的唯一规范化序列尚未证明；目前保证被测试输入的可逆信息。',
      '- 数据范围是有序、满占位、非磁性的三维周期结构。任意无序 CIF、任意超胞、所有实验元数据尚未支持。',
      '- 连续自由参数/残差尚未离散化。用于 DLM 时需明确连续预测头或数值分词机制，不能把字段数冒充模型实测 BPE 数。','',
      '## 使用与复现','',
      '测试环境为 Burgundy matgl / Python 3.10，依赖见 requirements.txt。编码/解码无需 GPU。','',
      '```bash','python program_codec.py encode corpus_512/mp-1095422/input.cif example.sym','python program_codec.py decode example.sym reconstructed.cif','```','',
      '完整样本和程序在 corpus_512/，原十个例子的简明结果在 pilot/。未见材料参考系检查在 unseen_frames/，标准群和位置族检查在 hall_audit/、wyckoff_audit/。测试生成用的官方表在 fixtures/；decoder 不读取它。',
      '源数据固定为 WyFormer 官方 MP-20 commit e8e9add5df904284116af8a06c203555d908f290。训练集按材料 ID 的 SHA256 排序取前 512，验证集独立取前 32，无失败替换。选择清单与源文件 SHA256 保留。完整审计脚本中的源 CSV 路径指向本次 Burgundy 数据目录，复跑时需使用相同原 CSV；单文件 encode/decode 命令可独立运行。','',
      'Slurm 568689，2 CPU，cpunode-058，同一个 tmux 分配完成所有迭代。关键源文件 SHA256、成功收据以及首次 45 个原点失败的记录均保留，最终结果没有隐藏这些失败。',
    ]
    report='\n'.join(lines)+'\n';(ROOT/'RESULTS.md').write_text(report)
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['program_codec.py','test_program.py','test_origin.py','test_family_blocks.py','audit_corpus.py','audit_spacegroups.py','audit_wyckoff_families.py','audit_unseen_frames.py','requirements.txt','README.md','RESULTS.md','PLAN.md']:
        shutil.copy2(ROOT/name,OUT/name)
    for directory in ['pilot','corpus_512','hall_audit','wyckoff_audit','unseen_frames','origin_tests','fixtures','hpc_receipts','snapshots']:
        shutil.copytree(ROOT/directory,OUT/directory,dirs_exist_ok=True)
    shutil.copy2(WORKSPACE/'research/notes/explicit_symmetry_constraints_prior_art_20260917.md',OUT/'prior_art.md')
    with (OUT/'token_comparison_512.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['material_id','formula','space_group','cif_tokens','program_tokens','ratio','residual_atoms','origin_tokens','family_blocks'])
        for r in records: writer.writerow([r[k] for k in ['id','formula','space_group','cif_tokens','program_tokens','ratio','residual_atoms','origin_tokens','family_blocks']])
    audit={'goal_requirements':[
        {'requirement':'Expose spatial/symmetry rules rather than Wyckoff/private IDs','evidence':['pilot/mp-1095422/encoded.sym','pilot/mp-1079976/encoded.sym','program_codec.py'],'verified':True},
        {'requirement':'Encoder and decoder preserve crystal structural information','evidence':['corpus_512/summary.json','hall_audit/summary.json','wyckoff_audit/summary.json','unseen_frames/summary.json'],'verified':True},
        {'requirement':'Prefer lengths not much larger than original CIF','evidence':['corpus_512/summary.json','token_comparison_512.csv'],'verified':True,'scope':'512 fixed real inputs; max 1.088x and median 0.564x; not a universal bound'},
        {'requirement':'Meaningful change from per-atom relation numbering','evidence':['program_codec.py','README.md'],'verified':True,'scope':'No serialized edge list or Wyckoff lookup; explicit coordinate programs and shared family blocks'},
    ],'core_sha256':corpus['code_sha256']['program_codec.py'],'all_required_experiments_passed':True,'learning_gain_not_claimed':True,'world_first_not_claimed':True}
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    zip_path=OUT.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file(): archive.write(path,str(path.relative_to(OUT.parent)))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert sum('/corpus_512/' in n and n.endswith('/encoded.sym') for n in archive.namelist())==512
    receipt={'output':str(OUT),'archive':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'audited_core_sha256':audit['core_sha256']}
    (ROOT/'delivery.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
