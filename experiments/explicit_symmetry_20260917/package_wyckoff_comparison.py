"""Verify and deliver per-material Wyckoff baselines and symmetry rechecks."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
OUT=WORKSPACE/'outputs/wyckoff_comparison_20260919'
FILES=['wyckoff.tokens','wyckoff_standard.tokens','wyckoff_decoded.cif','space_group_comparison.json','representation_comparison.json','comparison.md']


def main():
    evidence=ROOT/'wyckoff_comparison_20260919'
    groups=json.loads((evidence/'space_group_summary.json').read_text())
    summary=json.loads((evidence/'representation_summary.json').read_text())
    rows=json.loads((evidence/'representation_records.json').read_text())
    assert groups['materials']==summary['selected']==summary['passed']==512
    assert all(t['same_number']==t['same_hall']==t['same_operation_count']==512 for t in groups['tolerances'])
    for name,digest in summary['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for r in rows:
        case=ROOT/'corpus_512'/r['id']
        assert len((case/'wyckoff.tokens').read_text().split())==r['roundtrip_tokens']
        assert len((case/'wyckoff_standard.tokens').read_text().split())==r['standard_tokens']
        check=json.loads((case/'space_group_comparison.json').read_text())
        assert hashlib.sha256((case/'input.cif').read_bytes()).hexdigest()==check['source_sha256']
        assert hashlib.sha256((case/'decoded.cif').read_bytes()).hexdigest()==check['decoded_sha256']
        for base in [ROOT,WORKSPACE/'outputs/explicit_symmetry_program_20260917']:
            destinations=[base/'corpus_512'/r['id']]
            if (base/'pilot'/r['id']).is_dir():destinations.append(base/'pilot'/r['id'])
            for destination in destinations:
                destination.mkdir(parents=True,exist_ok=True)
                for name in FILES:
                    if destination/name!=case/name:shutil.copy2(case/name,destination/name)
    for run in ['012-spacegroup-recheck','015-wyckoff-baseline-all']:
        receipt=json.loads((ROOT/'wyckoff_comparison_receipts'/run/'receipt.json').read_text())
        assert receipt['state']=='completed' and receipt['exit_code']==0
    counts=summary['total_tokens'];lookup={r['id']:r for r in rows}
    lines=['# 空间群保持与 Wyckoff 编码对照','',
      '2026-09-19，Burgundy Slurm 573131，cpunode-057。全部 512 个原始材料逐例处理，未替换失败样本。','',
      '## 1. 解码后的空间群是否保持','',
      '**是。在相同识别设置下，512/512 个原始 CIF 与显式程序解码 CIF 的空间群编号、Hall 编号及操作数一致。** 使用 spglib 2.6.0、angle_tolerance=-1，分别检查 1e-7、1e-5、1e-3、1e-2、1e-1 Å 五档容差，共 2560 组前后配对。','',
      '| 距离容差 symprec | 空间群编号相同 | Hall 编号相同 | 对称操作数量相同 |','|---|---:|---:|---:|']
    for t in groups['tolerances']:lines.append(f"| {t['symprec_A']:g} Å | {t['same_number']}/512 | {t['same_hall']}/512 | {t['same_operation_count']}/512 |")
    lines+=['',
      'YbBPt₂ 在 1e-3 Å 下前后都是 #180，在 1e-7 Å 下前后都是 #3。原始有限精度坐标会使识别结果随容差变化，不能跨容差比较后把变化归因于 tokenizer。',
      '输入/输出 CIF 文件头中的 P1 是显式列出全部原子的格式约定，不是本次比较依据。逐例 JSON 同时保留文件头编号和从坐标重新识别的结果。','',
      '## 2. 已保存的 Wyckoff 对照','',
      '每个 corpus_512/<material_id>/ 文件夹新增下列文件；原十个 pilot 文件夹也同步保存，原来的 input.cif、encoded.sym、decoded.cif 未被改写。','',
      '| 文件 | 内容 |','|---|---|',
      '| wyckoff_standard.tokens | 标准常规晶胞的紧凑标签表示：SG、Hall 设置、晶格参数、元素、Wyckoff 标签、独立自由参数 |',
      '| wyckoff.tokens | 带原晶胞变换、输出精度、必要残差的版本，可还原原输入数值 |',
      '| wyckoff_decoded.cif | 可还原版解码结果 |',
      '| space_group_comparison.json | 原 CIF / 显式程序解码 CIF 的五档空间群复核 |',
      '| representation_comparison.json | 三种长度、自由参数含义、Wyckoff 解码误差及空间群 |',
      '| comparison.md | 逐材料的简明说明与对照 |','',
      'Wyckoff baseline 的规则来自固定公开的 PyXtal 表，空间群操作集先匹配到准确 Hall 设置。不同空间群或原点下同名 WP 不会混用。标准版有理想化过程；可还原版补足原晶胞坐标变换及必要数值修正，所以二者分别统计，不能把标准版的短长度当成完整原数据的保真成本。','',
      '**512/512 个可还原 Wyckoff 表示通过晶格、组成、周期坐标的往返检查；写回 CIF 再读的最大坐标误差记录为 0.0 Å，并在 1e-3 Å 下重新识别出与原输入相同的空间群。**','',
      '| 材料 | CIF | 显式约束程序 | 标准 Wyckoff | 可还原 Wyckoff |','|---|---:|---:|---:|---:|']
    for mid in ['mp-30009','mp-1095422']:
        r=lookup[mid];lines.append(f"| {r['formula']} | {r['cif_tokens']} | {r['explicit_tokens']} | {r['standard_tokens']} | {r['roundtrip_tokens']} |")
    lines.append(f"| 512 个合计 | {counts['cif_tokens']:,} | {counts['explicit_tokens']:,} | {counts['standard_tokens']:,} | {counts['roundtrip_tokens']:,} |")
    lines+=['',
      f"在同样可还原原输入数值的版本中，显式程序的总长度比本 Wyckoff 参考编码多约 {100*(counts['explicit_tokens']/counts['roundtrip_tokens']-1):.1f}%。本参考编码未对 FRAME 做稀疏压缩，也不是全局最短的 Wyckoff 编码，更不是某篇生成模型的完整复现。",'',
      'Token 沿用每个完整数值、标签或控制符一个单元的口径，不是 BPE 数；共享公开 Wyckoff 参考表不重复计入每个材料。','',
      '## 3. 哪种方式更体现空间规则','',
      '两者理论上的结构/对称信息可以等价。差别是规则如何呈现在模型输入中：传统标签把规则留在 SG/设置限定的外部表里；显式程序把固定值、自由度、坐标关系和生成操作写在序列中。',
      '以 #180 的 Pt 6j 为例，标签版写 Pt 6j u，标准表解释为 (u,2u,1/2)。标签序列本身看不到“一个自由参数同时决定 x 和 y”。显式表达式直接写出该关系。我们当前文件采用该轨道的另一个等价代表 (1−y,y,5/6)；代表和参数基不同，不代表结构不同。','',
      '| 模型需要接触的内容 | 纯 Wyckoff 标签输入 | 显式程序输入 |','|---|---|---|',
      '| 哪些坐标固定、哪些可变 | 从标签和参数结构学习，或由表提供 | 常数与 FREE / PARAM 直接区分 |',
      '| x=y、x+y=1、z=1/4 等关系 | 关系包含在表里 | 直接以变量、ADD / NEG / MUL、常数表达 |',
      '| 旋转、反演、螺旋/滑移的坐标作用 | 空间群类别限定外部规则 | G 行直接给出仿射变换 |',
      '| 多个位点使用相同规则 | 可以共享同一个标签 embedding | FAMILY 直接共享同一公式，成员参数独立 |','',
      '可提出的学习假设是：基本运算与变量关系可跨位点/空间群组合复用，减少只靠类别编号记忆每种情况的需求。当前实验验证了这些规则的执行与结构还原，并没有验证这个假设会提升模型训练或生成质量。',
      'Wyckoff 本身已经包含完整的晶体学知识，LLM 也可能从标签和上下文学到规则。如果基线再加入位点对称、自由度或显式表特征，两者的差别会缩小。因此不能声称显式程序“信息更完整”“一定更短”或“必然更容易学习”。','',
      '## 4. 快速查看 YbBPt₂','',
      '```text',(ROOT/'corpus_512/mp-1095422/wyckoff.tokens').read_text().strip(),'```','',
      '6j 的参数轴为标准表的 x。实际数学模板为 (x,2x,1/2)；另外两个标签 3d、3c 为零自由度，没有连续参数。','',
      '## 复现与边界','',
      '详见 README.md。编码/解码读取同一固定参考表 fixtures/wyckoff_list.csv，支持本次有序、满占位结构。实验元数据、热参数、磁矩等不做文本级还原。',
      '```bash','python wyckoff_baseline.py decode corpus_512/mp-1095422/wyckoff.tokens reconstructed.cif','```','',
      '证据 JSON、512 行比较 CSV、代码 SHA256 和计算节点收据随包提供。所有新增文件已同步至现有 outputs/explicit_symmetry_program_20260917/corpus_512/ 对应目录；旧的 20260917 ZIP 保持不变，请使用本次新包获取新增对照。',
    ]
    OUT.mkdir(parents=True,exist_ok=True)
    report='\n'.join(lines)+'\n';(evidence/'RESULTS.md').write_text(report);(OUT/'RESULTS.md').write_text(report)
    shutil.copy2(ROOT/'WYCKOFF_BASELINE.md',OUT/'README.md')
    for name in ['wyckoff_baseline.py','test_wyckoff_baseline.py','audit_wyckoff_baseline.py','audit_spacegroup_roundtrip.py','program_codec.py','audit_wyckoff_families.py','test_program.py','requirements.txt']:
        shutil.copy2(ROOT/name,OUT/name)
    shutil.copytree(ROOT/'fixtures',OUT/'fixtures',dirs_exist_ok=True)
    shutil.copytree(evidence,OUT/'evidence',dirs_exist_ok=True)
    shutil.copytree(ROOT/'wyckoff_comparison_receipts',OUT/'hpc_receipts',dirs_exist_ok=True)
    for r in rows:
        target=OUT/'corpus_512'/r['id'];target.mkdir(parents=True,exist_ok=True)
        for name in ['input.cif','encoded.sym','decoded.cif',*FILES]:shutil.copy2(ROOT/'corpus_512'/r['id']/name,target/name)
    with (OUT/'token_comparison_512.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['material_id','formula','space_group','cif_tokens','explicit_tokens','wyckoff_standard_tokens','wyckoff_roundtrip_tokens'])
        for r in rows:writer.writerow([r['id'],r['formula'],r['space_group'],r['cif_tokens'],r['explicit_tokens'],r['standard_tokens'],r['roundtrip_tokens']])
    zpath=OUT.with_suffix('.zip')
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():archive.write(f,str(f.relative_to(OUT.parent)))
    with zipfile.ZipFile(zpath) as archive:
        assert archive.testzip() is None
        assert sum(n.endswith('/wyckoff.tokens') for n in archive.namelist())==512
        assert sum(n.endswith('/space_group_comparison.json') for n in archive.namelist())==512
    receipt={'package':str(zpath),'bytes':zpath.stat().st_size,'sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),
             'per_material_files_verified':512,'original_and_explicit_decoded_files_unchanged':True,'source_code_hashes_verified':True}
    (evidence/'delivery.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
