"""Write and verify conventional Wyckoff comparisons beside all 512 materials."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import warnings

from pymatgen.io.cif import CifWriter

import wyckoff_baseline as baseline
from audit_spacegroup_roundtrip import identify
from test_program import parse,compare


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path('corpus_512');records=json.loads((root/'records.json').read_text())
    results=[];start=time.perf_counter()
    for i,row in enumerate(records):
        case=root/row['id'];source=case/'input.cif'
        result={'id':row['id'],'formula':row['formula'],'cif_tokens':row['cif_tokens'],'explicit_tokens':row['program_tokens']}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                full,standard,audit=baseline.encode_cif(source)
                (case/'wyckoff.tokens').write_text(full)
                (case/'wyckoff_standard.tokens').write_text(standard)
                reconstructed=baseline.decode((case/'wyckoff.tokens').read_text())
                original=parse(source);error=compare(original,reconstructed)
                output=case/'wyckoff_decoded.cif';CifWriter(reconstructed,symprec=None,significant_figures=14).write_file(output)
                reread=parse(output);reread_error=compare(original,reread)
                before=identify(original,1e-3);after=identify(reread,1e-3)
                assert before['number']==after['number']
            result.update(status='pass',error_A=error,reread_error_A=reread_error,source_identified=before,wyckoff_decoded_identified=after,**audit)
            (case/'representation_comparison.json').write_text(json.dumps(result,indent=2))
            note=['# '+row['formula']+' ('+row['id']+')','',
                  '- `input.cif`：原输入。',
                  '- `encoded.sym`：我们的完整显式位点约束程序。',
                  '- `decoded.cif`：我们的解码结果。',
                  '- `wyckoff_standard.tokens`：标准晶胞的 SG / Hall 设置 / 元素 / Wyckoff 标签 / 独立自由参数 / 晶格数。该版本是理想化的常规晶胞参考，不负责逐数值还原原晶胞。',
                  '- `wyckoff.tokens`：附带原晶胞变换、输出精度、必要残差的 Wyckoff 对照，可还原原始晶胞数值。',
                  '- `wyckoff_decoded.cif`：上述可还原版的独立格式解码结果。',
                  '- `space_group_comparison.json`：原 CIF 与我们的解码 CIF 在五种相同容差下重新识别空间群。',
                  '- `representation_comparison.json`：参数含义、长度、对照解码误差和空间群。','',
                  f"Token 数：CIF {row['cif_tokens']}；显式约束 {row['program_tokens']}；标准 Wyckoff {audit['standard_tokens']}；可还原 Wyckoff {audit['roundtrip_tokens']}。",'',
                  '## Wyckoff 参数对应的规则（说明文本，不计入标签版输入）','',
                  '| 元素 | WP | 参数轴 | 数值 | 查表得到的代表坐标 |','|---|---|---|---|---|']
            for node in audit['nodes']:
                note.append(f"| {node['element']} | {node['label']} | {', '.join(node['free_axes']) or '无'} | {', '.join(format(x,'.9g') for x in node['parameters']) or '无'} | {node['template']} |")
            note+=['','同名 WP 需要结合空间群与设置解释。我们的公式所选轨道代表可能与表中标准代表不同，但按完整空间群展开后应描述同一组位置。']
            (case/'comparison.md').write_text('\n'.join(note)+'\n')
        except Exception as e:
            result.update(status='fail',error=repr(e));(case/'wyckoff_failure.txt').write_text(traceback.format_exc())
        results.append(result)
        if (i+1)%64==0:print(json.dumps({'processed':i+1,'passed':sum(r['status']=='pass' for r in results),'seconds':time.perf_counter()-start}),flush=True)
    out=Path('wyckoff_comparison_20260919');out.mkdir(exist_ok=True)
    (out/'representation_records.json').write_text(json.dumps(results,indent=2))
    passed=[r for r in results if r['status']=='pass']
    summary={'selected':len(results),'passed':len(passed),'failures':[r for r in results if r['status']!='pass'],
             'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,'symprec_A':1e-3,
             'source_hashes':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['wyckoff_baseline.py','audit_wyckoff_baseline.py','audit_spacegroup_roundtrip.py']}}
    if passed:
        summary['total_tokens']={k:sum(r[k] for r in passed) for k in ['cif_tokens','explicit_tokens','standard_tokens','roundtrip_tokens']}
        summary['max_error_A']=max(r['reread_error_A'] for r in passed)
        summary['baseline_reidentified_same_group']=sum(r['source_identified']['number']==r['wyckoff_decoded_identified']['number'] for r in passed)
        summary['with_frame']=sum(r['frame_tokens']>0 for r in passed)
        summary['with_residuals']=sum(r['residual_atoms']>0 for r in passed)
    (out/'representation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    assert len(passed)==512,'Baseline failures require inspection, not sample replacement'


if __name__=='__main__':main()
