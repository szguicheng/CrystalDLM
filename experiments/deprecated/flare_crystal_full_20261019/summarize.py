"""Verify completed stages and write a result table without hiding unfinished arms."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    result={'status':'incomplete','arms':{},'scope':'full fixed7272/900;700designrequestsperarm;GRACEproxy'}
    table=['| 模型 | 训练步数 | 验证材料 | 生成次数 | 严格有效 | 四项读取全对 | 80meV最终新颖达标 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['ar','flare']:
        manifest=read(HERE/arm/'manifest.json')
        validation=read(HERE/arm/'validation-summary.json')
        generation=read(HERE/f'{arm}_evaluation'/'summary.json')
        scores=read(HERE/f'{arm}_evaluation'/'grace_score'/'summary.json')
        stage=(HERE/f'{arm}-stage.txt').read_text().strip() if (HERE/f'{arm}-stage.txt').exists() else 'not_started'
        exit_path=HERE/f'{arm}-exit.txt'
        state=dict(stage=stage,exit_code=int(exit_path.read_text()) if exit_path.exists() else None,
                   training=manifest,validation=validation,generation=generation,scoring=scores,complete=False)
        if manifest and manifest.get('status')=='complete':
            assert manifest['completed_updates']==909
            rows=[json.loads(s) for s in (HERE/arm/'updates.jsonl').read_text().splitlines()]
            assert len(rows)==909 and [r['update'] for r in rows]==list(range(1,910))
            ids=[m for r in rows for m in r['material_ids']]
            assert len(ids)==7272 and len(set(ids))==7272
        if validation:
            assert validation['status']=='complete' and validation['n']==900
            rows=[json.loads(s) for s in (HERE/arm/'validation.jsonl').read_text().splitlines()]
            assert len(rows)==900 and len({r['material_id'] for r in rows})==900
        if generation and generation.get('status')=='complete':
            assert generation['completed_attempts']==700
            rows=[json.loads(q.read_text()) for q in (HERE/f'{arm}_evaluation').glob('shard-*/attempts/*/*.json')]
            assert len(rows)==700 and len({(r['request_id'],r['attempt_id']) for r in rows})==700
            assert generation['strict_valid_structures']==sum(r['structural_valid'] for r in rows)
            generation['readback_all_four_correct']=sum(
                r.get('readback',{}).get('all_four_correct') is True for r in rows)
        if scores and scores.get('status')=='complete':
            assert scores['completed_attempts']==700
            score_rows=list((HERE/f'{arm}_evaluation'/'grace_score/attempts').glob('*/*.json'))
            assert len(score_rows)==700
        state['complete']=bool(manifest and manifest.get('status')=='complete' and validation and
            generation and generation.get('completed_attempts')==700 and scores and scores.get('status')=='complete'
            and state['exit_code']==0)
        result['arms'][arm]=state
        value=lambda obj,key: obj.get(key,'—') if obj else '—'
        novel=scores['threshold_counts']['0.08']['global_unique_novel_conditioned_converged_leq'] if scores and scores.get('status')=='complete' else '—'
        table.append(f"| {arm} | {value(manifest,'completed_updates')} | {value(validation,'n')} | {value(generation,'completed_attempts')} | {value(generation,'strict_valid_structures')} | {value(generation,'readback_all_four_correct')} | {novel} |")
    if all(x['complete'] for x in result['arms'].values()):
        result['status']='complete'
        plans=[read(HERE/f'{a}_evaluation'/'run_manifest.json')['attempt_plan'] for a in ['ar','flare']]
        assert plans[0]==plans[1]
    if not args.allow_partial: assert result['status']=='complete'
    for name,value in [('summary.json',json.dumps(result,indent=2)),('RESULTS.md',
        '# FLARE式完整训练与验证\n\n状态：'+result['status']+'。未完成项以“—”表示。\n\n'+'\n'.join(table)+
        '\n\n性质为既定GRACE/MP凸包代理ehull，不是新DFT。失败保留在700分母。验证集损失、结构有效、事实读取和性质评分分别记录。\n')]:
        tmp=HERE/(name+'.tmp');tmp.write_text(value);tmp.replace(HERE/name)
    print(result['status'])


if __name__=='__main__':main()
