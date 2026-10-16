"""Uniform post-run scoring repair for explicitly named a,b,c prose."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from prepare import sha, write_json
from score import grade_answer


def reading_grade(response,text):
    # Same axis order, numerical tolerance and contradiction checks as the original scorer.
    normalized=re.sub(r'\ba\s*,\s*b\s*,?\s*(?:and\s+)?c\s+are\b',
                      'lattice lengths are',response,flags=re.I)
    return grade_answer(normalized,text)


def grade(response,text):
    return reading_grade(response['value'],text) if response.get('ok') else dict(success=False,fields={})


def score_branch(folder,materials,tests):
    rows=[json.loads(l) for l in (folder/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==8
    scored=[]
    for row in rows:
        r=dict(row);source=materials[r['id']]['c1']
        r['reference_read_grade_v2']=grade(r['reference_read'],source)
        r['Q_reference']=r['reference_read_grade_v2']['success']
        r['swap_read_grade_v2']=grade(r['swap_read'],materials[r['swap_id']]['c1'])
        r['Q_swap']=r['swap_read_grade_v2']['success']
        r['no_crystal_grade_v2']=grade(r['no_crystal_read'],materials[r['swap_id']]['c1'])
        r['Q_no_crystal']=r['no_crystal_grade_v2']['success']
        r['closed_loop']=False
        if r['generation'].get('ok') and 'actual_read' in r:
            r['actual_read_grade_v2']=grade(r['actual_read'],r['generation']['value']['crystal'])
            if r.get('actual_edit',{}).get('ok') and 'actual_second_read' in r:
                r['actual_second_read_grade_v2']=grade(r['actual_second_read'],r['actual_edit']['value']['crystal'])
                r['closed_loop']=bool(r['actual_read_grade_v2']['success'] and
                     r.get('actual_edit_grade',{}).get('success') and r['actual_second_read_grade_v2']['success'])
        scored.append(r)
    keys=['G','Q_reference','E_reference','closed_loop','flow_complete','Q_swap','Q_no_crystal']
    summary=dict(n=8,counts={k:sum(r[k] for r in scored) for k in keys})
    conflicts=[json.loads(l) for l in (folder/'conflict_evaluation.jsonl').read_text().splitlines()]
    assert len(conflicts)==64
    indexed={(c['old_id'],c['current_id'],c['scale']):c for c in tests}
    for r in conflicts:
        expected=indexed[r['old_id'],r['current_id'],r['scale']]['crystal']
        r['grade_v2']=grade(r['response'],expected)
    summary['conflicts']={}
    for scale in ['1','1.03']:
        subset=[r for r in conflicts if r['scale']==scale];assert len(subset)==32
        fields=Counter()
        for r in subset:fields.update({k:int(v) for k,v in r['grade_v2']['fields'].items()})
        summary['conflicts'][scale]=dict(n=32,success=sum(r['grade_v2']['success'] for r in subset),correct_by_field=dict(fields))
    write_json(folder/'rescored_summary.json',summary)
    with (folder/'rescored_reading.jsonl').open('w') as f:
        for r in scored:f.write(json.dumps(r)+'\n')
    with (folder/'rescored_conflicts.jsonl').open('w') as f:
        for r in conflicts:f.write(json.dumps(r)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    original=json.loads((args.root/'summary.json').read_text());assert original['status']=='complete'
    materials={r['id']:r for r in map(json.loads,(args.data/'train.jsonl').read_text().splitlines())}
    tests=[json.loads(l) for l in (args.root/'conflict_test.jsonl').read_text().splitlines()]
    results={};source_hashes={}
    for mode in ['ar','hybrid']:
        results[mode]={}
        for branch in ['consistent','feedback']:
            folder=args.root/mode/branch
            for name in ['evaluation.jsonl','conflict_evaluation.jsonl']:
                source_hashes[f'{mode}/{branch}/{name}']=sha((folder/name).read_bytes())
            results[mode][branch]=score_branch(folder,materials,tests)
    write_json(args.root/'rescored_report.json',dict(status='complete',scorer='explicit named-axis prose v2',
         scorer_sha256=sha(Path(__file__).read_bytes()),original_scorer_sha256=sha((Path(__file__).parent/'score.py').read_bytes()),
         raw_output_hashes=source_hashes,results=results,
         note='All branches rescored uniformly. Only a,b,c phrase recognition added; raw generations, old scores and numerical tolerances unchanged.'))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
