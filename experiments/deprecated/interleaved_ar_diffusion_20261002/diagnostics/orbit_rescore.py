"""Uniform post-hoc v3 orbit auditing; retain original outputs and report unexecuted flows."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare import edit_factor, sha, write_json
from rescore import reading_grade
from score import audit, grade_edit


def main():
    materials={r['id']:r for split in ['train','val'] for r in map(json.loads,(ROOT/'data_002'/f'{split}.jsonl').read_text().splitlines())}
    report={};hashes={};pending=[]
    for experiment in ['cohort_001','recovery_001']:
        for file in sorted((ROOT/experiment).rglob('evaluation.jsonl')):
            folder=file.parent;summary_file=folder/'evaluation_summary.json'
            if not summary_file.exists():pending.append(str(folder.relative_to(ROOT)));continue
            rows=[json.loads(l) for l in file.read_text().splitlines()]
            assert len(rows)==json.loads(summary_file.read_text())['n']
            results=[];counts=Counter()
            for r in rows:
                source=materials[r['id']]['c1'];scale=edit_factor(r['direction']);g=r['generation'];e=r['reference_edit']
                checked=audit(g['value']['crystal'],materials[r['id']]['condition']) if g['ok'] else dict(valid=False)
                graded=grade_edit(e['value']['crystal'],source,scale) if e['ok'] else dict(success=False)
                measured=False;needs_flow=False
                if checked['valid']:
                    if 'actual_read' not in r:needs_flow=True
                    elif (r['actual_read']['ok'] and r.get('actual_edit',{}).get('ok') and r.get('actual_second_read',{}).get('ok')):
                        c1=g['value']['crystal'];c2=r['actual_edit']['value']['crystal']
                        measured=bool(reading_grade(r['actual_read']['value'],c1)['success'] and
                            grade_edit(c2,c1,scale)['success'] and reading_grade(r['actual_second_read']['value'],c2)['success'])
                counts.update(G=int(checked['valid']),E_reference=int(graded['success']),
                    newly_valid_G=int(checked['valid'] and not r['G']),closed_loop_measured_lower_bound=int(measured),
                    new_G_with_unexecuted_flow=int(needs_flow))
                results.append(dict(id=r['id'],generation_audit_v3=checked,reference_edit_grade_v3=graded,
                    closed_loop_measured=measured,new_G_with_unexecuted_flow=needs_flow))
            key=str(folder.relative_to(ROOT));hashes[key]=sha(file.read_bytes())
            report[key]=dict(n=len(rows),counts=dict(counts))
            write_json(folder/'orbit_summary_v3.json',report[key])
            (folder/'orbit_audits_v3.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
            print(json.dumps(dict(folder=key,**report[key])),flush=True)
    for probe in sorted(ROOT.glob('order_probe_*')):
        if not (probe/'summary.json').exists():continue
        rows=[json.loads(l) for l in (probe/'raw.jsonl').read_text().splitlines()];groups={};results=[]
        for r in rows:
            result=r['result'];boundary=bool(result['ok'] and result['value']['boundary_valid'])
            checked=audit(result['value']['crystal'],materials[r['id']]['condition']) if boundary else dict(valid=False)
            key=f"{r['split']}/{r['policy']}";groups.setdefault(key,dict(n=0,valid=0,boundary_valid=0,newly_valid=0))
            g=groups[key];g['n']+=1;g['valid']+=int(checked['valid']);g['boundary_valid']+=int(boundary)
            g['newly_valid']+=int(checked['valid'] and not r['audit']['valid'])
            results.append(dict(id=r['id'],split=r['split'],policy=r['policy'],seed=r['seed'],audit_v3=checked))
        report[probe.name]=groups;hashes[probe.name]=sha((probe/'raw.jsonl').read_bytes())
        write_json(probe/'orbit_summary_v3.json',groups)
        (probe/'orbit_audits_v3.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
        print(json.dumps(dict(folder=probe.name,groups=groups)),flush=True)
    write_json(ROOT/'orbit_rescore_v3_report.json',dict(status='complete_for_available_outputs',results=report,
        raw_sha256=hashes,scorer_sha256=sha((ROOT/'score.py').read_bytes()),pending_evaluations=pending,
        note='Same tolerance and all other validity/edit requirements. Orbit membership accepts equivalent representatives, not coordinate repair. New valid G with missing actual flows is not credited as a completed loop. Test unused.'))


if __name__=='__main__':main()
