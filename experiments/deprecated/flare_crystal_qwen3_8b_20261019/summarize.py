import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifests = [json.loads((HERE / arm / 'manifest.json').read_text()) for arm in ['ar','flare']]
    assert all(m['status']=='complete' and m['completed_updates']==64 for m in manifests)
    assert manifests[0]['metadata']==manifests[1]['metadata']
    updates = [[json.loads(s) for s in (HERE / arm / 'updates.jsonl').read_text().splitlines()]
               for arm in ['ar','flare']]
    assert all(len(rows)==64 for rows in updates)
    for i,(a,b) in enumerate(zip(*updates),1):
        assert a['update']==b['update']==i and a['material_ids']==b['material_ids']
        assert all(math.isfinite(r[k]) for r in [a,b] for k in ['loss','gradient_norm'])
    result={'scope':'fixed-reference-lead small learning probe; not end-to-end or property benchmark',
            'matched_initialization_and_data':True,'groups':{},'completed_attempts':0}
    for arm in ['source','ar','flare']:
        assert (HERE/f'{arm}-exit.txt').read_text().strip()=='0'
        out=HERE/'evaluation'/arm
        summary=json.loads((out/'summary.json').read_text())
        rows=[json.loads(p.read_text()) for p in sorted(out.glob('*.json')) if p.name!='summary.json']
        assert len(rows)==(16 if arm=='ar' else 32)
        assert all(r['state_hash']==summary['state_hash'] and r['reference_lead'] for r in rows)
        for key,group in summary['results'].items():
            split,mode=key.split('-')
            selected=[r for r in rows if r['split']==split and r['mode']==mode]
            assert len(selected)==8 and len({r['material_id'] for r in selected})==8
            assert {r['material_id'] for r in selected}==set(manifests[0]['metadata'][
                'train_material_ids' if split=='train' else 'validation_material_ids'])
            assert group['valid']==sum(r['valid'] for r in selected)
            result['groups'][f'{arm}/{key}']=group
        result['completed_attempts']+=len(rows)
    assert result['completed_attempts']==80
    result['flare_train_validity_gate']=result['groups']['flare/train-diffusion']['valid']>0
    result['property_evaluation']='not_run'
    (HERE/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
