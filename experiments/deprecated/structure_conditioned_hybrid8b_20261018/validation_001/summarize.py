"""Require all five diagnostic conditions and verify the paired training contract."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ['source-original', 'source-confidence', 'source-remask',
              'control-original', 'highmask-original']


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    slots = {(r['request_id'], r['attempt_id']): r for r in plan['requests']}
    assert len(slots) == 28
    plan_hash = hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
    result = dict(scope='fixed-prefix 28-candidate diagnostic; no property scoring', conditions={})
    outcomes = {}
    for condition in CONDITIONS:
        path = HERE / 'evaluation' / condition
        summary = json.loads((path / 'summary.json').read_text())
        rows = [json.loads(p.read_text()) for p in sorted(path.glob('sg-*.json'))]
        assert len(rows) == 28
        assert {(r['request_id'], r['attempt_id']) for r in rows} == set(slots)
        assert summary['identity']['plan_sha256'] == plan_hash
        errors = collections.Counter()
        for r in rows:
            assert r['identity'] == summary['identity']
            assert r['seed'] == slots[r['request_id'], r['attempt_id']]['seed']
            assert len(r['raw_tokens']) == 544
            if not r['boundary_ok']:
                errors[r['error']] += 1
            elif not r['structural_valid']:
                errors.update(r['audit']['reasons'])
        assert summary['valid'] == sum(r['structural_valid'] for r in rows)
        result['conditions'][condition] = dict(summary, failure_reasons=dict(errors))
        outcomes[condition] = {(r['request_id'], r['attempt_id']): r['structural_valid'] for r in rows}
    source_hash = result['conditions']['source-original']['identity']['state_sha256']
    assert all(result['conditions']['source-' + name]['identity']['state_sha256'] == source_hash
               for name in ['confidence', 'remask'])
    assert len({source_hash, *(result['conditions'][arm + '-original']['identity']['state_sha256']
                              for arm in ['control', 'highmask'])}) == 3
    control = [json.loads(s) for s in (HERE / 'control/updates.jsonl').read_text().splitlines()]
    highmask = [json.loads(s) for s in (HERE / 'highmask/updates.jsonl').read_text().splitlines()]
    assert len(control) == len(highmask) == 128
    for index, (a, b) in enumerate(zip(control, highmask), 1):
        assert a['update'] == b['update'] == index
        assert a['material_ids'] == b['material_ids']
        assert a['learning_rate'] == b['learning_rate'] == 5e-5
        assert a['finite_gradients'] and b['finite_gradients']
        for x, y in zip(a['noise_draws'], b['noise_draws']):
            assert (x['uniform_k'], x['branch_u']) == (y['uniform_k'], y['branch_u'])
            assert x['k'] == x['uniform_k']
            assert y['k'] == (32 if y['branch_u'] < .5 else y['uniform_k'])
    manifests = [json.loads((HERE / arm / 'manifest.json').read_text()) for arm in ['control', 'highmask']]
    assert all(m['status'] == 'complete' and m['completed_updates'] == 128 for m in manifests)
    assert manifests[0]['metadata']['source_trainable_sha256'] == manifests[1]['metadata']['source_trainable_sha256']
    assert manifests[0]['metadata']['ordered_material_ids'] == manifests[1]['metadata']['ordered_material_ids']
    result['training_pair'] = dict(updates_each=128, materials_each=1024, matched=True,
        full_mask_draws={arm: sum(d['k'] == 32 for r in rows for d in r['noise_draws'])
                        for arm, rows in [('control', control), ('highmask', highmask)]})
    result['paired_validity'] = {}
    for comparator, intervention in [('source-original', 'source-confidence'),
                                    ('source-confidence', 'source-remask'),
                                    ('source-original', 'control-original'),
                                    ('control-original', 'highmask-original')]:
        a, b = outcomes[comparator], outcomes[intervention]
        result['paired_validity'][intervention + ' vs ' + comparator] = dict(
            gains=sum(not a[k] and b[k] for k in slots), losses=sum(a[k] and not b[k] for k in slots))
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {x: v[x] for x in ['valid', 'boundary_pass', 'valid_requested_sg']}
                      for k, v in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
