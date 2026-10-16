"""Training-frequency sanity baselines, same conditions and legal MAP readout."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from evaluate_joint import make_masks
from joint_decode import exact_fixed_site_map


def main():
    assert os.environ.get('SLURM_JOB_ID')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    start = time.perf_counter()
    data = torch.load('count_merge_candidates/data.pt', map_location='cpu')
    masks = make_masks(data)
    counts, chemical_counts = defaultdict(Counter), defaultdict(Counter)
    for record in data['splits']['train']:
        for context, z, pid in zip(record['context'], record['Z'], record['profile_id']):
            counts[context][pid] += 1
            chemical_counts[context, z][pid] += 1
    records = sorted(data['splits'][args.split], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    if args.split == 'val':
        records = records[:2048]
    output = Path('frequency_baselines')
    if args.split == 'test':
        output = output / 'test'
    output.mkdir(exist_ok=True)
    results = {}
    for name in ['context', 'context_element']:
        cache, rows = {}, []
        for record in records:
            choices, nll = [], 0.
            for context, z, true in zip(record['context'], record['Z'], record['profile_id']):
                key = (context, z if name == 'context_element' else None)
                if key not in cache:
                    ids = [pid for pid in data['context_candidates'][context].tolist() if pid >= 0]
                    total = sum(counts[context].values())
                    prior = {pid: (counts[context][pid] + 1 / len(ids)) / (total + 1) for pid in ids}
                    local = chemical_counts[context, z] if name == 'context_element' else Counter()
                    denominator = sum(local.values()) + 1
                    logp = {pid: math.log((local[pid] + prior[pid]) / denominator) for pid in ids}
                    cache[key] = logp
                logp = cache[key]
                nll -= logp[true]
                choices.append([(pid, value, int(masks[pid])) for pid, value in logp.items()])
            solution = exact_fixed_site_map(choices)
            assert solution is not None
            predicted = list(solution[1])
            rows.append({'id': record['id'], 'exact': predicted == record['profile_id'], 'profile_nll': nll,
                         'elements': len(predicted), 'predicted_profile_ids': predicted, 'true_profile_ids': record['profile_id']})
        results[name] = {'joint_exact_match': sum(r['exact'] for r in rows) / len(rows),
                         'mean_profile_nll': sum(r['profile_nll'] for r in rows) / sum(r['elements'] for r in rows)}
        (output / f'{name}_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = {'results': results, 'split': args.split, 'train_structures': len(data['splits']['train']), 'evaluation_structures': len(records),
              'smoothing': 'Uniform Dirichlet mass 1 per context; chemical backoff to context prior with mass 1',
              'no_validation_or_test_fit': True, 'same_exact_fixed_site_map': True,
              'elapsed_seconds': time.perf_counter() - start, 'job_id': os.environ['SLURM_JOB_ID']}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
