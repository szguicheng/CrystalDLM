"""Paired validation evidence for the direct word experiment, on a Slurm node."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--root', type=Path, default=Path('direct_word_runs'))
    parser.add_argument('--method', default='setmerge')
    args = parser.parse_args()
    root = args.root
    values = {}
    for variant in ['rle', args.method]:
        folder = root / f'{variant}_seed{args.seed}'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETED'
        records = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        values[variant] = (result, records)
    ar, a = values['rle']; br, b = values[args.method]
    for key in ['parameter_count', 'train_structures',
                'validation_structures', 'data_sha256', 'source_sha256', 'steps', 'batch_size', 'seed']:
        assert ar['config'][key] == br['config'][key], key
    assert [x['id'] for x in a] == [x['id'] for x in b]
    assert ar['config'].get('position_parameter_slots') == br['config'].get('position_parameter_slots')
    assert [x['true_profile_ids'] for x in a] == [x['true_profile_ids'] for x in b]
    diff = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
    nll_diff = np.array([y['profile_nll'] - x['profile_nll'] for x, y in zip(a, b)])
    elements = np.array([x['elements'] for x in a])
    rng = np.random.default_rng(2026)
    accuracy_ci, nll_ci = [], []
    for _ in range(2000):
        ids = rng.integers(len(a), size=len(a))
        accuracy_ci.append(diff[ids].mean())
        nll_ci.append(nll_diff[ids].sum() / elements[ids].sum())
    comparison = {'seed': args.seed, 'structures': len(a), 'results': {k: r['metrics'] for k, (r, _) in values.items()},
                  f'{args.method}_minus_rle_exact': float(diff.mean()),
                  'paired_structure_bootstrap_exact_95ci': np.quantile(accuracy_ci, [.025, .975]).tolist(),
                  f'{args.method}_minus_rle_profile_nll': float(nll_diff.sum() / elements.sum()),
                  'paired_structure_bootstrap_nll_95ci': np.quantile(nll_ci, [.025, .975]).tolist(),
                  f'{args.method}_only_correct': int((diff > 0).sum()), 'rle_only_correct': int((diff < 0).sum()),
                  'single_training_seed_only': True, 'validation_only': True, 'resamples': 2000,
                  'job_id': os.environ['SLURM_JOB_ID']}
    comparison['training_seconds'] = {k: r['total_seconds'] for k, (r, _) in values.items()}
    comparison['peak_cuda_memory_bytes'] = {k: r['peak_cuda_memory_bytes'] for k, (r, _) in values.items()}
    (root / f'paired_seed{args.seed}.json').write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
