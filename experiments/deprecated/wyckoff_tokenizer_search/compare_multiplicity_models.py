"""Compare the hierarchy with both matched controls on development validation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    root = Path('multiplicity_model_runs')
    models = {}
    for mode in ['flat', 'degeneracy', 'hierarchical']:
        folder = root / f'{mode}_seed{args.seed}'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETED'
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        models[mode] = result, rows
    results = {}
    reference, b = models['hierarchical']
    for baseline in ['flat', 'degeneracy']:
        model, a = models[baseline]
        for key in ['parameter_count', 'train_structures', 'validation_structures', 'data_sha256',
                    'source_sha256', 'model_sha256', 'steps', 'batch_size', 'seed']:
            assert model['config'][key] == reference['config'][key], key
        assert [r['id'] for r in a] == [r['id'] for r in b]
        assert [r['true_profile_ids'] for r in a] == [r['true_profile_ids'] for r in b]
        exact = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
        nll = np.array([y['profile_nll'] - x['profile_nll'] for x, y in zip(a, b)])
        elements = np.array([r['elements'] for r in a])
        rng = np.random.default_rng(2026)
        ci_exact, ci_nll = [], []
        for _ in range(2000):
            ids = rng.integers(len(a), size=len(a))
            ci_exact.append(exact[ids].mean())
            ci_nll.append(nll[ids].sum() / elements[ids].sum())
        results[f'hierarchical_minus_{baseline}'] = {
            'exact_delta': float(exact.mean()), 'nll_delta': float(nll.sum() / elements.sum()),
            'paired_structure_exact_95ci': np.quantile(ci_exact, [.025, .975]).tolist(),
            'paired_structure_nll_95ci': np.quantile(ci_nll, [.025, .975]).tolist()}
    summary = {'seed': args.seed, 'results': {k: r['metrics'] for k, (r, rows) in models.items()},
               'comparisons': results, 'single_training_seed_only': True, 'validation_only': True,
               'training_seconds': {k: r['total_seconds'] for k, (r, rows) in models.items()},
               'parameter_count': reference['config']['parameter_count'], 'job_id': os.environ['SLURM_JOB_ID']}
    (root / f'comparisons_seed{args.seed}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
