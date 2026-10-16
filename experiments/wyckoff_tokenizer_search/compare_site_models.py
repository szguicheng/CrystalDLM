"""Paired development comparison of matched site aggregation variants."""
import json
import os
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root = Path('site_inventory_model_runs')
    values = {}
    for mode in ['aggregate', 'records']:
        folder = root / f'{mode}_seed17'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETED'
        rows = [json.loads(x) for x in (folder / 'predictions.jsonl').read_text().splitlines()]
        values[mode] = result, rows
    candidate, b = values['aggregate']
    control, a = values['records']
    for key in ['parameter_count', 'train_structures', 'validation_structures', 'data_sha256',
                'model_sha256', 'feature_model_sha256', 'source_sha256', 'metadata_sha256',
                'steps', 'batch_size', 'seed', 'eval_every']:
        assert candidate['config'][key] == control['config'][key], key
    assert [r['id'] for r in a] == [r['id'] for r in b]
    assert [r['true_profile_ids'] for r in a] == [r['true_profile_ids'] for r in b]
    assert [r['elements'] for r in a] == [r['elements'] for r in b]
    exact = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
    nll = np.array([y['profile_nll'] - x['profile_nll'] for x, y in zip(a, b)])
    elements = np.array([r['elements'] for r in a])
    rng = np.random.default_rng(2026)
    ci_exact, ci_nll = [], []
    for _ in range(2000):
        ids = rng.integers(len(a), size=len(a))
        ci_exact.append(exact[ids].mean())
        ci_nll.append(nll[ids].sum() / elements[ids].sum())
    delta_exact = float(exact.mean())
    delta_nll = float(nll.sum() / elements.sum())
    relative_nll = -delta_nll / control['metrics']['mean_profile_nll']
    advance = (delta_exact >= .01 or relative_nll >= .02) and delta_exact >= 0 and delta_nll <= 0
    summary = {'results': {k: r['metrics'] for k, (r, _) in values.items()},
               'aggregate_minus_records_exact': delta_exact,
               'aggregate_minus_records_nll': delta_nll,
               'relative_nll_reduction': relative_nll,
               'paired_structure_exact_95ci': np.quantile(ci_exact, [.025, .975]).tolist(),
               'paired_structure_nll_95ci': np.quantile(ci_nll, [.025, .975]).tolist(),
               'time_seconds': {k: r['total_seconds'] for k, (r, _) in values.items()},
               'peak_cuda_bytes': {k: r['peak_cuda_memory_bytes'] for k, (r, _) in values.items()},
               'same_parameter_and_source_audit': 'PASS', 'single_training_seed_only': True,
               'validation_only': True, 'does_not_isolate_end_to_end_diffusion_tokenizer': True,
               'advance_under_prefixed_protocol': bool(advance), 'job_id': os.environ['SLURM_JOB_ID']}
    (root / 'paired_seed17.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
