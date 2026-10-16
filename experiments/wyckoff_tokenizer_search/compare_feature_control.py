"""Paired validation comparison with explicitly audited reuse of the flat run."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root = Path('multiplicity_model_runs')
    values = {}
    for mode in ['flat', 'unaggregated']:
        folder = root / f'{mode}_seed17'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETED'
        rows = [json.loads(x) for x in (folder / 'predictions.jsonl').read_text().splitlines()]
        values[mode] = result, rows
    candidate, b = values['flat']
    control, a = values['unaggregated']
    for key in ['parameter_count', 'train_structures', 'validation_structures', 'data_sha256',
                'model_sha256', 'steps', 'batch_size', 'seed', 'eval_every']:
        assert candidate['config'][key] == control['config'][key], key
    receipt = json.loads(Path('feature_control_source_receipt.json').read_text())
    assert candidate['config']['source_sha256'] == receipt['frozen_trainer_sha256']
    assert control['config']['source_sha256'] == receipt['control_trainer_sha256']
    assert control['config']['frozen_trainer_sha256'] == candidate['config']['source_sha256']
    def loop(path):
        text = Path(path).read_text()
        return text[text.index('    for step in range(1, args.steps + 1):'):text.index("    result = {'status': 'COMPLETED'")]
    original, new = loop('train_multiplicity_model.py'), loop('train_unaggregated_model.py')
    assert original == new
    assert hashlib.sha256(original.encode()).hexdigest() == receipt['loop_sha256']
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
    summary = {'results': {k: r['metrics'] for k, (r, _) in values.items()},
               'aggregated_minus_unaggregated_exact': float(exact.mean()),
               'aggregated_minus_unaggregated_nll': float(nll.sum() / elements.sum()),
               'paired_structure_exact_95ci': np.quantile(ci_exact, [.025, .975]).tolist(),
               'paired_structure_nll_95ci': np.quantile(ci_nll, [.025, .975]).tolist(),
               'same_constructor_and_loop_audit': 'PASS', 'single_training_seed_only': True,
               'validation_only': True, 'previous_flat_result_reused_with_source_audit': True,
               'does_not_isolate_end_to_end_diffusion_tokenizer': True, 'job_id': os.environ['SLURM_JOB_ID']}
    (root / 'feature_control_seed17.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
