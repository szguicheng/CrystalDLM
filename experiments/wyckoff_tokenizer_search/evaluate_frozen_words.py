"""One frozen multi-seed evaluation; never fits on the requested evaluation split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from direct_words import DirectWordModel, evaluate
from evaluate_joint import make_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    start = time.perf_counter()
    seeds = [17, 29, 43]
    root = Path('compact_10k_runs')
    source_hash = hashlib.sha256(Path('direct_words.py').read_bytes()).hexdigest()
    data_path = Path('count_merge_candidates/data.pt')
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    # Check every frozen training receipt before touching evaluation labels.
    training = {}
    for seed in seeds:
        for variant in ['rle', 'rlemerge']:
            folder = root / f'{variant}_seed{seed}'
            result = json.loads((folder / 'result.json').read_text())
            c = result['config']
            assert result['status'] == 'COMPLETED' and c['steps'] == 10000
            assert c['source_sha256'] == source_hash and c['data_sha256'] == data_hash
            assert c['compact'] and c['parameter_count'] == 499776
            training[seed, variant] = result
    data = torch.load(data_path, map_location='cpu')
    masks = make_masks(data)
    records = sorted(data['splits'][args.split], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    if args.split == 'val':
        records = records[:2048]
    output = Path('frozen_word_evaluation') / args.split
    output.mkdir(parents=True, exist_ok=True)
    results, predictions = {}, {}
    for seed in seeds:
        for variant in ['rle', 'rlemerge']:
            tag = f'{variant}_seed{seed}'
            checkpoint = torch.load(root / tag / 'best.pt', map_location='cpu')
            model = DirectWordModel(data, variant, compact=True).cuda()
            model.load_state_dict(checkpoint['model'])
            metrics, rows = evaluate(model, data, records, 32, masks)
            if args.split == 'val':
                for metric in metrics:
                    assert abs(metrics[metric] - training[seed, variant]['metrics'][metric]) < 1e-6
            results[tag] = metrics
            predictions[seed, variant] = rows
            (output / f'{tag}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            print(json.dumps({'split': args.split, 'tag': tag, 'metrics': metrics}), flush=True)
            del model
            torch.cuda.empty_cache()
    exact, nll, seed_summaries = [], [], []
    for seed in seeds:
        a, b = predictions[seed, 'rle'], predictions[seed, 'rlemerge']
        assert [r['id'] for r in a] == [r['id'] for r in b]
        assert [r['true_profile_ids'] for r in a] == [r['true_profile_ids'] for r in b]
        d = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
        loss = np.array([y['profile_nll'] - x['profile_nll'] for x, y in zip(a, b)])
        elements = np.array([r['elements'] for r in a])
        exact.append(d); nll.append(loss)
        seed_summaries.append({'seed': seed, 'exact_delta': float(d.mean()), 'nll_delta': float(loss.sum() / elements.sum())})
    exact, nll = np.stack(exact), np.stack(nll)
    rng = np.random.default_rng(2026)
    ci_exact, ci_nll = [], []
    for _ in range(2000):
        s = rng.integers(len(seeds), size=len(seeds))
        ids = rng.integers(len(records), size=len(records))
        ci_exact.append(exact[s][:, ids].mean())
        ci_nll.append(nll[s][:, ids].sum() / (len(seeds) * elements[ids].sum()))
    means = {variant: {metric: float(np.mean([results[f'{variant}_seed{seed}'][metric] for seed in seeds]))
                       for metric in ['joint_exact_match', 'mean_profile_nll', 'element_accuracy', 'raw_word_sequence_exact']}
             for variant in ['rle', 'rlemerge']}
    summary = {'split': args.split, 'structures': len(records), 'seeds': seeds, 'results': results, 'means': means,
               'seed_deltas': seed_summaries, 'mean_exact_delta': float(exact.mean()),
               'mean_profile_nll_delta': float(nll.sum() / (len(seeds) * elements.sum())),
               'hierarchical_bootstrap_exact_95ci': np.quantile(ci_exact, [.025, .975]).tolist(),
               'hierarchical_bootstrap_nll_95ci': np.quantile(ci_nll, [.025, .975]).tolist(),
               'bootstrap_scope': 'Resample 3 paired training seeds and matched structures; only 3 observed initializations',
               'checkpoint_rule': 'Best validation profile NLL among steps 1000..10000; frozen before test',
               'mean_training_seconds': {v: float(np.mean([training[s, v]['total_seconds'] for s in seeds])) for v in ['rle', 'rlemerge']},
               'peak_training_cuda_bytes': {v: [training[s, v]['peak_cuda_memory_bytes'] for s in seeds] for v in ['rle', 'rlemerge']},
               'data_sha256': data_hash, 'source_sha256': source_hash,
               'elapsed_seconds': time.perf_counter() - start, 'job_id': os.environ['SLURM_JOB_ID'],
               'no_geometry_or_material_stability_claim': True, 'candidate_cap256_model_scope': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
