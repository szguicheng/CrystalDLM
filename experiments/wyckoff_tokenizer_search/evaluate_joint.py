"""Evaluate identical exact fixed-site exclusion decoding for both pilot models."""
import itertools
import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_decode import exact_fixed_site_map
from train_profile_model import ProfileModel, collate


def make_masks(data):
    pcounts = data['profile_counts'].numpy()
    wpids = data['profile_wp_ids'].numpy()
    groups = data['profile_sg'].numpy()
    meta = data['wp_metadata'].numpy()
    masks = np.zeros(len(pcounts), dtype=np.int64)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        row = wpids[ids[0]]
        fixed = sorted(w for w in row if w and meta[w, 1] == 0)
        assert len(fixed) <= 8
        mapping = {w: 1 << i for i, w in enumerate(fixed)}
        bits = np.array([mapping.get(w, 0) for w in row])
        masks[ids] = ((pcounts[ids] > 0) * bits).sum(1)
    return masks


def validate_decoder():
    rng = np.random.default_rng(123)
    for _ in range(100):
        candidates = [[(e * 5 + i, float(rng.normal()), int(rng.integers(16))) for i in range(5)]
                      for e in range(3)]
        brute = []
        for choices in itertools.product(*candidates):
            used = 0
            for _, _, mask in choices:
                if used & mask:
                    break
                used |= mask
            else:
                brute.append(sum(x[1] for x in choices))
        result = exact_fixed_site_map(candidates)
        if brute:
            assert result is not None and abs(result[0] - max(brute)) < 1e-8
        else:
            assert result is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--words', action='store_true')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    validate_decoder()
    root = Path('model_runs')
    data = torch.load('word_candidates/data.pt' if args.words else 'candidates_cap256/data.pt', map_location='cpu')
    masks = make_masks(data)
    results = {}
    variants = ['unigram', 'rle', 'setmerge'] if args.words else ['factorized', 'compositional']
    for variant in variants:
        start = time.perf_counter()
        checkpoint = torch.load(root / f'{variant}_seed{args.seed}/best.pt', map_location='cpu')
        config = checkpoint['config']
        import hashlib
        records = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:config['val_limit']]
        if args.words:
            from word_model import WordModel
            model = WordModel(data, variant, config['width'], config['layers']).cuda()
        else:
            model = ProfileModel(data, variant, config['width'], config['layers']).cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        outputs = []
        for start_index in range(0, len(records), config['batch_size']):
            examples = records[start_index:start_index + config['batch_size']]
            batch = collate(examples, torch.device('cuda'))
            with torch.no_grad():
                logits, allowed = model(batch, batch['Z'] > 0)
                scores = F.log_softmax(logits, dim=-1).cpu().numpy()
            for i, record in enumerate(examples):
                lists = []
                for e, context in enumerate(record['context']):
                    ids = data['context_candidates'][context].tolist()
                    lists.append([(pid, float(scores[i, e, j]), int(masks[pid]))
                                  for j, pid in enumerate(ids) if pid >= 0])
                result = exact_fixed_site_map(lists)
                assert result is not None, 'Ground truth was feasible but decoder found no solution.'
                predictions = list(result[1])
                exact = predictions == record['profile_id']
                outputs.append({'id': record['id'], 'sg': record['sg'], 'exact': exact,
                                'predicted_profile_ids': predictions, 'true_profile_ids': record['profile_id'],
                                'correct_elements': sum(a == b for a, b in zip(predictions, record['profile_id'])),
                                'elements': len(predictions)})
        metrics = {'variant': variant, 'structures': len(outputs),
                   'joint_exact_match': sum(x['exact'] for x in outputs) / len(outputs),
                   'joint_element_accuracy': sum(x['correct_elements'] for x in outputs) / sum(x['elements'] for x in outputs),
                   'elapsed_seconds': time.perf_counter() - start,
                   'source_checkpoint_step': checkpoint['step'], 'decoder': 'exact fixed-site bitmask MAP',
                   'not_geometric_validity': True, 'validation_only': True,
                   'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
                   'brute_force_decoder_fixtures_passed': 100}
        target = root / f'{variant}_seed{args.seed}'
        (target / 'joint_validation.json').write_text(json.dumps(metrics, indent=2))
        with (target / 'joint_predictions.jsonl').open('w') as stream:
            for record in outputs:
                stream.write(json.dumps(record) + '\n')
        results[variant] = outputs
        print(json.dumps(metrics), flush=True)
        del model
        torch.cuda.empty_cache()
    first, second = ('rle', 'setmerge') if args.words else ('factorized', 'compositional')
    a, b = results[first], results[second]
    assert [r['id'] for r in a] == [r['id'] for r in b]
    differences = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
    rng = np.random.default_rng(2026)
    bootstrap = np.array([rng.choice(differences, len(differences)).mean() for _ in range(2000)])
    comparison = {f'{second}_minus_{first}_joint_exact': float(differences.mean()),
                  'paired_structure_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
                  'resamples': 2000, 'single_training_seed_only': True, 'validation_development_result': True,
                  'training_seed_uncertainty_not_estimated': True}
    name = f'paired_{"word_" if args.words else ""}joint_validation_seed{args.seed}.json'
    (root / name).write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
