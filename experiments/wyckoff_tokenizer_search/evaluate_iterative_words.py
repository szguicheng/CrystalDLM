"""Deterministic, constraint-aware adaptation of MaskGXT confidence unmasking.

No target token is supplied to generation. Uses validation only and preserves
the original one-step joint MAP reader exactly at T=1.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from direct_words import collate, make_masks
from joint_decode import exact_fixed_site_map
from packed_words import PackedWordModel


@torch.no_grad()
def generate(model, data, examples, masks, steps):
    batch = collate(examples, torch.device('cuda'))
    del batch['profile_id'], batch['target']
    assert not {'profile_id', 'target'}.intersection(batch)
    shape = (*batch['Z'].shape, model.length)
    words = torch.zeros(shape, dtype=torch.long, device='cuda')
    valid = model.valid_slots(batch)
    hidden = valid.clone()
    initial_counts = valid.flatten(1).sum(-1)
    candidate_ids = model.context_candidates[batch['context']]
    candidate_words = model.profile_words[candidate_ids.clamp_min(0)]
    forward_calls = 0
    for step in range(steps):
        if not hidden.any():
            break
        logits = model(batch, words, hidden)
        forward_calls += 1
        scores = model.profile_scores(batch, logits)
        consistent = ((candidate_words == words.unsqueeze(2)) | hidden.unsqueeze(2)).all(-1)
        consistent &= candidate_ids >= 0
        scores = scores.masked_fill(~consistent, -1e4)
        probabilities = F.log_softmax(scores, -1).cpu().numpy()
        feasible = consistent.cpu().numpy()
        chosen = []
        for i, example in enumerate(examples):
            candidates = []
            for e, context in enumerate(example['context']):
                ids = data['context_candidates'][context].tolist()
                candidates.append([(pid, float(probabilities[i, e, j]), int(masks[pid]))
                    for j, pid in enumerate(ids) if pid >= 0 and feasible[i, e, j]])
            solution = exact_fixed_site_map(candidates)
            assert solution is not None, 'Committed model-generated prefix has no legal continuation.'
            chosen.append(list(solution[1]))
        proposed_ids = torch.tensor([r + [0] * (shape[1] - len(r)) for r in chosen], device='cuda')
        proposed_words = model.profile_words[proposed_ids]
        local = model.local_words[proposed_ids]
        confidence = F.log_softmax(logits, -1).gather(-1, local.unsqueeze(-1)).squeeze(-1)
        confidence = confidence.masked_fill(~hidden, -torch.inf)
        order = confidence.flatten(1).argsort(dim=-1, descending=True, stable=True)
        rank = order.argsort(dim=-1, stable=True)
        remaining = torch.floor(initial_counts * math.cos(.5 * math.pi * (step + 1) / steps)).long()
        if step == steps - 1:
            remaining.zero_()
        reveal_count = (hidden.flatten(1).sum(-1) - remaining).clamp_min(0)
        reveal = (rank < reveal_count.unsqueeze(-1)).reshape_as(hidden) & hidden
        words = torch.where(reveal, proposed_words, words)
        hidden &= ~reveal
    assert not hidden.any()
    assert ((words == proposed_words) | ~valid).all(), 'Final words differ from the final legal proposal.'
    rows = [{'id': record['id'], 'sg': record['sg'], 'predicted_profile_ids': pids,
             'true_profile_ids': record['profile_id'], 'exact': pids == record['profile_id']}
            for record, pids in zip(examples, chosen)]
    return rows, forward_calls


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    start = time.perf_counter()
    data = torch.load('count_merge_candidates/data.pt', map_location='cpu')
    records = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:2048]
    masks = make_masks(data)
    seed = 17
    output = Path('iterative_word_runs/packed_seed17')
    output.mkdir(parents=True, exist_ok=True)
    results, all_rows = {}, {}
    for variant in ['rle', 'rlemerge']:
        root = Path('packed_word_runs') / f'{variant}_seed{seed}'
        receipt = json.loads((root / 'result.json').read_text())
        assert receipt['status'] == 'COMPLETED'
        assert receipt['config']['model_sha256'] == hashlib.sha256(Path('packed_words.py').read_bytes()).hexdigest()
        checkpoint = torch.load(root / 'best.pt', map_location='cpu')
        model = PackedWordModel({k: v for k, v in data.items() if k != 'splits'}, variant).cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        for steps in [1, 4, 8]:
            then = time.perf_counter()
            rows, calls = [], 0
            for i in range(0, len(records), 32):
                part, n = generate(model, data, records[i:i + 32], masks, steps)
                rows.extend(part); calls += n
            exact = sum(r['exact'] for r in rows) / len(rows)
            if steps == 1:
                assert exact == receipt['metrics']['joint_exact_match'], (variant, exact, receipt['metrics'])
            tag = f'{variant}_T{steps}'
            results[tag] = {'exact_match': exact, 'steps_limit': steps, 'batch_forward_calls': calls,
                            'generation_seconds': time.perf_counter() - then,
                            'validity_guaranteed_by_common_candidate_reader': True}
            all_rows[tag] = rows
            (output / f'{tag}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            print(json.dumps({'tag': tag, **results[tag]}), flush=True)
        del model
        torch.cuda.empty_cache()
    comparisons = {}
    for steps in [1, 4, 8]:
        a, b = all_rows[f'rle_T{steps}'], all_rows[f'rlemerge_T{steps}']
        assert [r['id'] for r in a] == [r['id'] for r in b]
        difference = np.array([int(y['exact']) - int(x['exact']) for x, y in zip(a, b)])
        rng = np.random.default_rng(2026)
        boot = [rng.choice(difference, size=len(difference)).mean() for _ in range(2000)]
        comparisons[steps] = {'block_minus_rle_exact': float(difference.mean()),
                              'paired_structure_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    summary = {'results': results, 'comparisons': comparisons, 'structures': len(records), 'training_seed': seed,
               'validation_only': True, 'one_training_seed_only': True,
               'deterministic_no_sampling_noise': True, 'one_step_reader_parity': 'PASS',
               'no_target_fields_given_to_generation': True,
               'remaining_mask_schedule': 'floor(N_initial*cos(pi/2*(k+1)/T)); monotone confidence fixing',
               'proposal_rule': 'Exact fixed-site MAP among canonical candidates consistent with committed model words',
               'not_exact_MaskGXT_or_D3PM': True, 'not_full_geometry_generation': True,
               'no_multistep_marginal_likelihood_claim': True,
               'elapsed_seconds': time.perf_counter() - start, 'job_id': os.environ['SLURM_JOB_ID'],
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
