"""Validation-only sampler ablation. Original model and test artifacts are immutable."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from model import ROOT, MAXLEN, Denoiser, load_data, render, sample, seed_all

OUT = ROOT / 'followup'


def load_fresh(representation):
    records = [json.loads(line) for line in (OUT / 'fresh_records.jsonl').read_text().splitlines()]
    records = [r for r in records if r['status'] == 'ok']
    vocab = json.loads((ROOT / 'data/vocab.json').read_text())
    lookup = {t: i for i, t in enumerate(vocab)}
    ids = np.zeros((len(records), MAXLEN), dtype=np.int64)
    values = np.zeros_like(ids, dtype=np.float32)
    for i, record in enumerate(records):
        packed = record['packed'][representation]; n = len(packed['tokens'])
        ids[i, :n] = [lookup.get(t, 2) for t in packed['tokens']]
        values[i, :n] = packed['values']
    composition = np.log1p(np.array([r['composition'] for r in records], dtype=np.float32))
    return tuple(torch.tensor(x, device='cuda') for x in (ids, values, composition)), records, vocab


@torch.no_grad()
def variant(model, composition, mode, steps=32):
    if mode == 'one_step': return sample(model, composition, steps=1)
    if mode == 'confidence': return sample(model, composition, steps=steps)
    model.eval()
    ids = torch.full((len(composition), MAXLEN), 1, device=composition.device, dtype=torch.long)
    values = torch.zeros_like(ids, dtype=torch.float)
    for step in range(steps):
        hidden = ids == 1
        rate = hidden.float().mean(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16): logits, numbers = model(ids, values, composition, rate)
        logits[:, :, 1:3] = -torch.inf
        confidence, proposed = logits.float().softmax(-1).max(-1)
        remaining = int(MAXLEN * math.cos((step + 1) / steps * math.pi / 2))
        if mode == 'remask':
            ids[hidden] = proposed[hidden]
            values[hidden] = numbers.float()[hidden]
            if remaining:
                positions = torch.rand(ids.shape, device=ids.device).topk(remaining, dim=1).indices
                ids.scatter_(1, positions, 1); values.scatter_(1, positions, 0)
        else:
            if mode == 'random': confidence = torch.rand(ids.shape, device=ids.device)
            elif mode == 'content_first': confidence = confidence - (proposed == 0).float() * 2
            else: raise ValueError(mode)
            confidence.masked_fill_(~hidden, -torch.inf)
            reveal = int(hidden[0].sum()) - remaining
            positions = confidence.topk(reveal, dim=1).indices
            ids.scatter_(1, positions, proposed.gather(1, positions))
            values.scatter_(1, positions, numbers.float().gather(1, positions))
        values.masked_fill_(~model.numeric(ids), 0)
    assert not (ids == 1).any()
    return ids, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--modes', nargs='+', default=['one_step', 'confidence', 'random', 'content_first', 'remask'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--split', choices=['val', 'test', 'fresh'], default='val')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--label', default='sampler_v1')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'codec'))
    import wyckoff_baseline as wp
    import evaluate
    wp.reference_table()
    results = []
    for representation in ['explicit', 'wyckoff']:
        data, records, vocab = load_fresh(representation) if args.split == 'fresh' else load_data(args.split, representation)
        model = Denoiser(vocab).cuda()
        checkpoint = torch.load(args.checkpoint_dir / f'{representation}_{args.seed}' / 'best.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        for mode in args.modes:
            seed_all(1023)
            folder = OUT / args.label / f'{args.split}_{args.count}_{args.seed}' / f'{representation}_{mode}'
            folder.mkdir(parents=True, exist_ok=True)
            # Reuse unchanged v1 scorer, redirect only its CIF output root.
            evaluate.ROOT = folder
            (folder / 'runs' / f'{representation}_{args.seed}').mkdir(parents=True, exist_ok=True)
            generated, metrics = [], []
            correct = total = 0
            for start in range(0, min(args.count, len(records)), 32):
                end = min(start + 32, args.count)
                ids, values = variant(model, data[2][start:end], mode)
                correct += int(((ids == data[0][start:end]) & (data[0][start:end] != 0)).sum())
                total += int((data[0][start:end] != 0).sum())
                for reference, y, v in zip(records[start:end], ids.tolist(), values.tolist()):
                    try: program, error = render(vocab, y, v), None
                    except Exception as exception: program, error = '', str(exception)
                    record = {'id': reference['id'], 'representation': representation, 'seed': args.seed,
                              'program': program, 'render_error': error}
                    generated.append(record)
                    metrics.append(evaluate.score_one((record, reference)))
            (folder / 'generated.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generated))
            (folder / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics))
            summary = {'representation': representation, 'mode': mode, 'n': len(metrics), 'seed': args.seed,
                       'nonpad_token_accuracy': correct / total,
                       'counts': {k: sum(r[k] for r in metrics) for k in ['decoded', 'composition_match', 'usable', 'reference_match']},
                       'errors': Counter(r.get('error', '') for r in metrics if 'error' in r).most_common(5)}
            (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
            results.append(summary)
            print(json.dumps(summary), flush=True)
    (OUT / args.label / f'{args.split}_{args.count}_{args.seed}/summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__': main()
