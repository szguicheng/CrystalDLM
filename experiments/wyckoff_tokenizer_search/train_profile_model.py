"""Matched-capacity non-AR masked profile modeling diagnostic, no geometry claims."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ProfileModel(nn.Module):
    def __init__(self, data, variant, width=128, layers=3):
        super().__init__()
        self.variant = variant
        self.width = width
        self.register_buffer('profile_counts', data['profile_counts'])
        self.register_buffer('profile_wp_ids', data['profile_wp_ids'])
        self.register_buffer('wp_metadata', data['wp_metadata'])
        self.register_buffer('context_candidates', data['context_candidates'])
        self.element = nn.Embedding(119, width, padding_idx=0)
        self.space_group = nn.Embedding(231, width, padding_idx=0)
        self.amount = nn.Embedding(data['max_atoms'] + 1, width, padding_idx=0)
        self.wp = nn.Embedding(len(data['wp_metadata']), width, padding_idx=0)
        self.multiplicity = nn.Embedding(193, width, padding_idx=0)
        self.dof = nn.Embedding(4, width)
        self.occupancy = nn.Embedding(int(data['profile_counts'].max()) + 1, width, padding_idx=0)
        self.mask = nn.Parameter(torch.zeros(width))
        self.state_projection = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.query = nn.Linear(width, width)
        # Identical parameter count: the same MLP acts on query (factorized)
        # or on the complete candidate (compositional), with no dropout here.
        self.head = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def encode_profiles(self, profile_ids):
        counts = self.profile_counts[profile_ids]
        wp_ids = self.profile_wp_ids[profile_ids]
        metadata = self.wp_metadata[wp_ids]
        features = self.wp(wp_ids) + self.multiplicity(metadata[..., 0]) + self.dof(metadata[..., 1])
        contributions = features * self.occupancy(counts)
        return (contributions * (counts > 0).unsqueeze(-1)).sum(dim=-2)

    def forward(self, batch, hidden):
        real = batch['Z'] > 0
        candidates = self.context_candidates[batch['context']]
        valid_candidates = candidates >= 0
        # Candidate pool is wholly determined by input G,N, never by target choice.
        all_ids, inverse = torch.unique(torch.cat([candidates.clamp_min(0).reshape(-1),
                                                  batch['profile_id'].reshape(-1)]), return_inverse=True)
        encoded = self.encode_profiles(all_ids)
        split = candidates.numel()
        candidate_encoded = self.head(encoded) if self.variant == 'compositional' else encoded
        candidate_features = candidate_encoded[inverse[:split]].reshape(*candidates.shape, self.width)
        observed_features = encoded[inverse[split:]].reshape(*batch['profile_id'].shape, self.width)
        state = torch.where(hidden.unsqueeze(-1), self.mask,
                            self.state_projection(observed_features))
        x = self.element(batch['Z']) + self.amount(batch['N']) + self.space_group(batch['sg']).unsqueeze(1) + state
        h = self.query(self.trunk(x, src_key_padding_mask=~real))
        if self.variant == 'factorized':
            scores = (self.head(h).unsqueeze(-2) * candidate_features).sum(-1) / math.sqrt(self.width)
        elif self.variant == 'compositional':
            scores = (h.unsqueeze(-2) * candidate_features).sum(-1) / math.sqrt(self.width)
        else:
            raise ValueError(self.variant)
        return scores.masked_fill(~valid_candidates, -1e4), valid_candidates


def collate(records, device):
    size = max(len(x['Z']) for x in records)
    keys = ['Z', 'N', 'context', 'profile_id', 'target']
    batch = {key: torch.tensor([x[key] + [0] * (size - len(x[key])) for x in records],
                               dtype=torch.long, device=device) for key in keys}
    batch['sg'] = torch.tensor([x['sg'] for x in records], dtype=torch.long, device=device)
    return batch


def fixed_position_validity(data, examples, predicted_profile_ids):
    valid = []
    for example, pids in zip(examples, predicted_profile_ids):
        used = set()
        ok = True
        for pid in pids[:len(example['Z'])]:
            counts = data['profile_counts'][pid]
            wp_ids = data['profile_wp_ids'][pid]
            fixed = (data['wp_metadata'][wp_ids, 1] == 0) & (counts > 0)
            occupied = set(wp_ids[fixed].tolist())
            ok = ok and not bool(used & occupied)
            used.update(occupied)
        valid.append(ok)
    return valid


@torch.no_grad()
def evaluate(model, data, records, batch_size, device, detailed=False):
    model.eval()
    totals = {'elements': 0, 'correct_elements': 0, 'ambiguous_elements': 0,
              'correct_ambiguous': 0, 'structures': 0, 'exact_structures': 0,
              'valid_fixed_occupancy_structures': 0, 'nll_sum': 0.,
              'unseen_profile_elements': 0, 'correct_unseen_profile_elements': 0}
    details = []
    train_seen = set(data['train_profile_counts'])
    for offset in range(0, len(records), batch_size):
        examples = records[offset:offset + batch_size]
        batch = collate(examples, device)
        real = batch['Z'] > 0
        logits, allowed = model(batch, real)  # All structure unknown; only G + composition supplied.
        prediction = logits.argmax(-1)
        correct = prediction == batch['target']
        ambiguous = real & (allowed.sum(-1) > 1)
        losses = F.cross_entropy(logits.flatten(0, 1), batch['target'].flatten(), reduction='none').reshape_as(real)
        candidates = model.context_candidates[batch['context']]
        predicted_ids = candidates.gather(-1, prediction.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        legal = fixed_position_validity(data, examples, predicted_ids)
        totals['elements'] += int(real.sum())
        totals['correct_elements'] += int((correct & real).sum())
        totals['ambiguous_elements'] += int(ambiguous.sum())
        totals['correct_ambiguous'] += int((correct & ambiguous).sum())
        totals['structures'] += len(examples)
        totals['exact_structures'] += int((correct | ~real).all(-1).sum())
        totals['valid_fixed_occupancy_structures'] += sum(legal)
        totals['nll_sum'] += float(losses[real].sum())
        for i, e in enumerate(examples):
            unseen = [pid not in train_seen for pid in e['profile_id']]
            totals['unseen_profile_elements'] += sum(unseen)
            totals['correct_unseen_profile_elements'] += sum(u and bool(correct[i, j]) for j, u in enumerate(unseen))
            if detailed:
                details.append({'id': e['id'], 'sg': e['sg'], 'true_profile_ids': e['profile_id'],
                                'predicted_profile_ids': predicted_ids[i][:len(e['Z'])],
                                'exact': bool((correct[i] | ~real[i]).all()), 'fixed_occupancy_valid': legal[i],
                                'profile_nll': float(losses[i][real[i]].sum())})
    totals.update({'element_accuracy': totals['correct_elements'] / totals['elements'],
                   'ambiguous_element_accuracy': totals['correct_ambiguous'] / totals['ambiguous_elements'],
                   'structure_exact_match': totals['exact_structures'] / totals['structures'],
                   'fixed_occupancy_validity': totals['valid_fixed_occupancy_structures'] / totals['structures'],
                   'mean_profile_nll': totals['nll_sum'] / totals['elements']})
    return totals, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['factorized', 'compositional', 'unigram', 'rle', 'setmerge'], required=True)
    parser.add_argument('--data-path', type=Path, default=Path('candidates_cap256/data.pt'))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--val-limit', type=int, default=2048)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    assert torch.cuda.is_available(), 'This model pilot requires a scheduled GPU.'
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    data_path = args.data_path
    data = torch.load(data_path, map_location='cpu')
    train = data['splits']['train']
    val = sorted(data['splits']['val'], key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())[:args.val_limit]
    tag = f'{args.variant}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    output = Path('model_runs') / tag
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output / 'train_source.py')
    if args.variant in ['unigram', 'rle', 'setmerge']:
        shutil.copyfile('word_model.py', output / 'word_model_source.py')
    device = torch.device('cuda')
    if args.variant in ['unigram', 'rle', 'setmerge']:
        from word_model import WordModel
        model = WordModel(data, args.variant, args.width, args.layers).to(device)
    else:
        model = ProfileModel(data, args.variant, args.width, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    # Check that masked inputs cannot reveal hidden target-profile IDs.
    check = collate(train[:8], device)
    hidden = check['Z'] > 0
    model.eval()
    with torch.no_grad():
        a, _ = model(check, hidden)
        altered = {k: v.clone() for k, v in check.items()}
        altered['profile_id'] = model.context_candidates[check['context']][..., 0]
        b, _ = model(altered, hidden)
        assert torch.allclose(a, b, atol=1e-6), 'Hidden-target leakage.'
    config = {**vars(args), 'data_path': str(data_path), 'host': socket.gethostname(), 'job_id': os.environ['SLURM_JOB_ID'],
              'gpu': torch.cuda.get_device_name(), 'train_structures': len(train), 'val_structures': len(val),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'hidden_target_leakage_check': 'PASS',
              'both_variants_use_identical_complete_legal_candidate_sets': True,
              'task': 'Composition + SG -> one-element Wyckoff profiles; all-mask validation',
              'not_geometry_or_material_stability_evaluation': True}
    if args.variant in ['unigram', 'rle', 'setmerge']:
        config['word_model_sha256'] = hashlib.sha256(Path('word_model.py').read_bytes()).hexdigest()
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        examples = [train[i] for i in rng.integers(len(train), size=args.batch_size)]
        batch = collate(examples, device)
        real = batch['Z'] > 0
        rates = torch.rand(len(examples), 1, device=device)
        hidden = (torch.rand(real.shape, device=device) < rates) & real
        hidden |= (torch.rand(len(examples), 1, device=device) < .25) & real
        hidden[:, 0] |= ~hidden.any(-1)
        logits, allowed = model(batch, hidden)
        selected = hidden & real & (allowed.sum(-1) > 1)
        loss = F.cross_entropy(logits[selected], batch['target'][selected])
        assert torch.isfinite(loss), 'Non-finite training loss.'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'elapsed_seconds': time.perf_counter() - start}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _ = evaluate(model, data, val, args.batch_size, device)
            item = {'step': step, **metrics, 'elapsed_seconds': time.perf_counter() - start}
            history.append(item)
            (output / 'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps({'validation': item}), flush=True)
            if metrics['mean_profile_nll'] < best:
                best = metrics['mean_profile_nll']
                torch.save({'model': model.state_dict(), 'config': config, 'step': step,
                            'validation': metrics}, output / 'best.pt')
    checkpoint = torch.load(output / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    metrics, details = evaluate(model, data, val, args.batch_size, device, detailed=True)
    with (output / 'validation_predictions.jsonl').open('w') as stream:
        for row in details:
            stream.write(json.dumps(row) + '\n')
    result = {'status': 'COMPLETED', 'best_step': checkpoint['step'], 'metrics': metrics,
              'total_seconds': time.perf_counter() - start, 'config': config,
              'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated()}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
