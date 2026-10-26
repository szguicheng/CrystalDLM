"""Direct masked-word diagnostic with identical feasible-profile readout.

Predicts individual canonical word slots, including PAD. This is a denoising
experiment, not a claim of a complete crystal diffusion sampler.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_profile_model import collate
from evaluate_joint import make_masks, validate_decoder
from joint_decode import exact_fixed_site_map


def prepare_tables(data, variant):
    # A common length for both variants, derived from all theoretical candidates.
    length = data['profile_words']['unigram'].shape[1]
    source = data['profile_words'][variant]
    words = F.pad(source, (0, length - source.shape[1]))
    local_words = torch.zeros_like(words)
    contexts = []
    assigned = set()
    for candidates in data['context_candidates']:
        ids = candidates[candidates >= 0]
        assert not assigned.intersection(ids.tolist())
        assigned.update(ids.tolist())
        vocabulary = torch.unique(words[ids])
        assert vocabulary[0] == 0
        contexts.append(vocabulary)
        lookup = torch.full((len(data['word_counts']),), -1, dtype=torch.long)
        lookup[vocabulary] = torch.arange(len(vocabulary))
        local_words[ids] = lookup[words[ids]]
        assert torch.equal(vocabulary[local_words[ids]], words[ids])
    assert len(assigned) == len(words)
    max_vocab = max(map(len, contexts))
    context_words = torch.stack([F.pad(v, (0, max_vocab - len(v)), value=-1) for v in contexts])
    return words, local_words, context_words


class DirectWordModel(nn.Module):
    def __init__(self, data, variant, width=96, layers=3):
        super().__init__()
        self.width = width
        words, local, context_words = prepare_tables(data, variant)
        self.length = words.shape[1]
        for name, value in [('profile_words', words), ('local_words', local),
                            ('context_words', context_words),
                            ('context_candidates', data['context_candidates']),
                            ('word_counts', data['word_counts']), ('word_wp_ids', data['word_wp_ids']),
                            ('wp_metadata', data['wp_metadata'])]:
            self.register_buffer(name, value)
        self.element = nn.Embedding(119, width, padding_idx=0)
        self.amount = nn.Embedding(data['max_atoms'] + 1, width, padding_idx=0)
        self.sg = nn.Embedding(231, width, padding_idx=0)
        self.slot = nn.Embedding(self.length, width)
        self.wp = nn.Embedding(len(data['wp_metadata']), width, padding_idx=0)
        self.multiplicity = nn.Embedding(193, width, padding_idx=0)
        self.dof = nn.Embedding(4, width)
        self.occupancy = nn.Embedding(int(data['profile_counts'].max()) + 1, width, padding_idx=0)
        self.word_projection = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.pad = nn.Parameter(torch.zeros(width))
        self.mask = nn.Parameter(torch.zeros(width))
        self.state_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                    activation='gelu', batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.head = nn.Linear(width, width)

    def encode_words(self, ids):
        counts = self.word_counts[ids]
        wpids = self.word_wp_ids[ids]
        meta = self.wp_metadata[wpids]
        features = self.wp(wpids) + self.multiplicity(meta[..., 0]) + self.dof(meta[..., 1])
        vectors = self.word_projection((features * self.occupancy(counts) * (counts > 0).unsqueeze(-1)).sum(-2))
        return torch.where((ids == 0).unsqueeze(-1), self.pad, vectors)

    def forward(self, batch, words, hidden):
        allowed_words = self.context_words[batch['context']]
        ids, inverse = torch.unique(torch.cat((allowed_words.clamp_min(0).reshape(-1), words.reshape(-1))), return_inverse=True)
        vectors = self.encode_words(ids)
        split = allowed_words.numel()
        output_vectors = vectors[inverse[:split]].reshape(*allowed_words.shape, self.width)
        observed = vectors[inverse[split:]].reshape(*words.shape, self.width)
        state = torch.where(hidden.unsqueeze(-1), self.mask, observed)
        condition = self.element(batch['Z']) + self.amount(batch['N']) + self.sg(batch['sg']).unsqueeze(1)
        positions = self.slot(torch.arange(self.length, device=words.device))
        x = condition.unsqueeze(2) + positions + self.state_norm(state)
        real = (batch['Z'] > 0).unsqueeze(-1).expand_as(words)
        h = self.head(self.trunk(x.flatten(1, 2), src_key_padding_mask=~real.flatten(1, 2)))
        h = h.reshape(*words.shape, self.width)
        logits = torch.einsum('beld,bekd->belk', h, output_vectors) / math.sqrt(self.width)
        return logits.masked_fill((allowed_words < 0).unsqueeze(2), -1e4)

    def profile_scores(self, batch, word_logits):
        ids = self.context_candidates[batch['context']]
        local = self.local_words[ids.clamp_min(0)]
        logp = F.log_softmax(word_logits, dim=-1)
        expanded = logp.unsqueeze(2).expand(*local.shape, logp.shape[-1])
        scores = expanded.gather(-1, local.unsqueeze(-1)).squeeze(-1).sum(-1)
        return scores.masked_fill(ids < 0, -1e4)


@torch.no_grad()
def evaluate(model, data, records, batch_size, masks):
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        examples = records[start:start + batch_size]
        batch = collate(examples, torch.device('cuda'))
        words = model.profile_words[batch['profile_id']]
        logits = model(batch, words, torch.ones_like(words, dtype=torch.bool))
        scores = model.profile_scores(batch, logits)
        nll = F.cross_entropy(scores.flatten(0, 1), batch['target'].flatten(), reduction='none').reshape_as(batch['Z'])
        score_values = F.log_softmax(scores, -1).cpu().numpy()
        raw_correct = (logits.argmax(-1) == model.local_words[batch['profile_id']]).all(-1)
        for i, record in enumerate(examples):
            candidates = []
            for e, context in enumerate(record['context']):
                ids = data['context_candidates'][context].tolist()
                candidates.append([(pid, float(score_values[i, e, j]), int(masks[pid])) for j, pid in enumerate(ids) if pid >= 0])
            solution = exact_fixed_site_map(candidates)
            assert solution is not None
            predicted = list(solution[1])
            rows.append({'id': record['id'], 'sg': record['sg'], 'exact': predicted == record['profile_id'],
                         'predicted_profile_ids': predicted, 'true_profile_ids': record['profile_id'],
                         'elements': len(predicted), 'correct_elements': sum(a == b for a, b in zip(predicted, record['profile_id'])),
                         'raw_word_sequence_exact': bool(raw_correct[i, :len(predicted)].all()),
                         'profile_nll': float(nll[i, :len(predicted)].sum())})
    metrics = {'structures': len(rows), 'joint_exact_match': np.mean([r['exact'] for r in rows]).item(),
               'element_accuracy': sum(r['correct_elements'] for r in rows) / sum(r['elements'] for r in rows),
               'mean_profile_nll': sum(r['profile_nll'] for r in rows) / sum(r['elements'] for r in rows),
               'raw_word_sequence_exact': np.mean([r['raw_word_sequence_exact'] for r in rows]).item()}
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['unigram', 'rle', 'setmerge'], required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-limit', type=int, default=2048)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    data_path = Path('word_candidates/data.pt')
    data = torch.load(data_path, map_location='cpu')
    masks = make_masks(data)
    validate_decoder()
    train = data['splits']['train']
    val = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:args.val_limit]
    model = DirectWordModel(data, args.variant).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    output = Path('direct_word_runs') / (f'{args.variant}_seed{args.seed}' + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output / 'source.py')
    check = collate(train[:8], torch.device('cuda'))
    words = model.profile_words[check['profile_id']]
    model.eval()
    with torch.no_grad():
        hidden = torch.ones_like(words, dtype=torch.bool)
        a = model(check, words, hidden)
        b = model(check, torch.zeros_like(words), hidden)
        assert torch.allclose(a, b, atol=1e-6), 'Target-word leakage'
        scores = model.profile_scores(check, a)
        candidates = model.context_candidates[check['context']]
        lp = F.log_softmax(a, -1)
        for i in range(3):
            pid = int(candidates[0, 0, i])
            if pid >= 0:
                expected = lp[0, 0, torch.arange(model.length), model.local_words[pid]].sum()
                assert torch.allclose(scores[0, 0, i], expected)
    config = {**vars(args), 'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'gpu': torch.cuda.get_device_name(), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'common_slots_per_element': model.length, 'train_structures': len(train), 'validation_structures': len(val),
              'hidden_target_leakage_check': 'PASS', 'candidate_score_gather_check': 'PASS',
              'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'task': 'Direct canonical word masking conditioned on SG and composition; full-mask validation',
              'no_target_length_given': True, 'not_full_diffusion_or_geometry_evaluation': True}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    history, best = [], float('inf')
    for step in range(1, args.steps + 1):
        model.train()
        batch = collate([train[i] for i in rng.integers(len(train), size=args.batch_size)], torch.device('cuda'))
        words = model.profile_words[batch['profile_id']]
        real = (batch['Z'] > 0).unsqueeze(-1).expand_as(words)
        rate = torch.rand(len(words), 1, 1, device='cuda')
        hidden = (torch.rand(words.shape, device='cuda') < rate) & real
        hidden |= (torch.rand(len(words), 1, 1, device='cuda') < .25) & real
        hidden[:, 0, 0] |= ~hidden.flatten(1).any(-1)
        logits = model(batch, words, hidden)
        target = model.local_words[batch['profile_id']]
        loss = F.cross_entropy(logits[hidden], target[hidden])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(grad)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({'variant': args.variant, 'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - start}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics, rows = evaluate(model, data, val, args.batch_size, masks)
            item = {'step': step, **metrics, 'seconds': time.perf_counter() - start}
            history.append(item)
            (output / 'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps({'variant': args.variant, 'validation': item}), flush=True)
            if metrics['mean_profile_nll'] < best:
                best = metrics['mean_profile_nll']
                torch.save({'model': model.state_dict(), 'config': config, 'step': step}, output / 'best.pt')
                (output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                best_metrics, best_step = metrics, step
    result = {'status': 'COMPLETED', 'best_step': best_step, 'metrics': best_metrics, 'config': config,
              'total_seconds': time.perf_counter() - start, 'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated()}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
