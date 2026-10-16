"""Matched factorization pilot on the existing development validation cohort."""
import argparse
import hashlib
import json
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
from multiplicity_model import MultiplicityModel, normalize_scores
from site_inventory_model import SiteInventoryModel


@torch.no_grad()
def evaluate(model, data, records, masks, batch_size):
    model.eval()
    rows = []
    for offset in range(0, len(records), batch_size):
        examples = records[offset:offset + batch_size]
        batch = collate(examples, torch.device('cuda'))
        scores, allowed = model(batch, batch['Z'] > 0)
        losses = F.cross_entropy(scores.flatten(0, 1), batch['target'].flatten(), reduction='none').reshape_as(batch['Z'])
        values = scores.cpu().numpy()
        for i, record in enumerate(examples):
            choices = []
            for e, context in enumerate(record['context']):
                ids = data['context_candidates'][context].tolist()
                choices.append([(pid, float(values[i, e, j]), int(masks[pid])) for j, pid in enumerate(ids) if pid >= 0])
            solution = exact_fixed_site_map(choices)
            assert solution is not None
            predicted = list(solution[1])
            rows.append({'id': record['id'], 'sg': record['sg'], 'exact': predicted == record['profile_id'],
                         'predicted_profile_ids': predicted, 'true_profile_ids': record['profile_id'],
                         'elements': len(predicted), 'profile_nll': float(losses[i, :len(predicted)].sum())})
    return {'structures': len(rows), 'joint_exact_match': sum(r['exact'] for r in rows) / len(rows),
            'mean_profile_nll': sum(r['profile_nll'] for r in rows) / sum(r['elements'] for r in rows)}, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['aggregate', 'records'], required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--val-limit', type=int, default=2048)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    path = Path('multiplicity_candidates/data.pt')
    data = torch.load(path, map_location='cpu')
    metadata_path = Path('site_point_groups/metadata.json')
    metadata = json.loads(metadata_path.read_text())
    types = torch.zeros(len(data['wp_metadata']), dtype=torch.long)
    for position in metadata['positions']:
        types[position['wp_id']] = position['point_group_type']
    assert (types[1:] > 0).all()
    data['wp_site_type'] = types
    model = SiteInventoryModel(data, args.variant).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    train = data['splits']['train']
    val = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:args.val_limit]
    masks = make_masks(data)
    validate_decoder()
    with torch.no_grad():
        fine = torch.tensor([[[.2, -.5, 1., .4, -.1]]], device='cuda')
        coarse = torch.tensor([[[.3, -.7]]], device='cuda')
        local = torch.tensor([[[0, 0, 1, 1, 1]]], device='cuda')
        valid = torch.ones_like(local, dtype=torch.bool)
        hierarchical = normalize_scores(coarse, fine, local, valid, 'hierarchical').exp()
        parent = torch.stack((hierarchical[..., :2].sum(-1), hierarchical[..., 2:].sum(-1)), -1)
        assert torch.allclose(parent, coarse.softmax(-1), atol=1e-6)
        single = [normalize_scores(coarse[..., :1], fine, torch.zeros_like(local), valid, mode)
                  for mode in ['flat', 'degeneracy', 'hierarchical']]
        assert all(torch.allclose(single[0], x, atol=1e-6) for x in single)
        check = collate(train[:16], torch.device('cuda'))
        model.eval()
        hidden = check['Z'] > 0
        a, allowed = model(check, hidden)
        changed = {k: v.clone() for k, v in check.items()}
        changed['profile_id'] = model.context_candidates[check['context']][..., 0]
        b, _ = model(changed, hidden)
        assert torch.allclose(a, b, atol=1e-6)
        assert torch.allclose(a.exp().sum(-1), torch.ones_like(a[..., 0]), atol=1e-6)
    output = Path('site_inventory_model_runs') / (f'{args.variant}_seed{args.seed}' + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=True)
    for source in ['train_site_inventory_model.py', 'site_inventory_model.py', 'multiplicity_unaggregated.py', 'train_multiplicity_model.py', 'multiplicity_model.py']:
        shutil.copyfile(source, output / source)
    config = {**vars(args), 'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'gpu': torch.cuda.get_device_name(), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'train_structures': len(train), 'validation_structures': len(val),
              'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(Path('multiplicity_model.py').read_bytes()).hexdigest(),
              'feature_model_sha256': hashlib.sha256(Path('site_inventory_model.py').read_bytes()).hexdigest(),
              'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'frozen_trainer_sha256': hashlib.sha256(Path('train_multiplicity_model.py').read_bytes()).hexdigest(),
              'hidden_target_leakage_check': 'PASS', 'normalization_and_single_parent_identity': 'PASS',
              'same_parameter_constructor_between_site_variants': True,
              'feature_change': 'Both variants have identical site-type fine features; coarse branch aggregates by site type before count embedding only for aggregate',
              'validation_only': True, 'not_complete_diffusion_or_geometry_evaluation': True}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    del fine, coarse, local, valid, hierarchical, parent, single, check, hidden, a, b, changed
    torch.cuda.reset_peak_memory_stats()
    history, best = [], float('inf')
    for step in range(1, args.steps + 1):
        model.train()
        batch = collate([train[i] for i in rng.integers(len(train), size=args.batch_size)], torch.device('cuda'))
        real = batch['Z'] > 0
        rate = torch.rand(len(real), 1, device='cuda')
        hidden = (torch.rand(real.shape, device='cuda') < rate) & real
        hidden |= (torch.rand(len(real), 1, device='cuda') < .25) & real
        hidden[:, 0] |= ~hidden.any(-1)
        scores, valid = model(batch, hidden)
        selected = hidden & real & (valid.sum(-1) > 1)
        assert selected.any()
        loss = F.cross_entropy(scores[selected], batch['target'][selected])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        if step == 1 or step % 500 == 0:
            print(json.dumps({'variant': args.variant, 'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - start}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics, rows = evaluate(model, data, val, masks, args.batch_size)
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
