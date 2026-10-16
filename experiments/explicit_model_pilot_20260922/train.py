"""Preflight and matched two-seed training. Test set is never loaded here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
from model import ROOT, Denoiser, corrupt, load_data, objective, seed_all, validate, render


def preflight():
    seed_all(901)
    data, records, vocab = load_data('train', 'explicit')
    from prepare import sp, wp, compare
    lookup = {t: i for i, t in enumerate(vocab)}
    codec_checks = []
    for record in records[:32]:
        for representation, codec in [('explicit', sp), ('wyckoff', wp)]:
            packed = record['packed'][representation]
            restored = render(vocab, [lookup[t] for t in packed['tokens']], packed['values'])
            error = compare(codec.decode(record['programs'][representation]), codec.decode(restored))
            codec_checks.append(error)
    model = Denoiser(vocab).cuda()
    y, v, c = [x[:8] for x in data]
    model.eval()
    rate = torch.ones(8, device='cuda')
    x, u, hidden = corrupt(y, v, rate)
    x2, u2, _ = corrupt(y.flip(1), v.flip(1) + 13, rate)
    assert torch.equal(x, x2) and torch.equal(u, u2)
    with torch.no_grad():
        a, b = model(x, u, c, rate)
        aa, bb = model(x2, u2, c, rate)
    assert torch.equal(a, aa) and torch.equal(b, bb), 'Full mask leaks target information'
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curve = []
    for step in range(301):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, numbers = model(x, u, c, rate)
        loss, ce, mse = objective(model, logits, numbers, y, v, hidden)
        assert torch.isfinite(loss)
        if step % 50 == 0:
            curve.append({'step': step, 'loss': float(loss), 'ce': float(ce), 'mse': float(mse)})
            print(json.dumps({'preflight': curve[-1]}), flush=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
    assert curve[-1]['loss'] < 0.35 * curve[0]['loss'], 'Tiny batch did not learn'
    model.eval()
    with torch.no_grad(): before = model(x, u, c, rate)
    checkpoint = ROOT / 'runs/preflight.pt'
    torch.save(model.state_dict(), checkpoint)
    other = Denoiser(vocab).cuda(); other.load_state_dict(torch.load(checkpoint)); other.eval()
    with torch.no_grad(): after = other(x, u, c, rate)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    result = {'full_mask_target_leakage': False, 'uses_true_target_length': False,
              'checkpoint_exact_reload': True, 'tiny_batch_curve': curve,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'condition': 'reduced composition only', 'sample_ids': [r['id'] for r in records[:8]],
              'numeric_packing_roundtrips': len(codec_checks), 'max_packing_error_A': max(codec_checks)}
    (ROOT / 'runs/preflight.json').write_text(json.dumps(result, indent=2))


def train(representation, seed, steps):
    seed_all(seed)
    train_data, records, vocab = load_data('train', representation)
    val_data, _, _ = load_data('val', representation)
    model = Denoiser(vocab).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    folder = ROOT / 'runs' / f'{representation}_{seed}'; folder.mkdir(parents=True, exist_ok=True)
    config = {'representation': representation, 'seed': seed, 'steps': steps, 'batch_size': 32,
              'condition': '118 log1p reduced composition counts; no SG, N, length, geometry',
              'architecture': {'layers': 4, 'width': 192, 'heads': 6, 'ffn': 768, 'max_length': 256},
              'optimizer': 'AdamW', 'lr': 3e-4, 'weight_decay': 0.01, 'pad_loss_weight': 0.1,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'train_count': len(records), 'validation_count': len(val_data[0]),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'host': socket.gethostname(), 'slurm_job': os.environ['SLURM_JOB_ID'],
              'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
              'data_sha256': hashlib.sha256((ROOT / 'data/records.jsonl').read_bytes()).hexdigest()}
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    began = time.time(); best = float('inf'); best_step = None
    initial = validate(model, val_data)
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'curve.jsonl').open('w') as log:
        log.write(json.dumps({'step': 0, 'validation': initial, 'seconds': 0}) + '\n'); log.flush()
        running = 0
        for step in range(1, steps + 1):
            model.train()
            index = torch.randint(len(records), (32,), device='cuda')
            y, v, c = [tensor[index] for tensor in train_data]
            rate = 0.15 + 0.85 * torch.rand(32, device='cuda')
            rate[torch.rand(32, device='cuda') < 0.25] = 1.0
            x, u, hidden = corrupt(y, v, rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, predicted = model(x, u, c, rate)
                loss, ce, mse = objective(model, logits, predicted, y, v, hidden)
            assert torch.isfinite(loss), f'Nonfinite training loss at {step}'
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            running += float(loss)
            if step % 100 == 0:
                entry = {'step': step, 'training_loss': running / 100, 'seconds': time.time() - began}
                running = 0
                if step % 500 == 0:
                    validation = validate(model, val_data); entry['validation'] = validation
                    if validation['loss'] < best:
                        best, best_step = validation['loss'], step
                        torch.save({'model': model.state_dict(), 'vocab': vocab, 'config': config,
                                    'step': step, 'validation': validation}, folder / 'best.pt')
                log.write(json.dumps(entry) + '\n'); log.flush()
                print(json.dumps({'run': folder.name, **entry}), flush=True)
    result = {'best_step': best_step, 'best_validation_loss': best, 'initial_validation': initial,
              'elapsed_seconds': time.time() - began, 'max_gpu_memory_MB': torch.cuda.max_memory_allocated() / 2**20}
    (folder / 'training_result.json').write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['preflight', 'train'])
    parser.add_argument('--representation', choices=['explicit', 'wyckoff']); parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int, default=4000); args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4)
    (ROOT / 'runs').mkdir(exist_ok=True)
    if args.mode == 'preflight': preflight()
    else: train(args.representation, args.seed, args.steps)


if __name__ == '__main__': main()
