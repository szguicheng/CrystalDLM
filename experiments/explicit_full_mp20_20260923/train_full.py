"""Fixed true-epoch training, with a speed-only decision between 500 and 1000."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
from model import ROOT, MAXLEN, Denoiser, corrupt, load_data, objective, seed_all, validate


def update(model, optimizer, batch):
    y, v, c = batch
    rate = .15 + .85 * torch.rand(len(y), device='cuda')
    rate[torch.rand(len(y), device='cuda') < .25] = 1
    x, u, hidden = corrupt(y, v, rate)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, predicted = model(x, u, c, rate)
        loss, _, _ = objective(model, logits, predicted, y, v, hidden)
    assert torch.isfinite(loss), 'Nonfinite training objective'
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    return float(loss.detach())


def benchmark():
    rows = []
    for representation in ['explicit', 'wyckoff']:
        data, records, vocab = load_data('train', representation)
        n = len(records); del records
        if representation == 'explicit':
            seed_all(17)
            first = Denoiser(vocab).cuda().train()
            optimizer = torch.optim.AdamW(first.parameters(), lr=3e-4, weight_decay=.01)
            index = torch.randint(n, (32,), device='cuda')
            update(first, optimizer, tuple(tensor[index] for tensor in data))
            path = ROOT / 'runs/resume_probe.pt'
            save(path, {'model': first.state_dict(), 'optimizer': optimizer.state_dict(),
                        'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()})
            before = {name: p.detach().clone() for name, p in first.named_parameters()}
            index_a = torch.randint(n, (32,), device='cuda')
            loss_a = update(first, optimizer, tuple(tensor[index_a] for tensor in data))
            after_rng = torch.cuda.get_rng_state()
            second = Denoiser(vocab).cuda().train()
            optimizer_b = torch.optim.AdamW(second.parameters(), lr=3e-4, weight_decay=.01)
            state = torch.load(path, map_location='cpu')
            second.load_state_dict(state['model']); optimizer_b.load_state_dict(state['optimizer'])
            initial_difference = max(float((p-before[name]).abs().max()) for name, p in second.named_parameters())
            restored_optimizer = optimizer_b.state_dict()
            optimizer_equal = all(torch.equal(value.cpu(), restored_optimizer['state'][key][name].cpu())
                    if isinstance(value, torch.Tensor) else value == restored_optimizer['state'][key][name]
                    for key, values in state['optimizer']['state'].items() for name, value in values.items())
            torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'])
            index_b = torch.randint(n, (32,), device='cuda')
            assert torch.equal(index_a, index_b)
            loss_b = update(second, optimizer_b, tuple(tensor[index_b] for tensor in data))
            differences = [(name, float((a-b).abs().max())) for (name, a), b in zip(first.named_parameters(), second.parameters())]
            difference = max(v for _, v in differences)
            same_rng = torch.equal(after_rng, torch.cuda.get_rng_state())
            passed = initial_difference == 0 and optimizer_equal and same_rng and abs(loss_a-loss_b) <= 1e-7 * max(1, abs(loss_a))
            probe = {'same_next_batch': True, 'initial_parameter_difference': initial_difference,
                     'optimizer_tensors_exactly_restored': optimizer_equal,
                     'same_rng_after_next_step': torch.equal(after_rng, torch.cuda.get_rng_state()),
                     'loss_a': loss_a, 'loss_b': loss_b, 'largest_differences': sorted(differences, key=lambda x: x[1], reverse=True)[:6],
                     'max_parameter_difference_after_next_update': difference, 'passed': passed,
                     'note': 'Require exact restored parameters/optimizer/RNG and matching next loss; mixed-precision GPU backward is not claimed bitwise deterministic.'}
            (ROOT / 'runs/resume_probe.json').write_text(json.dumps(probe, indent=2))
            print(json.dumps({'resume_probe': probe}), flush=True)
            assert passed, probe
            del first, second, optimizer, optimizer_b, state, before
            torch.cuda.empty_cache()
        for batch_size in [32, 128, 256, 512]:
            seed_all(17)
            model = Denoiser(vocab).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for step in range(80):
                torch.cuda.synchronize(); began = time.perf_counter()
                index = torch.randint(n, (batch_size,), device='cuda')
                loss = update(model, optimizer, tuple(tensor[index] for tensor in data))
                torch.cuda.synchronize()
                if step >= 20: timings.append(time.perf_counter() - began)
            ordered = sorted(timings)
            seconds = ordered[len(ordered)//2]
            entry = {'representation': representation, 'batch_size': batch_size,
                     'seconds_per_step_median': seconds, 'examples_per_second': batch_size / seconds,
                     'seconds_per_epoch': math.ceil(n / batch_size) * seconds,
                     'gpu_peak_MB': torch.cuda.max_memory_allocated() / 2**20, 'last_loss': loss}
            rows.append(entry); print(json.dumps(entry), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
        del data
    costs = {b: sum(r['seconds_per_epoch'] for r in rows if r['batch_size'] == b) for b in [32, 128, 256, 512]}
    fastest = min(costs.values())
    batch = min(b for b, cost in costs.items() if cost <= 1.1 * fastest)
    # 15% for validation/checkpoint overhead. Fixed cap, no convergence stopping.
    estimated_1000 = costs[batch] * 1000 * 1.15
    epochs = 1000 if estimated_1000 <= 3600 else 500
    result = {'batch_size': batch, 'epochs': epochs, 'seed': 17, 'train_count': n,
              'steps_per_epoch': math.ceil(n / batch), 'steps_per_model': math.ceil(n / batch) * epochs,
              'estimated_pair_seconds': costs[batch] * epochs * 1.15,
              'estimated_pair_1000_seconds': estimated_1000,
              'rule': 'smallest batch within 10% of best measured pair throughput; 1000 epochs if pair including 15% reserve <=60 min, else 500',
              'gpu': torch.cuda.get_device_name(), 'max_length': MAXLEN, 'measurements': rows,
              'slurm_job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname()}
    assert not (ROOT / 'budget.json').exists(), 'Budget is already frozen'
    (ROOT / 'budget.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'frozen_budget': result}), flush=True)


def save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(representation):
    budget = json.loads((ROOT / 'budget.json').read_text())
    epochs, batch_size, seed = (budget[key] for key in ['epochs', 'batch_size', 'seed'])
    train_data, records, vocab = load_data('train', representation)
    n = len(records); del records
    val_data, records, _ = load_data('val', representation)
    nval = len(records); del records
    assert n == 27136 and nval == 9047
    seed_all(seed)
    model = Denoiser(vocab).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    folder = ROOT / 'runs' / f'{representation}_{seed}'; folder.mkdir(parents=True, exist_ok=True)
    config = {'representation': representation, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
              'train_count': n, 'validation_count': nval, 'steps_per_epoch': math.ceil(n/batch_size),
              'condition': 'reduced composition only', 'max_length': MAXLEN,
              'epoch_definition': 'one shuffled traversal of every training structure; retain final partial batch',
              'lr': 3e-4, 'weight_decay': .01, 'early_stopping': False,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'slurm_job': os.environ['SLURM_JOB_ID'],
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['model.py', 'prepare.py', 'train_full.py']},
              'data_sha256': hashlib.sha256((ROOT / 'data/records.jsonl').read_bytes()).hexdigest()}
    first, steps, best, elapsed = 1, 0, float('inf'), 0.
    best_epoch = None
    if (folder / 'last.pt').exists():
        state = torch.load(folder / 'last.pt', map_location='cpu')
        assert state['config']['source_sha256'] == config['source_sha256']
        assert state['config']['data_sha256'] == config['data_sha256']
        assert state['config']['epochs'] == epochs and state['config']['batch_size'] == batch_size
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_cpu']); torch.cuda.set_rng_state(state['rng_cuda'])
        first, steps, best, best_epoch, elapsed = state['epoch']+1, state['step'], state['best'], state['best_epoch'], state['elapsed_seconds']
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    began = time.time(); torch.cuda.reset_peak_memory_stats()
    with (folder / 'curve.jsonl').open('a') as log:
        for epoch in range(first, epochs + 1):
            model.train(); total = 0.
            permutation = torch.randperm(n, device='cuda')
            for start in range(0, n, batch_size):
                index = permutation[start:start+batch_size]
                total += update(model, optimizer, tuple(tensor[index] for tensor in train_data)) * len(index)
                steps += 1
            duration = elapsed + time.time() - began
            entry = {'epoch': epoch, 'step': steps, 'examples_seen': epoch * n,
                     'training_loss': total / n, 'elapsed_seconds': duration}
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                validation = validate(model, val_data)
                entry['validation'] = validation
                core = {'model': model.state_dict(), 'vocab': vocab, 'config': config,
                        'epoch': epoch, 'step': steps, 'validation': validation}
                if validation['loss'] < best:
                    best, best_epoch = validation['loss'], epoch
                    save(folder / 'best.pt', core)
                state = {**core, 'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state(), 'best': best, 'best_epoch': best_epoch,
                         'elapsed_seconds': elapsed + time.time() - began}
                save(folder / 'last.pt', state)
                if epoch % 100 == 0: save(folder / f'epoch_{epoch:04d}.pt', core)
                (folder / 'status.json').write_text(json.dumps({'state': 'running', 'target_epochs': epochs,
                                                               **entry, 'best_epoch': best_epoch}, indent=2))
                print(json.dumps({'run': folder.name, **entry, 'best_epoch': best_epoch}), flush=True)
            log.write(json.dumps(entry) + '\n'); log.flush()
    assert steps == math.ceil(n/batch_size) * epochs
    result = {'state': 'complete', 'epochs': epochs, 'steps': steps, 'examples_seen': epochs*n,
              'best_epoch': best_epoch, 'best_validation_loss': best,
              'elapsed_seconds': elapsed + time.time() - began,
              'max_gpu_memory_MB': torch.cuda.max_memory_allocated()/2**20}
    (folder / 'training_result.json').write_text(json.dumps(result, indent=2))
    (folder / 'status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'complete': folder.name, **result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['benchmark', 'train'])
    parser.add_argument('--representation', choices=['explicit', 'wyckoff']); args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4)
    if args.mode == 'benchmark': benchmark()
    else: train(args.representation)
