"""Matched packed-word experiment; preserves all frozen earlier sources/results."""
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
from direct_words import DirectWordModel, collate, evaluate, make_masks, validate_decoder
from packed_words import PackedWordModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['rle', 'rlemerge'], required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--val-limit', type=int, default=2048)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    path = Path('count_merge_candidates/data.pt')
    data = torch.load(path, map_location='cpu')
    train = data['splits']['train']
    val = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:args.val_limit]
    masks = make_masks(data)
    validate_decoder()
    model = PackedWordModel(data, args.variant).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    output = Path('packed_word_runs') / (f'{args.variant}_seed{args.seed}' + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=True)
    for source in ['train_packed_words.py', 'packed_words.py', 'direct_words.py']:
        shutil.copyfile(source, output / source)
    check = collate(train[:16], torch.device('cuda'))
    words = model.profile_words[check['profile_id']]
    hidden = torch.ones_like(words, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(len(model.word_counts), device='cuda')
        sparse_words = model.encode_words(all_ids)
        dense_words = DirectWordModel.encode_words(model, all_ids)
        encoding_error = float((sparse_words - dense_words).abs().max())
        assert torch.allclose(sparse_words, dense_words, atol=1e-5, rtol=1e-5), encoding_error
        packed = model(check, words, hidden)
        dense = model(check, words, hidden, packed=False)
        max_error = float((packed - dense).abs().max())
        assert torch.allclose(packed, dense, atol=1e-5, rtol=1e-5), max_error
        altered = {k: v.clone() for k, v in check.items()}
        altered['profile_id'] = model.context_candidates[check['context']][..., 0]
        assert torch.equal(model.valid_slots(check), model.valid_slots(altered))
        assert torch.allclose(packed, model(check, torch.zeros_like(words), hidden), atol=1e-6)
        assert ((words == 0) | model.valid_slots(check) | (check['Z'] == 0).unsqueeze(-1)).all()
        scores = model.profile_scores(check, packed)
        ids = model.context_candidates[check['context']]
        lp = F.log_softmax(packed, -1)
        for i in range(3):
            pid = int(ids[0, 0, i])
            if pid >= 0:
                expected = lp[0, 0, torch.arange(model.length), model.local_words[pid]].sum()
                assert torch.allclose(scores[0, 0, i], expected)
    valid = model.valid_slots(check)
    targets = model.local_words[check['profile_id']]
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(check, words, hidden)[valid], targets[valid]).backward()
    packed_gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(check, words, hidden, packed=False)[valid], targets[valid]).backward()
    gradient_error = 0.
    for before, parameter in zip(packed_gradients, model.parameters()):
        after = parameter.grad
        if before is None or after is None:
            assert before is None and after is None
        else:
            gradient_error = max(gradient_error, float((before - after).abs().max()))
            assert torch.allclose(before, after, atol=1e-4, rtol=1e-4), gradient_error
    model.zero_grad(set_to_none=True)
    config = {**vars(args), 'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'gpu': torch.cuda.get_device_name(), 'parameter_count': sum(x.numel() for x in model.parameters()),
              'position_parameter_slots': model.slot.num_embeddings, 'slots_per_element_upper_bound': model.length,
              'train_structures': len(train), 'validation_structures': len(val),
              'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(Path('packed_words.py').read_bytes()).hexdigest(),
              'frozen_base_sha256': hashlib.sha256(Path('direct_words.py').read_bytes()).hexdigest(),
              'packed_dense_eval_max_error': max_error, 'target_leakage_and_length_independence': 'PASS',
              'packed_dense_gradient_max_error': gradient_error,
              'sparse_dense_word_embedding_max_error': encoding_error,
              'sparse_dense_word_embedding_checked_definitions': len(model.word_counts),
              'candidate_score_gather_check': 'PASS', 'all_padding_decisions_from_conditions': True,
              'validation_only': True, 'not_full_diffusion_or_geometry_evaluation': True}
    assert config['parameter_count'] == 499776
    assert config['frozen_base_sha256'] == '8ec754e67884714c160def31714e42a226254e2d6a3555747a0bbd96751c8590'
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    del packed, dense, altered, scores, ids, lp, check, words, hidden, all_ids, sparse_words, dense_words
    del valid, targets, packed_gradients, before, after
    history, best = [], float('inf')
    packed_tokens = dense_tokens = 0
    # Exclude smoke-only dense-reference allocations from the measured training peak.
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        model.train()
        batch = collate([train[i] for i in rng.integers(len(train), size=args.batch_size)], torch.device('cuda'))
        words = model.profile_words[batch['profile_id']]
        real = model.valid_slots(batch)
        rate = torch.rand(len(words), 1, 1, device='cuda')
        hidden = (torch.rand(words.shape, device='cuda') < rate) & real
        hidden |= (torch.rand(len(words), 1, 1, device='cuda') < .25) & real
        hidden[:, 0, 0] |= ~hidden.flatten(1).any(-1)
        logits = model(batch, words, hidden)
        packed_tokens += len(words) * model.last_packed_length
        dense_tokens += len(words) * model.last_dense_length
        target = model.local_words[batch['profile_id']]
        loss = F.cross_entropy(logits[hidden], target[hidden])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.))
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
              'total_seconds': time.perf_counter() - start, 'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
              'training_attention_token_slots': packed_tokens, 'unpacked_global_slot_reference': dense_tokens,
              'mean_packed_slots_per_structure_in_batch': packed_tokens / (args.steps * args.batch_size)}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
