"""Replay fixed failed contexts and vary one switch-related input at a time."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('frozen_feedback_loop', HERE.parent / 'run.py')
LOOP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LOOP)


def select_cases():
    groups = [json.loads(p.read_text()) for p in (HERE.parent / 'run_001').glob('shard-*/groups/*/*.json')]
    selected = []
    for sg in LOOP.RUNNER.SG_NUMBERS:
        eligible = sorted((g for g in groups if g['plan_item']['space_group_number'] == sg
                           and not g['arms']['feedback']['attempts'][0]['success']), key=lambda g: g['group_index'])
        for group in eligible[:2]:
            selected.append(dict(case_id=f"{group['request_id']}-{group['group_index']:03d}",
                item=group['plan_item'], previous=group['arms']['feedback']['attempts'][0],
                replay=group['arms']['feedback']['attempts'][1]))
    assert len(selected) == 14
    return selected


def context(case, variant, model, tok):
    original = tok.encode(case['item']['prompt'], add_special_tokens=False)
    previous, replay = case['previous'], case['replay']
    if variant in ('baseline', 'long'):
        return replay['context_token_ids']
    if variant == 'eos_boundary':
        history, _ = LOOP._feedback_context(original, previous, [], model.start_id, model.end_id)
        suffix = tok.encode('\nUser: ' + replay['feedback'] + '\nAssistant:', add_special_tokens=False)
        return history + [tok.eos_token_id] + suffix
    if variant == 'repeat_request':
        suffix = '\n' + case['item']['prompt']
    elif variant == 'concise':
        request = case['item']['prompt'].removeprefix('User: ').removesuffix('\nAssistant:')
        errors = '; '.join(previous['failure_categories'])
        suffix = ('\nUser: ' + request + ' The previous candidate failed: ' + errors +
                  '. Revise the design. Keep the explanation brief and separate from the crystal structure.\nAssistant:')
    else:
        raise ValueError(variant)
    return LOOP._feedback_context(original, previous, tok.encode(suffix, add_special_tokens=False),
                                  model.start_id, model.end_id)[0]


def sample(ids, limit, model, tok, seed):
    import torch
    generator = torch.Generator(device=model.time.weight.device).manual_seed(seed)
    sampler = LOOP.RUNNER.CachedCausalSampler(model, tok)
    sampler.prefill(ids)
    generated, stop = [], 'limit'
    banned = {model.mask_id, model.pad_id, model.end_id}
    started = time.monotonic()
    for _ in range(limit):
        token = sampler.sample(generator, banned)
        generated.append(token)
        if token == model.start_id:
            stop = 'start'
            break
        if token == tok.eos_token_id:
            stop = 'eos'
            break
        sampler.append(token)
    text = tok.decode(generated, skip_special_tokens=False)
    return dict(stop=stop, tokens=generated, text=text, token_count=len(generated),
                contains_wyckoff_headers=all(x in text for x in ('Spacegroup:', 'abc:', 'angles:', 'Sites (')),
                seconds=time.monotonic() - started)


def main():
    import torch
    p = argparse.ArgumentParser();p.add_argument('--arm', choices=('ar', 'flare'), required=True)
    p.add_argument('--variants', nargs='+', required=True);args = p.parse_args()
    cases = select_cases()
    trainer = LOOP._load_module('switch_diagnosis_trainer', LOOP.TRAINER_PATH)
    train, model, tok, identity, source_hash, software = trainer.load_model()
    model.eval()
    checkpoint = LOOP.E.DEFAULT_CHECKPOINTS[args.arm]
    trainer.load_checkpoint(model, checkpoint, train, args.arm)
    state = train.trainable_sha256(model)
    out = HERE / args.arm
    manifest = dict(arm=args.arm, checkpoint_sha256=LOOP.E._sha256_file(checkpoint),state_sha256=state,
                    cases=[c['case_id'] for c in cases], variants=args.variants,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),software=software)
    LOOP.RUNNER._write_json(out / 'manifest.json', manifest)
    for variant in args.variants:
        for case in cases:
            ids = context(case, variant, model, tok)
            result = sample(ids, 768 if variant == 'long' else 256, model, tok, case['replay']['seed'])
            result.update(case_id=case['case_id'],variant=variant,arm=args.arm,context_token_ids=ids,
                          seed=case['replay']['seed'],state_sha256=state)
            if args.arm == 'flare' and variant == 'baseline':
                assert result['tokens'] == case['replay']['lead_token_ids'], 'original failure replay drifted'
            LOOP.RUNNER._write_json(out / variant / (case['case_id'] + '.json'),result)
        rows=[json.loads(q.read_text()) for q in (out / variant).glob('*.json')]
        print(json.dumps(dict(arm=args.arm,variant=variant,n=len(rows),
                             start=sum(r['stop']=='start' for r in rows),
                             eos=sum(r['stop']=='eos' for r in rows),limit=sum(r['stop']=='limit' for r in rows))),flush=True)
    assert train.trainable_sha256(model) == state


if __name__ == '__main__':main()
