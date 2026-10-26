"""Small fixed-reference-prefix test of crystal generation and AR retention."""
import argparse
import json
from pathlib import Path
import time

import run

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['source', 'ar', 'flare'], required=True)
    args = p.parse_args()
    train, model, tok, identity, source_hash, versions = run.load_model()
    objective = run._load_module('flare_objective_evaluation', HERE / 'model.py')
    if args.arm != 'source':
        run.load_checkpoint(model, HERE / args.arm / 'checkpoint.pt', train, args.arm)
    model.eval()
    torch = train.torch
    device = model.time.weight.device
    ev = run._load_evaluator()
    score = ev.RUNNER._load_score_module()
    _, _, train_rows, val_rows = train.read_frozen_data(ev.DATA)
    selected = run.select_pilot_rows(train_rows, val_rows)
    hybrid_eval = run._load_module('old_hybrid_sampling_helpers',
        run.ROOT / 'experiments/structure_conditioned_hybrid8b_20261018/evaluate.py')
    state_hash = train.trainable_sha256(model)
    out = HERE / 'evaluation' / args.arm
    out.mkdir(parents=True, exist_ok=True)
    results = []
    modes = ['ar', 'diffusion'] if args.arm in ('source', 'flare') else ['ar']
    for split, rows in zip(['train', 'val'], selected):
        for index, row in enumerate(rows):
            example = train.tasks(row)[0]
            prefix = tok.encode(example['prefix'], add_special_tokens=False) + tok.encode(example['lead'], add_special_tokens=False)
            assert prefix[-1] == model.start_id
            for mode in modes:
                path = out / f'{split}-{index:02d}-{mode}.json'
                if path.exists():
                    r = json.loads(path.read_text()); assert r['state_hash'] == state_hash
                    results.append(r); continue
                g = torch.Generator(device=device).manual_seed(42 + index)
                raw, forwards = [], 0
                started = time.monotonic()
                banned = set(tok.all_special_ids) - {model.end_id}
                if mode == 'ar':
                    sampler = ev.RUNNER.CachedCausalSampler(model,tok)
                    sampler.prefill(prefix)
                    for _ in range(train.K):
                        token = sampler.sample(g,banned)
                        raw.append(token); forwards += 1
                        if token == model.end_id: break
                        sampler.append(token)
                else:
                    while len(raw) < train.K and model.end_id not in raw:
                        current = [model.mask_id] * min(objective.BLOCK_SIZE, train.K-len(raw))
                        for step in range(len(current)):
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                logits = objective.denoise_logits(model,prefix+raw,current,len(prefix))
                                positions = [i for i,t in enumerate(current) if t==model.mask_id]
                                sampled = hybrid_eval._sample_rows(logits[positions],g,banned,.8,.95)
                                probabilities = logits[positions].float()/.8
                                probabilities[:,list(banned)] = -torch.inf
                                confidence = probabilities.softmax(-1).gather(1,sampled[:,None]).squeeze(1)
                                best = int(confidence.argmax())
                                current[positions[best]] = int(sampled[best])
                            forwards += 1
                        raw += current
                    if model.end_id in raw: raw=raw[:raw.index(model.end_id)+1]
                ended = bool(raw and raw[-1]==model.end_id)
                body = tok.decode(raw[:-1] if ended else raw,skip_special_tokens=False)
                audit = score.audit(body) if ended else dict(valid=False,reasons=['END not generated within capacity'])
                r=dict(arm=args.arm,mode=mode,split=split,material_id=row['material_id'],seed=42+index,
                    state_hash=state_hash,source_hash=source_hash,reference_lead=True,raw_tokens=raw,body=body,
                    ended=ended,audit=audit,valid=audit.get('valid') is True,forwards=forwards,
                    seconds=time.monotonic()-started,requested_sg=int(row['core_facts']['space_group_number']))
                if r['valid']:
                    structure=score.parse_fn_wyckoff(body,to_cif=False)
                    r['actual_sg']=int(score.SpacegroupAnalyzer(structure,symprec=.01,angle_tolerance=5).get_space_group_number())
                    r['sg_match']=r['actual_sg']==r['requested_sg']
                run._write_json(path,r);results.append(r)
                print(args.arm,split,index,mode,r['valid'],flush=True)
    summary=dict(arm=args.arm,state_hash=state_hash,property_evaluation='not_run',
        scope='eight train/eight held-out, reference text prefixes; overfit mechanism probe',results={})
    for split in ['train','val']:
        for mode in modes:
            rows=[r for r in results if r['split']==split and r['mode']==mode]
            assert len(rows)==8
            summary['results'][f'{split}-{mode}']=dict(n=8,valid=sum(r['valid'] for r in rows),
                ended=sum(r['ended'] for r in rows),sg_match=sum(r.get('sg_match',False) for r in rows))
    run._write_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
