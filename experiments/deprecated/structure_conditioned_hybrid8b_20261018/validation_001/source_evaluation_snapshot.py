"""Fixed-prefix diagnostic evaluation; all failures remain in the denominator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['source', 'control', 'highmask'], required=True)
    parser.add_argument('--samplers', nargs='+', choices=['original', 'confidence', 'remask'], required=True)
    args = parser.parse_args()
    ev = load('frozen_hybrid_evaluation', HERE.parent / 'evaluate.py')
    sampling = load('validation_sampling', HERE / 'sampling.py')
    train, model, tok, identity, state_hash, versions = ev._load_model(SimpleNamespace(
        model_path=ev.MODEL, data=ev.DATA, checkpoint=ev.DEFAULT_CHECKPOINT))
    if args.arm != 'source':
        continuation = load('validation_continuation', HERE / 'continuation.py')
        continuation.load_continuation(model, HERE / args.arm / 'checkpoint.pt', train)
        state_hash = train.trainable_sha256(model)
    model.eval()
    torch = train.torch
    device = model.time.weight.device
    score = load('validation_score', ev.ROOT / 'experiments/interleaved_ar_diffusion_20261002/score.py')
    plan_path = HERE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert len(plan['requests']) == 28
    sources = [Path(__file__), HERE / 'sampling.py']
    if args.arm != 'source':
        sources.append(HERE / 'continuation.py')
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for sampler in args.samplers:
        out = HERE / 'evaluation' / f'{args.arm}-{sampler}'
        out.mkdir(parents=True, exist_ok=True)
        receipt_identity = dict(arm=args.arm, sampler=sampler, state_sha256=state_hash,
                                plan_sha256=plan_hash, source_hashes=source_hashes)
        results = []
        for request in plan['requests']:
            path = out / f"{request['request_id']}-{request['attempt_id']:03d}.json"
            if path.exists():
                receipt = json.loads(path.read_text())
                assert receipt['identity'] == receipt_identity
                results.append(receipt)
                continue
            prefix = tok.encode(request['prompt'], add_special_tokens=False) + request['lead_token_ids']
            assert prefix[-1] == model.start_id and len(prefix) + train.K <= train.BASELINE.MAXLEN
            generator = torch.Generator(device=device).manual_seed(request['seed'])
            banned = set(tok.all_special_ids) - {model.end_id, model.pad_id}
            started = time.monotonic()
            if sampler == 'original':
                raw, forwards = ev._diffuse_crystal(model, prefix, train.K, train.STEPS,
                    generator, banned, .8, .95)
                trace = None
            else:
                raw, trace = sampling.diffuse(model, prefix, train.K, train.STEPS,
                    generator, banned, .8, .95, ev._sample_rows, revisit=sampler == 'remask')
                forwards = len(trace)
            receipt = dict(identity=receipt_identity, request_id=request['request_id'],
                attempt_id=request['attempt_id'], seed=request['seed'],
                requested_sg=request['space_group_number_requested'], raw_tokens=raw,
                raw_text=tok.decode(raw, skip_special_tokens=False), trace=trace,
                forward_steps=forwards, generation_seconds=time.monotonic()-started,
                boundary_ok=False, structural_valid=False, requested_sg_match=False)
            try:
                body = ev.RUNNER._boundary_content(raw, model, tok)
                receipt['boundary_ok'] = True
                receipt['audit'] = score.audit(body)
                receipt['structural_valid'] = receipt['audit'].get('valid') is True
                if receipt['structural_valid']:
                    structure = score.parse_fn_wyckoff(body, to_cif=False)
                    actual = int(score.SpacegroupAnalyzer(structure, symprec=.01, angle_tolerance=5).get_space_group_number())
                    receipt.update(actual_sg=actual, requested_sg_match=actual == receipt['requested_sg'],
                                   cif=structure.to(fmt='cif'))
            except ev.RUNNER.GenerationFailure as error:
                receipt['error'] = str(error)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(receipt, indent=2))
            tmp.replace(path)
            results.append(receipt)
            print(args.arm, sampler, len(results), 'valid', receipt['structural_valid'], flush=True)
        summary = dict(identity=receipt_identity, completed=len(results),
            valid=sum(r['structural_valid'] for r in results),
            boundary_pass=sum(r['boundary_ok'] for r in results),
            valid_requested_sg=sum(r['structural_valid'] and r['requested_sg_match'] for r in results),
            generation_seconds=sum(r['generation_seconds'] for r in results),
            property_scoring='not_run', scope='fixed frozen original-model text prefixes; diagnostic, not end-to-end')
        assert len(results) == 28
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
