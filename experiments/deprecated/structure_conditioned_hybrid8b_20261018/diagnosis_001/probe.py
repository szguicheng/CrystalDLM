"""Bounded diagnosis of the frozen checkpoint; no optimizer or output repair."""
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hybrid_eval_diagnosis", HERE.parent / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    train, model, tok, identity, state_hash, versions = ev._load_model(SimpleNamespace(
        model_path=ev.MODEL, data=ev.DATA, checkpoint=ev.DEFAULT_CHECKPOINT))
    torch = train.torch
    device = model.time.weight.device
    _, _, train_rows, val_rows = train.read_frozen_data(ev.DATA)
    selected = {name: random.Random(42).sample(sorted(rows, key=lambda r: r['material_id']), 8)
                for name, rows in [('train', train_rows), ('val', val_rows)]}
    manifest = {name: [r['material_id'] for r in rows] for name, rows in selected.items()}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = dict(manifest=manifest, checkpoint=state_hash, versions=versions, denoising=[], generation=[])
    def save():
        (HERE / 'probe-results.json').write_text(json.dumps(results, indent=2))
    for split, rows in selected.items():
        for index, row in enumerate(rows):
            example = train.tasks(row)[0]
            for k in [8, 16, 24, 32]:
                packed = model.pack(dict(example, noise_step=k), tok, 'hybrid',
                                    torch.Generator().manual_seed(4200 + index))
                group = next(g for g in packed['groups'] if g['kind'] == 'crystal')
                ids = packed['ids'].to(device)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = model.hidden(ids[None], packed['allowed'].to(device),
                                          packed['times'][None].to(device))[0]
                    logits = model.base.lm_head(hidden[group['positions']]).float()
                    labels = torch.tensor(group['labels'], device=device)
                    ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                    correct = logits.argmax(-1) == labels
                    record = dict(split=split, material_id=row['material_id'], k=k, groups={})
                    for name, mask in [('content', (labels != model.end_id) & (labels != model.pad_id)),
                                       ('END', labels == model.end_id), ('PAD', labels == model.pad_id)]:
                        record['groups'][name] = dict(n=int(mask.sum()), correct=int(correct[mask].sum()),
                                                     ce_sum=float(ce[mask].sum()))
                    weights = torch.tensor(group['weights'], device=device)
                    record['weighted_training_loss'] = float((ce * weights).sum() / group['normalizer'])
                results['denoising'].append(record)
                del hidden, logits, ce, ids
            save()
            print('denoising', split, row['material_id'], flush=True)
    score = ev._load_module('diagnosis_score', ev.ROOT / 'experiments/interleaved_ar_diffusion_20261002/score.py')
    for row in selected['val'][:4]:
        example = train.tasks(row)[0]
        prefix = tok.encode(example['prefix'], add_special_tokens=False) + tok.encode(example['lead'], add_special_tokens=False)
        assert prefix[-1] == model.start_id
        raw, forwards = ev._diffuse_crystal(model, prefix, train.K, train.STEPS,
            torch.Generator(device=device).manual_seed(42), set(tok.all_special_ids) - {model.end_id, model.pad_id}, .8, .95)
        record = dict(material_id=row['material_id'], reference_lead=True, raw_tokens=raw,
                      raw_text=tok.decode(raw, skip_special_tokens=False), forward_steps=forwards)
        try:
            text = ev.RUNNER._boundary_content(raw, model, tok)
            record.update(boundary_ok=True, audit=score.audit(text))
        except ev.RUNNER.GenerationFailure as error:
            record.update(boundary_ok=False, error=str(error))
        results['generation'].append(record)
        save()
        print('generation', row['material_id'], record.get('boundary_ok'), record.get('audit'), flush=True)
    assert len(results['denoising']) == 64 and len(results['generation']) == 4
    results['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
