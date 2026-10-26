"""Post-run H checkpoint probe: oracle mask recovery and official free sampling."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from adapt import fingerprint
from model import Hybrid, prediction_positions, unpack_block
from prepare import K, STEPS, sha, tasks, write_json
from run import attempt, generate, records
from score import audit

ROOT = Path(__file__).resolve().parents[1]
MASK_SEED = 31415
SAMPLE_SEEDS = [17, 29, 43]


def freeze_materials(data, smoke_data, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    training = records(smoke_data, 'train')
    validation = records(data, 'val')
    by_id = {row['id']: row for row in validation}
    ids = manifest['validation_ids']
    groups = manifest['validation_groups']
    assert len(training) == 8 and len({r['id'] for r in training}) == 8
    assert set(r['id'] for r in training) <= set(manifest['train_ids'])
    assert len(ids) == 128 and len(set(ids)) == 128 and set(ids) == set(by_id)
    assert len(groups['original64']) == len(groups['later64']) == 64
    assert groups['original64'] + groups['later64'] == ids
    assert not set(r['id'] for r in training) & set(ids)
    return training, [by_id[i] for i in ids], groups


def accuracy(predictions, target, selected):
    count = int(selected.sum())
    correct = int(((predictions == target) & selected).sum())
    return dict(correct=correct, count=count, accuracy=correct / count if count else None)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sample_class(generation, checked):
    if not generation['ok']:
        error = generation['error'].lower()
        if 'trigger' in error:
            return 'trigger_boundary'
        if 'end' in error or 'pad' in error:
            return 'end_pad_boundary'
        return 'generation_error'
    if checked['valid']:
        return 'valid'
    if any('condition_mismatch' in reason for reason in checked['reasons']):
        return 'condition'
    return 'structure' if 'facts' in checked else 'parse'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'), 'Use an allocated compute GPU'
    assert not args.out.exists(), 'Never overwrite diagnostic evidence'
    torch.set_num_threads(4)

    cohort_root = args.checkpoint.parent.parent
    cohort_manifest_path = cohort_root / 'manifest.json'
    smoke_data = args.data.parent / 'smoke_data_002'
    training, validation, groups = freeze_materials(args.data, smoke_data, cohort_manifest_path)
    manifest = json.loads(cohort_manifest_path.read_text())
    data_manifest_path = args.data / 'manifest.json'
    data_manifest_sha256 = sha(data_manifest_path.read_bytes())
    data_manifest = json.loads(data_manifest_path.read_text())
    assert manifest['source_manifest_sha256'] == data_manifest_sha256
    assert data_manifest['capacity'] == K and data_manifest['steps'] == STEPS
    smoke_manifest_path = smoke_data / 'manifest.json'
    smoke_manifest = json.loads(smoke_manifest_path.read_text())
    assert smoke_manifest['status'] == 'frozen' and smoke_manifest['sizes']['train'] == 8
    assert smoke_manifest['selection']['train']['ids'] == [r['id'] for r in training]

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    provenance = saved['config']
    assert saved['update'] == 4096 and provenance['mode'] == 'hybrid' and provenance['phase'] == 'joint'
    source_hashes = provenance['source_sha256']
    snapshot = cohort_root / 'source_snapshot'
    assert all(sha((snapshot / name).read_bytes()) == digest for name, digest in source_hashes.items())
    for name in ['adapt.py', 'model.py', 'prepare.py', 'run.py', 'score.py']:
        assert sha((ROOT / name).read_bytes()) == source_hashes[name]

    tok = AutoTokenizer.from_pretrained(args.data / 'tokenizer')
    model = Hybrid.pretrained(tok)
    model.restore_checkpoint(saved, expected_mode='hybrid')
    del saved
    model = model.cuda().eval()
    before = fingerprint(model)
    assert before == provenance['final_fingerprint']

    rows = [(row, 'smoke_train8') for row in training]
    group_for = {material_id: name for name, ids in groups.items() for material_id in ids}
    rows.extend((row, group_for[row['id']]) for row in validation)
    args.out.mkdir(parents=True, exist_ok=False)
    probe_manifest = dict(status='running', checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha(args.checkpoint), checkpoint_update=4096,
        checkpoint_config=provenance, source_sha256=source_hashes,
        source_snapshot_verified=True, diagnostic_source_sha256=sha(Path(__file__).read_bytes()),
        model_fingerprint_before=before, model_fingerprint_after=None,
        data_manifest_sha256=data_manifest_sha256,
        smoke_manifest_sha256=sha(smoke_manifest_path.read_bytes()),
        cohort_manifest_sha256=sha(cohort_manifest_path.read_bytes()),
        cohorts={name: ids for name, ids in [('smoke_train8', [r['id'] for r in training]),
            ('original64', groups['original64']), ('later64', groups['later64'])]},
        physical_audit=manifest['physical_audit'],
        recovery=dict(mask_seed=MASK_SEED, k=[16, 32], lead='oracle reference trigger'),
        sampling=dict(seeds=SAMPLE_SEEDS, temperature=.8, steps=STEPS, capacity=K,
            path='run.generate -> model.text(trigger=True) -> model.crystal'),
        updates_applied=0, retries=0, answer_repair=False, test_used=False,
        note='Recovery uses gold lead and gold-reference tokens; token accuracy is not a unique generation target. Three seeds are repeated measurements per material, not independent materials.')
    write_json(args.out / 'manifest.json', probe_manifest)

    recovery_rng = torch.Generator(device='cuda').manual_seed(MASK_SEED)
    recovery_rows = []
    with (args.out / 'recovery.jsonl').open('w') as log:
        for row, cohort in rows:
            ex = tasks(row, 'val', 'plus')[0]
            prefix = tok.encode(ex['prefix'] + ex['lead'], add_special_tokens=False)
            content = tok.encode(ex['crystal'], add_special_tokens=False) + [model.end_id]
            assert len(content) <= K
            target = torch.tensor(content + [model.pad_id] * (K - len(content)), device='cuda')
            for k in [16, 32]:
                hidden = torch.rand(K, device='cuda', generator=recovery_rng) < k / STEPS
                masked = torch.where(hidden, model.mask_id, target)
                start = len(prefix)
                ids = torch.cat([torch.tensor(prefix, device='cuda'), masked])
                positions = hidden.nonzero().flatten() + start
                offset = model.prediction_offset
                span = (start-offset, start+K)
                queries = prediction_positions(start, positions.sub(start).tolist(), 'hybrid', offset)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model.logits(ids, 'hybrid', span, k, queries).float()
                logits[:, list(set(tok.all_special_ids) - {model.end_id, model.pad_id})] = -torch.inf
                predictions = logits.argmax(-1)
                completed = target.clone()
                completed[hidden] = predictions
                body = hidden & (target != model.pad_id) & (target != model.end_id)
                padding = hidden & (target == model.pad_id)
                ending = hidden & (target == model.end_id)
                item = dict(cohort=cohort, id=row['id'], k=k,
                    masked_body=accuracy(predictions, target[hidden], body[hidden]),
                    masked_padding=accuracy(predictions, target[hidden], padding[hidden]),
                    masked_end=accuracy(predictions, target[hidden], ending[hidden]),
                    reference_tokens=target.tolist(), masked_positions=positions.sub(start).tolist(),
                    predicted_tokens=predictions.tolist(), completed_tokens=completed.tolist())
                try:
                    text = tok.decode(unpack_block(completed.tolist(), model.end_id, model.pad_id), skip_special_tokens=False)
                    item['recovered_text'] = text
                    item['physical_audit'] = audit(text, row['condition'])
                except ValueError as error:
                    item.update(recovered_text=None, reconstruction_error=str(error),
                        physical_audit=dict(valid=False, reasons=['end_pad_boundary']))
                log.write(json.dumps(item, allow_nan=False) + '\n')
                log.flush()
                recovery_rows.append(item)

    sampling_rows = []
    with (args.out / 'sampling.jsonl').open('w') as log:
        for row, cohort in rows:
            prefix = tasks(row, 'val', 'plus')[0]['prefix']
            for seed in SAMPLE_SEEDS:
                rng = torch.Generator(device='cuda').manual_seed(seed)
                generation = attempt(lambda: generate(model, tok, 'hybrid', prefix, rng))
                checked = audit(generation['value']['crystal'], row['condition']) if generation['ok'] else None
                item = dict(cohort=cohort, id=row['id'], seed=seed, generation=generation,
                    physical_audit=checked, failure_class=sample_class(generation, checked or {}))
                log.write(json.dumps(item, allow_nan=False) + '\n')
                log.flush()
                sampling_rows.append(item)

    after = fingerprint(model)
    assert after == before, 'Diagnostic changed checkpoint weights'
    probe_manifest.update(status='complete', model_fingerprint_after=after)
    write_json(args.out / 'manifest.json', probe_manifest)

    recovery_summary = {}
    for cohort in ['smoke_train8', 'original64', 'later64']:
        for k in [16, 32]:
            subset = [r for r in recovery_rows if r['cohort'] == cohort and r['k'] == k]
            values = {}
            for field in ['masked_body', 'masked_padding', 'masked_end']:
                correct = sum(r[field]['correct'] for r in subset)
                count = sum(r[field]['count'] for r in subset)
                values[field] = dict(correct=correct, count=count, accuracy=correct / count if count else None)
            values.update(materials=len({r['id'] for r in subset}), physical_audit_valid=sum(r['physical_audit']['valid'] for r in subset))
            recovery_summary[f'{cohort}/k{k}'] = values
    sampling_summary = {}
    for cohort in ['smoke_train8', 'original64', 'later64']:
        subset = [r for r in sampling_rows if r['cohort'] == cohort]
        per_material = Counter(r['id'] for r in subset if r['failure_class'] == 'valid')
        sampling_summary[cohort] = dict(unique_materials=len({r['id'] for r in subset}),
            repeated_seed_attempts=len(subset), failure_class_attempt_counts=dict(Counter(r['failure_class'] for r in subset)),
            materials_by_valid_seed_count=dict(Counter(per_material[material_id] for material_id in {r['id'] for r in subset})))
    write_json(args.out / 'summary.json', dict(status='complete', unique_materials=136,
        sampling_attempts=408, recovery=recovery_summary, sampling=sampling_summary,
        model_fingerprint_before=before, model_fingerprint_after=after,
        note='Sampling seeds are repeated measurements; denominators for attempts must not be described as independent materials.'))


if __name__ == '__main__':
    main()
