"""Three-arm AR warm-start pilot for diffusion prediction alignment."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer

from adapt import fingerprint, train_phase
from cohort import unseen_reading
from model import Hybrid, LEGACY_OFFSET0_MODEL_SHA256
from prepare import ROOT, sha, write_json
from rebalance import evaluation, joint_stream
from run import records

UPDATES = 1024
BATCH_SIZE = 16
DATA_MANIFEST_SHA256 = 'e3a5a86e58eaae871fa2718d3c9f646551cb297047ef724bf841b24bc48c0004'
BRANCHES = [('ar_control', 'ar', 0), ('hybrid_same', 'hybrid', 0), ('hybrid_shift', 'hybrid', 1)]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore_checkpoint(model, saved, expected_mode, expected_offset):
    config = model.restore_checkpoint(saved, expected_mode, expected_offset)
    assert fingerprint(model) == config['final_fingerprint']
    return model


def freeze_exposure(rows, out):
    source = joint_stream(rows, 17)
    tasks = Counter(); materials = Counter(); batches = Counter(); path = out / 'training_exposure.jsonl'
    with path.open('w') as log:
        for _ in range(UPDATES):
            batch = [next(source) for _ in range(BATCH_SIZE)]
            ids = {e['source_id'] for e in batch}
            counts = Counter(e['task'] for e in batch)
            assert len(ids) == 1 and counts['G'] == 8
            assert counts['E'] + counts['E_varied'] == 4
            assert counts['Q1'] + counts['Q2'] + counts['Q_varied'] == 4
            material = next(iter(ids)); materials[material] += len(batch); batches[material] += 1; tasks.update(counts)
            for example in batch:
                log.write(json.dumps(example, allow_nan=False) + '\n')
    assert len(materials) == 1024 and set(batches.values()) == {1}
    result = dict(status='frozen', updates=UPDATES, examples=UPDATES*BATCH_SIZE,
        optimizer_batch_size=BATCH_SIZE, unique_materials=len(materials),
        batches_per_material={str(k):v for k,v in Counter(batches.values()).items()},
        examples_per_material={str(k):v for k,v in Counter(materials.values()).items()},
        tasks=dict(tasks), sha256=file_sha(path),
        note='First full pass of rebalance.joint_stream(seed=17); each batch contains one condition material.')
    assert result['examples'] == 16384 and result['examples_per_material'] == {'16': 1024}
    write_json(out / 'exposure_audit.json', result)
    return result


def read_exposure(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'), 'Use an allocated compute GPU'
    assert args.data.name == 'data_002' and args.initial.name == 'ar'
    assert args.initial.parent.name == 'rebalance_001' and args.out.name == 'predictor_align_1024_001'
    assert not args.out.exists(), 'Never overwrite experiment evidence'
    torch.set_num_threads(4)

    initial_status = json.loads((args.initial / 'status.json').read_text())
    assert initial_status['status'] == 'validated' and initial_status['mode'] == 'ar'
    assert initial_status['completed_updates'] == 4096
    checkpoint_path = args.initial / 'checkpoint.pt'
    checkpoint_sha256 = file_sha(checkpoint_path)
    initial = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    old_config = initial['config']; old_sources = old_config['source_sha256']
    old_snapshot = args.initial.parent / 'source_snapshot'
    assert initial['update'] == 4096 and old_config['mode'] == 'ar'
    assert 'prediction_offset' not in old_config, 'Expected the exact V8 AR checkpoint migration input'
    assert old_config['final_fingerprint'] == initial_status['final_fingerprint']
    assert old_sources == initial_status['source_sha256']
    assert old_sources['model.py'] == LEGACY_OFFSET0_MODEL_SHA256
    assert all(file_sha(old_snapshot / name) == digest for name, digest in old_sources.items())
    new_sources = {p.name: sha(p.read_bytes()) for p in ROOT.glob('*.py')}
    assert old_sources['model.py'] != new_sources['model.py']

    train = records(args.data, 'train')
    validation = sorted(records(args.data, 'val'), key=lambda row: sha(row['id'].encode()))
    old8 = records(ROOT / 'smoke_data_002', 'train')
    assert len(train) == 1024 and len(validation) == 128 and len(old8) == 8
    assert not {r['id'] for r in train} & {r['id'] for r in validation}
    materials = {r['id']: r for r in train + validation}
    data_manifest_sha = sha((args.data / 'manifest.json').read_bytes())
    data_manifest = json.loads((args.data / 'manifest.json').read_text())
    v8_manifest = json.loads((args.initial.parent / 'manifest.json').read_text())
    assert data_manifest_sha == DATA_MANIFEST_SHA256 == v8_manifest['source_manifest_sha256']
    assert data_manifest['sizes']['train'] == 1024 and data_manifest['sizes']['val'] == 128

    args.out.mkdir(parents=True, exist_ok=False)
    exposure = freeze_exposure(train, args.out)
    source_snapshot = args.out / 'source_snapshot'
    source_snapshot.mkdir()
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, source_snapshot / source.name)
    manifest = dict(status='running', updates=UPDATES, seed=17, optimizer='AdamW',
        arms=[dict(name=n, mode=m, prediction_offset=o) for n, m, o in BRANCHES],
        initial_checkpoint=str(checkpoint_path), initial_checkpoint_sha256=checkpoint_sha256,
        initial_mode='ar', initial_updates=4096, initial_fingerprint=old_config['final_fingerprint'],
        initial_source_sha256=old_sources, initial_source_snapshot_verified=True,
        migration=dict(old_model_sha256=old_sources['model.py'], new_model_sha256=new_sources['model.py'],
            source_sha256=new_sources), data_manifest_sha256=data_manifest_sha,
        train_ids=[r['id'] for r in train], validation_ids=[r['id'] for r in validation],
        validation_groups={'original64':[r['id'] for r in validation[:64]],
                           'later64':[r['id'] for r in validation[64:]]},
        old8_ids=[r['id'] for r in old8], exposure_sha256=exposure['sha256'],
        physical_audit='orbit_membership_v3', reading_audit='named_axis_v2',
        test_used=False, answer_repair=False, retries=0)
    write_json(args.out / 'manifest.json', manifest)

    tok = AutoTokenizer.from_pretrained(args.data / 'tokenizer')
    results = {}; initial_fingerprints = set()
    for name, mode, offset in BRANCHES:
        model = Hybrid.pretrained(tok)
        model.restore_checkpoint(initial, expected_mode='ar', expected_offset=0)
        assert fingerprint(model) == old_config['final_fingerprint']
        model.prediction_offset = offset
        assert model.prediction_offset == offset
        warm = fingerprint(model)
        assert warm == old_config['final_fingerprint']
        initial_fingerprints.add(warm)
        branch = args.out / name
        model, receipt = train_phase(model, tok, mode, train, branch, 'joint', True, 17,
            warm_fingerprint=warm, sequence=read_exposure(args.out / 'training_exposure.jsonl'), updates=UPDATES)
        assert receipt['prediction_offset'] == offset
        assert all(file_sha(source_snapshot / source) == digest for source, digest in receipt['source_sha256'].items())
        del model
        torch.cuda.empty_cache()

        branch_checkpoint = branch / 'checkpoint.pt'
        branch_checkpoint_sha = file_sha(branch_checkpoint)
        saved = torch.load(branch_checkpoint, map_location='cpu', weights_only=True)
        assert saved['config']['prediction_offset'] == offset
        model = restore_checkpoint(Hybrid.pretrained(tok), saved, mode, offset).cuda().eval()
        del saved
        core = evaluation(model, tok, mode, validation, branch / 'validation_128', materials)
        reading = unseen_reading(model, tok, mode, validation, branch / 'validation_128')
        regression = evaluation(model, tok, mode, old8, branch / 'old8_regression', materials, 'train')
        raw = [json.loads(line) for line in (branch / 'validation_128/rescored_evaluation_v2.jsonl').read_text().splitlines()]
        ids = {r['id'] for r in raw}
        strata = {group: dict(n=len(group_ids), counts={key:sum(r[key] for r in raw if r['id'] in set(group_ids))
            for key in ['G','Q_reference','E_reference','closed_loop','flow_complete','Q_swap','Q_no_crystal']})
            for group, group_ids in manifest['validation_groups'].items()}
        write_json(branch / 'validation_128/stratified_summary.json', strata)
        assert ids == set(manifest['validation_ids']) and len(raw) == 128
        receipt.update(status='validated', checkpoint_sha256=branch_checkpoint_sha,
            checkpoint_reload_verified=True, validation=core,
            reading=reading, old8_regression=regression, strata=strata)
        write_json(branch / 'status.json', receipt)
        results[name] = dict(validation=core, reading=reading, old8=regression, strata=strata,
            initial_fingerprint=warm, final_fingerprint=receipt['final_fingerprint'], prediction_offset=offset)
        write_json(args.out / 'summary.json', dict(status='partial', results=results))
        del model
        torch.cuda.empty_cache()

    assert len(initial_fingerprints) == 1
    write_json(args.out / 'summary.json', dict(status='complete', results=results,
        note='Three branches share one AR initialization and one 1024-batch frozen exposure pass. Validation was reused; test256 untouched.'))


if __name__ == '__main__':
    main()
