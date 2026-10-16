"""Fixed continuation of predictor_align_1024_002 with its original V8 seed."""
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
from model import Hybrid
from prepare import ROOT, sha, write_json
from rebalance import evaluation, joint_stream
from rescore import score_evaluation
from run import records

UPDATES = 1024
BATCH_SIZE = 16
DATA_MANIFEST_SHA256 = 'e3a5a86e58eaae871fa2718d3c9f646551cb297047ef724bf841b24bc48c0004'
V8_FINGERPRINT = '7567d5dd878a1e02a51b05a9ec4ecd67e68cd1a79da88235c975e1eeb4b21c85'
AR_FINGERPRINT = '45684803b4513e4ac6c086c8e8440990b2cbefd323380530174fd0957c036242'
AR_CHECKPOINT_SHA256 = '826d1cfdc7fd26c684706ff6f258ba89aaef9680b754b1024c4a4d36b6c276bb'
HSAME_FINGERPRINT = 'b5ed6bff994d9cf227e31ae983a187f9ed07ef7ceacffe1454ceca38fda391b0'
HSAME_PARTIAL_SHA256 = '420f3914bea8aa0203b5ff318d2da3d5f24f8da16ee21804b91177f7766c9e17'
MIGRATED_SOURCES = {'predictor_align.py', 'score.py', 'test_checks.py'}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_exposure(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def freeze_exposure(rows, out):
    source = joint_stream(rows, 17)
    tasks, materials, batches = Counter(), Counter(), Counter()
    path = out / 'training_exposure.jsonl'
    with path.open('w') as log:
        for _ in range(UPDATES):
            batch = [next(source) for _ in range(BATCH_SIZE)]
            ids = {example['source_id'] for example in batch}
            counts = Counter(example['task'] for example in batch)
            assert len(ids) == 1 and counts['G'] == 8
            assert counts['E'] + counts['E_varied'] == 4
            assert counts['Q1'] + counts['Q2'] + counts['Q_varied'] == 4
            material = next(iter(ids))
            materials[material] += len(batch)
            batches[material] += 1
            tasks.update(counts)
            for example in batch:
                log.write(json.dumps(example, allow_nan=False) + '\n')
    assert len(materials) == 1024 and set(batches.values()) == {1}
    result = dict(status='frozen', updates=UPDATES, examples=UPDATES*BATCH_SIZE,
        optimizer_batch_size=BATCH_SIZE, unique_materials=len(materials),
        batches_per_material={str(k): v for k, v in Counter(batches.values()).items()},
        examples_per_material={str(k): v for k, v in Counter(materials.values()).items()},
        tasks=dict(tasks), sha256=file_sha(path),
        note='First full pass of rebalance.joint_stream(seed=17); each batch contains one condition material.')
    assert result['examples'] == 16384 and result['examples_per_material'] == {'16': 1024}
    write_json(out / 'exposure_audit.json', result)
    return result


def restore_checkpoint(model, saved, expected_mode, expected_offset):
    config = model.restore_checkpoint(saved, expected_mode, expected_offset)
    assert fingerprint(model) == config['final_fingerprint']
    return model


def scientific_output(record):
    return {key: value for key, value in record.items() if key not in {'seconds', 'forward_calls'}}


def compare_hsame_prefix(previous_rows, replay_rows, validation_ids):
    assert len(previous_rows) == 66 and len(replay_rows) == 128
    assert [row['id'] for row in previous_rows] == validation_ids[:66]
    assert [row['id'] for row in replay_rows] == validation_ids
    mismatch = None
    for old, new in zip(previous_rows, replay_rows):
        old_science, new_science = scientific_output(old), scientific_output(new)
        if old_science != new_science:
            mismatch = dict(id=old['id'], fields={key: dict(previous=old_science.get(key), replay=new_science.get(key))
                for key in sorted(old_science.keys() | new_science.keys())
                if old_science.get(key) != new_science.get(key)})
            break
    return dict(rows=66, validation_ids_match=True, scientific_fields_exact=mismatch is None,
        first_mismatch=mismatch,
        forward_call_differences=[dict(id=old['id'], previous=old.get('forward_calls'), replay=new.get('forward_calls'))
            for old, new in zip(previous_rows, replay_rows)
            if old.get('forward_calls') != new.get('forward_calls')])


def verify_predecessors(data, initial):
    old = ROOT / 'predictor_align_1024_002'
    old_manifest = read_json(old / 'manifest.json')
    submission = read_json(ROOT / 'submission_predictor_align_1024_002.json')
    exception = ROOT / 'execution_exception_predictor_align_1024_002.json'
    exception_record = read_json(exception)
    v8_manifest = read_json(initial.parent / 'manifest.json')
    v8_status = read_json(initial / 'status.json')
    data_manifest = read_json(data / 'manifest.json')

    assert old_manifest['updates'] == UPDATES and old_manifest['seed'] == 17
    assert old_manifest['optimizer'] == 'AdamW'
    assert old_manifest['arms'] == [
        dict(name='ar_control', mode='ar', prediction_offset=0),
        dict(name='hybrid_same', mode='hybrid', prediction_offset=0),
        dict(name='hybrid_shift', mode='hybrid', prediction_offset=1)]
    assert old_manifest['initial_checkpoint'] == 'rebalance_001/ar/checkpoint.pt'
    assert old_manifest['initial_checkpoint_sha256'] == '3f83d86d6e955c538741cd7e981bcbd413d86783a685ee0f8cd836d7b3ed3678'
    assert old_manifest['initial_mode'] == 'ar' and old_manifest['initial_updates'] == 4096
    assert old_manifest['initial_fingerprint'] == V8_FINGERPRINT
    assert old_manifest['initial_source_snapshot_verified'] is True
    assert old_manifest['data_manifest_sha256'] == DATA_MANIFEST_SHA256
    assert old_manifest['test_used'] is False and old_manifest['answer_repair'] is False
    assert old_manifest['retries'] == 0
    assert submission['data_manifest_sha256'] == DATA_MANIFEST_SHA256
    assert submission['source_sha256'] == old_manifest['migration']['source_sha256']
    assert exception_record['slurm_job'] == '601858'
    assert exception_record['failure_context']['hybrid_shift_started'] is False

    assert sha((data / 'manifest.json').read_bytes()) == DATA_MANIFEST_SHA256
    assert v8_manifest['source_manifest_sha256'] == DATA_MANIFEST_SHA256
    assert data_manifest['sizes']['train'] == 1024 and data_manifest['sizes']['val'] == 128
    assert v8_status['status'] == 'validated' and v8_status['mode'] == 'ar'
    assert v8_status['completed_updates'] == v8_status['updates'] == 4096
    assert v8_status['final_fingerprint'] == V8_FINGERPRINT
    assert old_manifest['initial_source_sha256'] == v8_status['source_sha256']
    v8_snapshot = initial.parent / 'source_snapshot'
    for name, digest in old_manifest['initial_source_sha256'].items():
        assert file_sha(v8_snapshot / name) == digest

    old_sources = old_manifest['migration']['source_sha256']
    old_snapshot = old / 'source_snapshot'
    current_sources = {path.name: sha(path.read_bytes()) for path in ROOT.glob('*.py')}
    assert set(current_sources) == set(old_sources)
    for name, digest in old_sources.items():
        assert file_sha(old_snapshot / name) == digest
        if name in MIGRATED_SOURCES:
            assert current_sources[name] != digest
        else:
            assert current_sources[name] == digest
    exposure = old / 'training_exposure.jsonl'
    exposure_sha = file_sha(exposure)
    exposure_audit = read_json(old / 'exposure_audit.json')
    assert exposure_sha == old_manifest['exposure_sha256'] == exposure_audit['sha256']
    tasks = Counter()
    rows = 0
    with exposure.open() as source:
        for line in source:
            tasks.update([json.loads(line)['task']])
            rows += 1
    assert rows == UPDATES * BATCH_SIZE == 16384
    assert tasks['G'] == 8192
    assert tasks['E'] + tasks['E_varied'] == 4096
    assert tasks['Q1'] + tasks['Q2'] + tasks['Q_varied'] == 4096
    return dict(old=old, old_manifest=old_manifest, exception=exception,
        exception_sha256=file_sha(exception), exposure=exposure,
        exposure_sha256=exposure_sha, current_sources=current_sources)


def verify_run_evidence(old, old_manifest, data):
    ar_dir, same_dir = old / 'ar_control', old / 'hybrid_same'
    ar = read_json(ar_dir / 'status.json')
    same = read_json(same_dir / 'status.json')
    summary = read_json(old / 'summary.json')
    assert summary['status'] == 'partial' and 'ar_control' in summary['results']
    assert not (old / 'hybrid_shift').exists()

    for receipt, mode, offset, status, final in [
        (ar, 'ar', 0, 'validated', AR_FINGERPRINT),
        (same, 'hybrid', 0, 'trained', HSAME_FINGERPRINT)]:
        assert receipt['mode'] == mode and receipt['prediction_offset'] == offset
        assert receipt['phase'] == 'joint' and receipt['feedback'] is True
        assert receipt['updates'] == receipt['completed_updates'] == UPDATES
        assert receipt['seed'] == 17
        assert receipt['initial_fingerprint'] == receipt['warm_fingerprint'] == V8_FINGERPRINT
        assert receipt['final_fingerprint'] == final
        assert receipt['source_sha256'] == old_manifest['migration']['source_sha256']

    assert ar['checkpoint_sha256'] == AR_CHECKPOINT_SHA256
    assert ar['checkpoint_reload_verified'] is True
    assert ar['validation']['v2']['n'] == 128
    assert ar['reading']['n'] == 128 and ar['old8_regression']['v2']['n'] == 8
    assert same['status'] == 'trained'
    assert len((ar_dir / 'curve.jsonl').read_text().splitlines()) == UPDATES
    assert len((same_dir / 'curve.jsonl').read_text().splitlines()) == UPDATES

    val = sorted(records(data, 'val'), key=lambda row: sha(row['id'].encode()))
    train = records(data, 'train')
    old8 = records(ROOT / 'smoke_data_002', 'train')
    assert len(train) == 1024 and len(val) == 128 and len(old8) == 8
    assert not {row['id'] for row in train} & {row['id'] for row in val}
    materials = {row['id']: row for row in train + val}

    ar_raw_path = ar_dir / 'validation_128' / 'evaluation.jsonl'
    ar_rows = [json.loads(line) for line in ar_raw_path.read_text().splitlines()]
    ar_old_summary = read_json(ar_dir / 'validation_128' / 'rescored_summary_v2.json')
    assert len(ar_rows) == 128 and [row['id'] for row in ar_rows] == [row['id'] for row in val]
    assert file_sha(ar_raw_path) == ar_old_summary['raw_sha256']
    assert ar_old_summary['counts'] == ar['validation']['v2']['counts']
    assert ar_rows and len({row['id'] for row in ar_rows}) == 128

    same_raw_path = same_dir / 'validation_128' / 'evaluation.jsonl'
    same_rows = [json.loads(line) for line in same_raw_path.read_text().splitlines()]
    assert file_sha(same_raw_path) == HSAME_PARTIAL_SHA256
    assert len(same_rows) == 66 and len({row['id'] for row in same_rows}) == 66
    assert [row['id'] for row in same_rows] == [row['id'] for row in val[:66]]
    assert not same.get('checkpoint_sha256') and not same.get('checkpoint_reload_verified')
    return dict(ar=ar, same=same, validation=val, train=train, old8=old8,
        materials=materials, ar_rows=ar_rows, same_rows=same_rows)


def strata(rows, groups):
    keys = ['G', 'Q_reference', 'E_reference', 'closed_loop', 'flow_complete', 'Q_swap', 'Q_no_crystal']
    return {name: dict(n=len(ids), counts={key: sum(row[key] for row in rows if row['id'] in set(ids))
        for key in keys}) for name, ids in groups.items()}


def paired_delta(same_rows, shifted_rows):
    assert len(same_rows) == len(shifted_rows) == 128
    assert [row['id'] for row in same_rows] == [row['id'] for row in shifted_rows]
    result = {}
    for key in ['G', 'closed_loop']:
        gain = sum(not same[key] and shifted[key] for same, shifted in zip(same_rows, shifted_rows))
        loss = sum(same[key] and not shifted[key] for same, shifted in zip(same_rows, shifted_rows))
        result[key] = dict(n=128, gain=gain, loss=loss, net=gain-loss,
            gain_rate=gain / 128, loss_rate=loss / 128, net_rate=(gain-loss) / 128)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'), 'Use an allocated compute GPU'
    assert args.data.name == 'data_002' and args.initial.name == 'ar'
    assert args.initial.parent.name == 'rebalance_001' and args.out.name == 'predictor_align_1024_003'
    assert not args.out.exists(), 'Never overwrite experiment evidence'
    torch.set_num_threads(4)

    evidence = verify_predecessors(args.data, args.initial)
    previous = verify_run_evidence(evidence['old'], evidence['old_manifest'], args.data)
    checkpoint_path = args.initial / 'checkpoint.pt'
    assert file_sha(checkpoint_path) == evidence['old_manifest']['initial_checkpoint_sha256']
    initial = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert initial['update'] == 4096 and initial['config']['mode'] == 'ar'
    assert 'prediction_offset' not in initial['config']
    assert initial['config']['final_fingerprint'] == V8_FINGERPRINT
    assert initial['config']['source_sha256'] == read_json(args.initial / 'status.json')['source_sha256']
    assert (evidence['old'] / 'hybrid_same' / 'checkpoint.pt').is_file()

    train, validation, old8 = previous['train'], previous['validation'], previous['old8']
    materials = previous['materials']
    validation_groups = {'original64': [row['id'] for row in validation[:64]],
                         'later64': [row['id'] for row in validation[64:]]}
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(evidence['exception'], args.out / evidence['exception'].name)
    source_snapshot = args.out / 'source_snapshot'
    source_snapshot.mkdir()
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, source_snapshot / source.name)
    manifest = dict(status='running', continuation='fixed002', updates=UPDATES, seed=17,
        optimizer='AdamW', batch_size=BATCH_SIZE, task_mix={'G': 8, 'E': 4, 'Q': 4},
        initial_checkpoint=str(checkpoint_path),
        initial_checkpoint_sha256=file_sha(checkpoint_path), initial_mode='ar',
        initial_updates=4096, initial_fingerprint=V8_FINGERPRINT,
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        exposure_source=str(evidence['exposure']), exposure_sha256=evidence['exposure_sha256'],
        source_sha256=evidence['current_sources'],
        validation_ids=[row['id'] for row in validation], validation_groups=validation_groups,
        train_ids=[row['id'] for row in train], old8_ids=[row['id'] for row in old8],
        test_used=False, answer_repair=False, retries=0,
        predecessor=dict(path='predictor_align_1024_002', manifest_sha256=file_sha(
            evidence['old'] / 'manifest.json'),
            exception_path=evidence['exception'].name,
            exception_sha256=evidence['exception_sha256']),
        actions=dict(ar_control='inherit complete raw and rescore; no sampling or training',
            hybrid_same='strictly load final offset-0 checkpoint and replay all 128 validation rows',
            hybrid_shift='train only this arm for 1024 updates from the shared V8 AR checkpoint'))
    write_json(args.out / 'manifest.json', manifest)

    # AR evidence is inherited verbatim, then rescored with the repaired scorer.
    ar_out = args.out / 'ar_control' / 'validation_128'
    ar_out.mkdir(parents=True)
    shutil.copy2(evidence['old'] / 'ar_control' / 'validation_128' / 'evaluation.jsonl',
                 ar_out / 'evaluation.jsonl')
    ar_summary, ar_scored = score_evaluation(ar_out, materials)
    assert ar_summary['n'] == 128
    assert ar_summary['counts'] == previous['ar']['validation']['v2']['counts']
    (ar_out / 'rescored_evaluation_v2.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in ar_scored))
    write_json(ar_out / 'rescored_summary_v2.json', dict(**ar_summary,
        raw_sha256=file_sha(ar_out / 'evaluation.jsonl'), scorer_sha256=sha((ROOT / 'score.py').read_bytes())))
    ar_result = dict(provenance='inherited_raw_rescored', validation=ar_summary,
        reading=previous['ar']['reading'], old8=previous['ar']['old8_regression']['v2'],
        initial_fingerprint=V8_FINGERPRINT, final_fingerprint=AR_FINGERPRINT,
        strata=strata(ar_scored, validation_groups),
        source_raw_sha256=file_sha(evidence['old'] / 'ar_control' / 'validation_128' / 'evaluation.jsonl'),
        prior_failure_context=read_json(evidence['exception'])['failure_context'])
    results = {'ar_control': ar_result}
    write_json(args.out / 'summary.json', dict(status='partial', results=results))

    tok = AutoTokenizer.from_pretrained(args.data / 'tokenizer')
    same_dir = evidence['old'] / 'hybrid_same'
    same_saved = torch.load(same_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
    same_model = Hybrid.pretrained(tok)
    restore_checkpoint(same_model, same_saved, expected_mode='hybrid', expected_offset=0)
    assert same_saved['config']['final_fingerprint'] == HSAME_FINGERPRINT
    assert fingerprint(same_model) == HSAME_FINGERPRINT
    same_model = same_model.cuda().eval()
    same_out = args.out / 'hybrid_same'
    same_out.mkdir()
    core = evaluation(same_model, tok, 'hybrid', validation, same_out / 'validation_128', materials)
    replay_rows = [json.loads(line) for line in (same_out / 'validation_128' / 'evaluation.jsonl').read_text().splitlines()]
    assert core['v2']['n'] == len(replay_rows) == 128
    prefix = compare_hsame_prefix(previous['same_rows'], replay_rows, [row['id'] for row in validation])
    prefix.update(previous_raw_sha256=HSAME_PARTIAL_SHA256,
        replay_raw_sha256=file_sha(same_out / 'validation_128' / 'evaluation.jsonl'))
    write_json(same_out / 'validation_128' / 'prefix_replay_comparison.json', prefix)
    if prefix['first_mismatch'] is not None:
        manifest['status'] = 'stopped_prefix_mismatch'
        manifest['prefix_replay'] = prefix
        write_json(args.out / 'manifest.json', manifest)
    assert prefix['first_mismatch'] is None, json.dumps(prefix['first_mismatch'], sort_keys=True)
    reading = unseen_reading(same_model, tok, 'hybrid', validation, same_out / 'validation_128')
    regression = evaluation(same_model, tok, 'hybrid', old8, same_out / 'old8_regression', materials, 'train')
    assert reading['n'] == 128 and regression['v2']['n'] == 8
    same_scored = [json.loads(line) for line in (same_out / 'validation_128' / 'rescored_evaluation_v2.jsonl').read_text().splitlines()]
    same_strata = strata(same_scored, validation_groups)
    write_json(same_out / 'validation_128' / 'stratified_summary.json', same_strata)
    same_receipt = dict(status='validated', provenance='checkpoint_replay', mode='hybrid',
        prediction_offset=0, updates=UPDATES, final_fingerprint=HSAME_FINGERPRINT,
        checkpoint_sha256=file_sha(same_dir / 'checkpoint.pt'), checkpoint_reload_verified=True,
        source_sha256=previous['same']['source_sha256'],
        replay_source_sha256=evidence['current_sources'],
        validation=core, reading=reading, old8_regression=regression, prefix_replay=prefix,
        strata=same_strata)
    write_json(same_out / 'status.json', same_receipt)
    results['hybrid_same'] = dict(provenance='replayed_existing_checkpoint',
        initial_fingerprint=V8_FINGERPRINT, final_fingerprint=HSAME_FINGERPRINT,
        checkpoint_sha256=same_receipt['checkpoint_sha256'], validation=core,
        reading=reading, old8=regression, prefix_replay=prefix, strata=same_strata)
    write_json(args.out / 'summary.json', dict(status='partial', results=results))
    del same_model, same_saved
    torch.cuda.empty_cache()

    shifted = Hybrid.pretrained(tok)
    restore_checkpoint(shifted, initial, expected_mode='ar', expected_offset=0)
    assert fingerprint(shifted) == V8_FINGERPRINT
    shifted.prediction_offset = 1
    assert shifted.prediction_offset == 1 and fingerprint(shifted) == V8_FINGERPRINT
    shifted = shifted.cuda().train()
    shift_out = args.out / 'hybrid_shift'
    shifted, receipt = train_phase(shifted, tok, 'hybrid', train, shift_out, 'joint', True, 17,
        warm_fingerprint=V8_FINGERPRINT, sequence=read_exposure(evidence['exposure']), updates=UPDATES)
    assert receipt['prediction_offset'] == 1
    assert all(file_sha(source_snapshot / name) == digest for name, digest in receipt['source_sha256'].items())
    del shifted
    del initial
    torch.cuda.empty_cache()
    saved = torch.load(shift_out / 'checkpoint.pt', map_location='cpu', weights_only=True)
    shifted = Hybrid.pretrained(tok)
    restore_checkpoint(shifted, saved, expected_mode='hybrid', expected_offset=1)
    assert fingerprint(shifted) == receipt['final_fingerprint']
    del saved
    shifted = shifted.cuda().eval()
    core = evaluation(shifted, tok, 'hybrid', validation, shift_out / 'validation_128', materials)
    reading = unseen_reading(shifted, tok, 'hybrid', validation, shift_out / 'validation_128')
    regression = evaluation(shifted, tok, 'hybrid', old8, shift_out / 'old8_regression', materials, 'train')
    shift_scored = [json.loads(line) for line in (shift_out / 'validation_128' / 'rescored_evaluation_v2.jsonl').read_text().splitlines()]
    assert core['v2']['n'] == len(shift_scored) == 128
    assert reading['n'] == 128 and regression['v2']['n'] == 8
    shift_strata = strata(shift_scored, validation_groups)
    write_json(shift_out / 'validation_128' / 'stratified_summary.json', shift_strata)
    receipt.update(status='validated', validation=core, reading=reading, old8_regression=regression,
        strata=shift_strata, checkpoint_sha256=file_sha(shift_out / 'checkpoint.pt'), checkpoint_reload_verified=True)
    write_json(shift_out / 'status.json', receipt)
    results['hybrid_shift'] = dict(provenance='new_training_from_v8_ar',
        initial_fingerprint=receipt['initial_fingerprint'], final_fingerprint=receipt['final_fingerprint'],
        prediction_offset=1, validation=core, reading=reading, old8=regression, strata=shift_strata)
    shifted_rows = shift_scored
    comparison = paired_delta(same_scored, shifted_rows)
    write_json(args.out / 'paired_shift_vs_hsame.json', comparison)
    manifest['status'] = 'complete'
    write_json(args.out / 'manifest.json', manifest)
    write_json(args.out / 'summary.json', dict(status='complete', results=results,
        validation_groups=validation_groups, paired_hshift_vs_hsame=comparison, test_used=False,
        note='AR inherited and rescored; Hsame checkpoint replayed with the first 66 rows matched; only Hshift was trained.'))


if __name__ == '__main__':
    main()
