"""Choose a sampler on validation, then score the fixed untouched test cohort."""
from collections import Counter
import hashlib
import json
import os
import signal
import sys

import torch
from model import ROOT, Denoiser, load_data, render, seed_all, validate
from sampling_followup import variant
from followup_report import symmetry_audit, wp
from project_lattice import project
import evaluate

OUT = ROOT / 'evaluation'
METRICS = ['decoded', 'composition_match', 'usable', 'reference_match']


def score_program(record, reference, folder):
    rep = record['representation']
    evaluate.ROOT = folder / 'raw'
    (evaluate.ROOT / f'runs/{rep}_17').mkdir(parents=True, exist_ok=True)
    raw = evaluate.score_one((record, reference))
    projected = {key: False for key in METRICS}
    raw['basic_and_symmetry_consistent'] = False
    projected['basic_and_symmetry_consistent'] = False
    projection_error = None
    projected_program = None
    if raw['decoded']:
        try:
            signal.signal(signal.SIGALRM, evaluate.time_limit); signal.alarm(10)
            raw.update(symmetry_audit(record['program'], rep))
            raw['basic_and_symmetry_consistent'] = raw['usable'] and raw['symmetry_consistent']
            projected_program = project(record['program'], rep)
            signal.alarm(0)
            evaluate.ROOT = folder / 'projected'
            (evaluate.ROOT / f'runs/{rep}_17').mkdir(parents=True, exist_ok=True)
            projected = evaluate.score_one(({**record, 'program': projected_program}, reference))
            signal.alarm(10)
            projected.update(symmetry_audit(projected_program, rep))
            projected['basic_and_symmetry_consistent'] = projected['usable'] and projected['symmetry_consistent']
        except Exception as error:
            projection_error = f'{type(error).__name__}: {error}'
        finally: signal.alarm(0)
    return {'id': record['id'], 'raw': raw, 'projected': projected,
            'projection_error': projection_error, 'projected_program': projected_program}


def run_split(split, modes, count=512):
    summaries = []
    for rep in ['explicit', 'wyckoff']:
        data, records, vocab = load_data(split, rep)
        checkpoint = torch.load(ROOT / f'runs/{rep}_17/best.pt', map_location='cpu')
        model = Denoiser(vocab).cuda(); model.load_state_dict(checkpoint['model'])
        if split == 'test':
            recovery = validate(model, data)
            (ROOT / f'runs/{rep}_17/test_recovery.json').write_text(json.dumps(recovery, indent=2))
        for mode in modes:
            seed_all(1023)
            folder = OUT / split / f'{rep}_{mode}'; folder.mkdir(parents=True, exist_ok=True)
            generated, metrics = [], []
            for start in range(0, min(count, len(records)), 32):
                end = min(start+32, count)
                ids, values = variant(model, data[2][start:end], mode)
                for reference, y, v in zip(records[start:end], ids.tolist(), values.tolist()):
                    try: program, error = render(vocab, y, v), None
                    except Exception as exception: program, error = '', str(exception)
                    record = {'id': reference['id'], 'representation': rep, 'seed': 17,
                              'program': program, 'render_error': error}
                    generated.append(record)
                    metrics.append(score_program(record, reference, folder))
            (folder / 'generated.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generated))
            (folder / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics))
            keys = METRICS + ['basic_and_symmetry_consistent']
            summary = {'split': split, 'representation': rep, 'mode': mode, 'n': len(metrics),
                       'checkpoint_epoch': checkpoint['epoch'],
                       'raw': {k: sum(r['raw'].get(k, False) for r in metrics) for k in keys},
                       'projected': {k: sum(r['projected'].get(k, False) for r in metrics) for k in keys},
                       'generated_space_groups': dict(Counter(str(r['projected'].get('generated_sg')) for r in metrics if r['projected'].get('basic_and_symmetry_consistent'))),
                       'raw_errors': Counter(r['raw'].get('error', '') for r in metrics if 'error' in r['raw']).most_common(8)}
            (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        del model, data
        torch.cuda.empty_cache()
    (OUT / f'{split}_summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4)
    for rep in ['explicit', 'wyckoff']:
        assert json.loads((ROOT / f'runs/{rep}_17/training_result.json').read_text())['state'] == 'complete'
    wp.reference_table()
    validation = run_split('val', ['one_step', 'confidence', 'remask'])
    ranking = []
    for mode in ['one_step', 'confidence', 'remask']:
        rows = [r for r in validation if r['mode'] == mode]
        ranking.append({'mode': mode,
                        'projected_valid': sum(r['projected']['basic_and_symmetry_consistent'] for r in rows),
                        'raw_usable': sum(r['raw']['usable'] for r in rows),
                        'raw_decoded': sum(r['raw']['decoded'] for r in rows)})
    ranking.sort(key=lambda r: (r['projected_valid'], r['raw_usable'], r['raw_decoded'], r['mode']), reverse=True)
    selection = {'mode': ranking[0]['mode'], 'ranking': ranking,
                 'selection': 'same sampler for both representations, selected on first 512 hash-ordered validation structures',
                 'checkpoint_selection': 'lowest full-validation fixed-mask reconstruction loss within the fixed epoch budget',
                 'checkpoint_sha256': {rep: hashlib.sha256((ROOT / f'runs/{rep}_17/best.pt').read_bytes()).hexdigest() for rep in ['explicit', 'wyckoff']}}
    assert not (OUT / 'selection.json').exists(), 'Test sampler already frozen'
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps({'selection': selection}), flush=True)
    run_split('test', [selection['mode']])


if __name__ == '__main__': main()
