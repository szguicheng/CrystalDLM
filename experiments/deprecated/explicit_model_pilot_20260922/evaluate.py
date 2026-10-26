"""One-shot held-out scoring after all four trainings, no target-guided repair."""
import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import warnings

import numpy as np
import torch
from model import ROOT, Denoiser, load_data, render, sample, seed_all, validate


def generate():
    assert all((ROOT / 'runs' / f'{r}_{s}' / 'training_result.json').exists()
               for r in ['explicit', 'wyckoff'] for s in [17, 29]), 'Finish all training before inspecting test'
    torch.set_num_threads(4)
    for representation in ['explicit', 'wyckoff']:
        data, records, vocab = load_data('test', representation)
        for seed in [17, 29]:
            folder = ROOT / 'runs' / f'{representation}_{seed}'
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu')
            model = Denoiser(vocab).cuda(); model.load_state_dict(checkpoint['model'])
            recovery = validate(model, data)
            (folder / 'test_recovery.json').write_text(json.dumps(recovery, indent=2))
            began = time.time()
            with (folder / 'generated.jsonl').open('w') as file:
                for start in range(0, len(records), 32):
                    ids, values = sample(model, data[2][start:start+32], steps=32)
                    for r, y, v in zip(records[start:start+32], ids.cpu().tolist(), values.cpu().tolist()):
                        try: program, error = render(vocab, y, v), None
                        except Exception as exception: program, error = '', str(exception)
                        file.write(json.dumps({'id': r['id'], 'representation': representation, 'seed': seed,
                                               'program': program, 'render_error': error}) + '\n')
                    file.flush()
            timing = {'seconds': time.time() - began, 'count': len(records), 'steps': 32,
                      'sampling': 'greedy confidence reveal, cosine remaining-mask schedule, no target length',
                      'checkpoint_step': checkpoint['step']}
            (folder / 'sampling.json').write_text(json.dumps(timing, indent=2))
            print(json.dumps({'generated': folder.name, **timing}), flush=True)


def time_limit(signum, frame):
    raise TimeoutError('10 second per-sample decode and structure check budget')


def reduced_counts(structure):
    counts = np.bincount(structure.atomic_numbers, minlength=119)[1:119]
    divisor = math.gcd(*map(int, counts[counts > 0]))
    return (counts // divisor).tolist()


def score_one(item):
    record, reference = item
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import spglib
    import program_codec as sp
    import wyckoff_baseline as wp
    result = {key: record[key] for key in ['id', 'representation', 'seed']}
    result.update(decoded=False, positive_volume=False, no_overlap=False, composition_match=False,
                  usable=False, reference_match=False, reference_sg_match=False)
    signal.signal(signal.SIGALRM, time_limit); signal.alarm(10)
    stage = 'decode'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if record['render_error']: raise ValueError(record['render_error'])
            structure = (sp if record['representation'] == 'explicit' else wp).decode(record['program'])
            result['decoded'] = True
            result['atoms'] = len(structure)
            result['formula'] = structure.composition.reduced_formula
            stage = 'physical_checks'
            result['positive_volume'] = bool(np.isfinite(structure.volume) and structure.volume > 0
                                             and np.isfinite(structure.frac_coords).all())
            result['composition_match'] = reduced_counts(structure) == reference['composition']
            # Include periodic copies of the same atom; a pairwise N x N matrix alone misses them.
            centers, points, images, distances = structure.get_neighbor_list(0.5, exclude_self=True)
            result['no_overlap'] = len(distances) == 0
            result['usable'] = result['positive_volume'] and result['no_overlap'] and result['composition_match']
            stage = 'symmetry'
            dataset = spglib.get_symmetry_dataset((structure.lattice.matrix, structure.frac_coords,
                                                  structure.atomic_numbers), symprec=1e-3, angle_tolerance=-1)
            result['generated_sg'] = int(dataset.number) if dataset else None
            result['reference_sg_match'] = result['generated_sg'] == reference['sg']
            if record['representation'] == 'wyckoff':
                tokens = record['program'].split()
                result['declared_sg'] = int(tokens[tokens.index('SG') + 1])
                result['declared_sg_match'] = result['generated_sg'] == result['declared_sg']
            stage = 'reference_match'
            target = reference['reference']
            target = Structure(target['lattice'], target['species'], target['positions'])
            if result['composition_match']:
                matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, scale=False,
                                           primitive_cell=True, attempt_supercell=False)
                result['reference_match'] = bool(matcher.fit(target, structure))
            stage = 'write_cif'
            folder = ROOT / 'runs' / f"{record['representation']}_{record['seed']}" / 'samples'
            folder.mkdir(exist_ok=True)
            CifWriter(structure, symprec=None, significant_figures=12).write_file(folder / (record['id'] + '.cif'))
    except Exception as error:
        result['error_stage'] = stage
        result['error'] = f'{type(error).__name__}: {error}'
    finally: signal.alarm(0)
    return result


def score():
    sys.path.insert(0, str(ROOT / 'codec'))
    import wyckoff_baseline as wp
    wp.reference_table()
    references = {}
    with (ROOT / 'data/records.jsonl').open() as file:
        for line in file:
            r = json.loads(line)
            if r['split'] == 'test' and r['status'] == 'ok': references[r['id']] = r
    summaries = {}
    with mp.get_context('fork').Pool(4) as pool:
        for representation in ['explicit', 'wyckoff']:
            for seed in [17, 29]:
                name = f'{representation}_{seed}'; folder = ROOT / 'runs' / name
                generated = [json.loads(line) for line in (folder / 'generated.jsonl').read_text().splitlines()]
                results = list(pool.imap(score_one, [(r, references[r['id']]) for r in generated]))
                (folder / 'structure_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
                metrics = ['decoded', 'positive_volume', 'no_overlap', 'composition_match', 'usable',
                           'reference_match', 'reference_sg_match']
                summary = {'n': len(results), 'counts': {key: sum(r[key] for r in results) for key in metrics},
                           'rates': {key: sum(r[key] for r in results) / len(results) for key in metrics},
                           'space_group_counts': dict(Counter(str(r.get('generated_sg')) for r in results if r['decoded'])),
                           'top_errors': Counter(r.get('error', '').split(':')[0] + ':' + r.get('error_stage', '')
                                                 for r in results if 'error' in r).most_common(12)}
                summary['recovery'] = json.loads((folder / 'test_recovery.json').read_text())
                summary['training'] = json.loads((folder / 'training_result.json').read_text())
                summaries[name] = summary
                (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
                print(json.dumps({'scored': name, 'n': summary['n'], 'counts': summary['counts']}), flush=True)
    (ROOT / 'results.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['generate', 'score'])
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if args.mode == 'generate': generate()
    else: score()
