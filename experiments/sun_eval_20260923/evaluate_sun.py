"""Bounded CHGNet SUN proxy evaluation of existing paired generations."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import time
import warnings

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'explicit_full_mp20_20260923'
MATCHER = StructureMatcher(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True,
                           scale=True, attempt_supercell=False)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def key(s):
    return s.composition.reduced_formula


def interrupted(*args):
    raise TimeoutError('120 second candidate limit')


def match_any(s, candidates):
    for ident, ref in candidates:
        if MATCHER.fit(s, ref):
            return ident
    return None


def unique_indices(rows):
    seen = collections.defaultdict(list)
    indices = []
    for i, r in enumerate(rows):
        s = Structure.from_dict(r['structure'])
        if match_any(s, seen[key(s)]) is None:
            seen[key(s)].append((r['id'], s))
            indices.append(i)
    return indices


def novelty(rows, refs):
    for r in rows:
        s = Structure.from_dict(r['structure'])
        r['train_match'] = match_any(s, refs['train'][key(s)])
        r['val_match'] = match_any(s, refs['val'][key(s)])
    return rows


def summary(rows, converged_only=False):
    successful = [r for r in rows if 'structure' in r and 'error' not in r]
    result = {'total_attempts': 512, 'evaluated': len(successful),
              'unique': len(unique_indices(successful)),
              'novel_train': sum(r['train_match'] is None for r in successful),
              'novel_train_val': sum(r['train_match'] is None and r['val_match'] is None for r in successful)}
    if not successful or 'ehull' not in successful[0]:
        return result
    result['converged'] = sum(r['converged'] for r in successful)
    result['thresholds'] = {}
    for threshold in [0., .08, .1]:
        eligible = [r for r in successful if r['ehull'] < threshold and r['post_basic']
                    and (r['converged'] or not converged_only)]
        novel = [r for r in eligible if r['train_match'] is None]
        count = len(unique_indices(novel))
        result['thresholds'][str(threshold)] = {
            'stable': len(eligible), 'stable_unique_novel': count,
            'sun_percent_of_all_attempts': 100 * count / 512,
            'sun_train_val': len(unique_indices([r for r in novel if r['val_match'] is None]))}
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a compute allocation'
    warnings.filterwarnings('ignore', category=UserWarning)
    start = time.time()
    inputs = {}
    for rep in ['explicit', 'wyckoff']:
        base = SOURCE / 'evaluation/test' / f'{rep}_remask'
        metrics = [json.loads(line) for line in (base / 'metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == 512
        rows = []
        for m in sorted(metrics, key=lambda x: x['id']):
            if not m['projected'].get('basic_and_symmetry_consistent'):
                continue
            path = base / f"projected/runs/{rep}_17/samples/{m['id']}.cif"
            s = Structure.from_file(path)
            rows.append({'id': m['id'], 'structure': s.as_dict(), 'formula': key(s),
                         'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'initial_sg': SpacegroupAnalyzer(s, symprec=.001).get_space_group_number()})
        assert len(rows) == {'explicit': 119, 'wyckoff': 145}[rep]
        inputs[rep] = rows
    keys = {r['formula'] for rows in inputs.values() for r in rows}
    refs = {split: collections.defaultdict(list) for split in ['train', 'val']}
    ref_counts = collections.Counter()
    with (SOURCE / 'data/records.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['split'] not in refs:
                continue
            ref_counts[r['split']] += 1
            d = r['reference']
            s = Structure(d['lattice'], d['species'], d['positions'])
            if key(s) in keys:
                refs[r['split']][key(s)].append((r['id'], s))
    assert ref_counts == {'train': 27136, 'val': 9047}
    example = Structure.from_dict(inputs['explicit'][0]['structure'])
    assert MATCHER.fit(example, example.copy())
    assert len(unique_indices([inputs['explicit'][0]] * 2)) == 1
    for rep, rows in inputs.items():
        novelty(rows, refs)
        dump(ROOT / f'{rep}_unrelaxed.json', rows)
    dump(ROOT / 'unrelaxed_summary.json', {rep: summary(rows) for rep, rows in inputs.items()})
    print('UNRELAXED_DONE', flush=True)
    hull_path = ROOT / 'assets/2023-02-07-ppd-mp.pkl.gz'
    assert hashlib.md5(hull_path.read_bytes()).hexdigest() == '60d19d691fa1d338aa496a40a9641bef'
    with gzip.open(hull_path, 'rb') as f:
        hull = pickle.load(f)
    import torch
    from chgnet.model import CHGNet, StructOptimizer
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    model = CHGNet.load(model_name='0.3.0', use_device='cuda')
    optimizer = StructOptimizer(model=model, use_device='cuda', optimizer_class='FIRE')
    signal.signal(signal.SIGALRM, interrupted)
    outputs = {rep: [] for rep in inputs}
    dump(ROOT / 'runtime.json', {'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
                                'gpu': torch.cuda.get_device_name(), 'reference_counts': ref_counts})
    # Interleave representations to avoid spending all the budget on one arm.
    for i in range(max(map(len, inputs.values()))):
        for rep, rows in inputs.items():
            if i >= len(rows):
                continue
            r = rows[i]
            result = {k: v for k, v in r.items() if k not in ['structure', 'train_match', 'val_match']}
            began = time.time()
            try:
                signal.alarm(120)
                s = Structure.from_dict(r['structure'])
                hull_energy = float(hull.get_hull_energy_per_atom(s.composition))
                relaxed = optimizer.relax(s, fmax=.1, steps=500, relax_cell=True, verbose=False)
                final = relaxed['final_structure']
                traj = relaxed['trajectory']
                forces = np.linalg.norm(traj.forces[-1], axis=1)
                energy = float(traj.energies[-1]) / len(final)
                assert np.isfinite(energy)
                post_basic = bool(np.isfinite(final.volume) and final.volume > 0
                                  and key(final) == r['formula']
                                  and len(final.get_neighbor_list(.5, exclude_self=True)[0]) == 0)
                result.update(structure=final.as_dict(), energy_per_atom=energy,
                              initial_energy_per_atom=float(traj.energies[0]) / len(s),
                              hull_energy_per_atom=hull_energy, ehull=energy-hull_energy,
                              initial_ehull=float(traj.energies[0]) / len(s)-hull_energy,
                              steps=len(traj.energies)-2, trajectory_frames=len(traj.energies),
                              max_atomic_force=float(forces.max()),
                              converged=bool(len(traj.energies)-2 < 500 and forces.max() <= .1),
                              post_basic=post_basic,
                              final_sg=SpacegroupAnalyzer(final, symprec=.001).get_space_group_number())
                novelty([result], refs)
                target = ROOT / 'relaxed' / rep
                target.mkdir(parents=True, exist_ok=True)
                CifWriter(final, symprec=None).write_file(target / f"{r['id']}.cif")
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
            finally:
                signal.alarm(0)
            result['seconds'] = time.time()-began
            outputs[rep].append(result)
            with (ROOT / f'{rep}_relaxed.jsonl').open('a') as f:
                f.write(json.dumps(result, allow_nan=False)+'\n')
            dump(ROOT / 'status.json', {'state': 'running', 'done': {k: len(v) for k,v in outputs.items()},
                                       'last': {'rep': rep, 'id': r['id'], 'seconds': result['seconds'],
                                                'error': result.get('error')}, 'elapsed_seconds': time.time()-start})
            print(rep, i+1, result.get('ehull', result.get('error')), round(result['seconds'], 2), flush=True)
    report = {'unrelaxed': {rep: summary(rows) for rep, rows in inputs.items()},
              'relaxed_converged': {rep: summary(rows, True) for rep, rows in outputs.items()},
              'relaxed_endpoints': {rep: summary(rows, False) for rep, rows in outputs.items()},
              'errors': {rep: [r for r in rows if 'error' in r] for rep, rows in outputs.items()},
              'elapsed_seconds': time.time()-start}
    dump(ROOT / 'RESULTS.json', report)
    dump(ROOT / 'status.json', {'state': 'complete', 'elapsed_seconds': time.time()-start})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
