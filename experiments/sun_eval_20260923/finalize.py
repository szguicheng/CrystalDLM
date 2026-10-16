"""Verify ASE callback counting, fix metadata, and summarize without rerelaxing."""
import json
import os
from pathlib import Path
import runpy
import importlib.metadata
import hashlib

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from evaluate_sun import summary

assert os.environ.get('SLURM_JOB_ID')
root = Path(__file__).resolve().parent
# CHGNet adds obs() once after optimizer.run; ASE also observes step zero.
atoms = Atoms('Cu2', positions=[[0, 0, 0], [2, 0, 0]])
atoms.calc = EMT()
observations = []
optimizer = FIRE(atoms, logfile=None)
optimizer.attach(lambda: observations.append(atoms.get_potential_energy()))
optimizer.run(fmax=1e-20, steps=2)
observations.append(atoms.get_potential_energy())
assert optimizer.nsteps == 2 and len(observations) == optimizer.nsteps + 2
results = json.loads((root / 'RESULTS.json').read_text())
assert not (root / 'metadata_correction.json').exists(), 'Finalize once only'
changes = []
for rep in ['explicit', 'wyckoff']:
    path = root / f'{rep}_relaxed.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({row['id'] for row in rows}) == len(rows)
    source_rows = json.loads((root / f'{rep}_unrelaxed.json').read_text())
    assert {row['id'] for row in rows} == {row['id'] for row in source_rows}
    (root / f'{rep}_relaxed_before_step_correction.jsonl').write_text(path.read_text())
    for row in rows:
        if 'steps' not in row or 'trajectory_frames' in row:
            continue
        original = row['converged']
        row['trajectory_frames'] = row['steps'] + 1
        row['steps'] -= 1
        row['converged'] = row['steps'] < 500 and row['max_atomic_force'] <= .1
        if original != row['converged']:
            changes.append({'representation': rep, 'id': row['id']})
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    results['relaxed_converged'][rep] = summary(rows, True)
    results['relaxed_endpoints'][rep] = summary(rows, False)
    assert len(rows) == {'explicit': 119, 'wyckoff': 145}[rep]
    for variant in ['relaxed_converged', 'relaxed_endpoints']:
        for value in results[variant][rep]['thresholds'].values():
            assert value['stable_unique_novel'] <= value['stable'] <= len(rows) <= 512
(root / 'RESULTS.json').write_text(json.dumps(results, indent=2, allow_nan=False))
(root / 'metadata_correction.json').write_text(json.dumps({
    'reason': 'ASE observes step zero and CHGNet appends a duplicate final observation; nsteps = nframes - 2',
    'runtime_test': {'ase_nsteps': optimizer.nsteps, 'observations': len(observations)},
    'convergence_class_changes': changes, 'energies_and_structures_unchanged': True}, indent=2))
(root / 'verification.json').write_text(json.dumps({
    'all_candidate_ids_accounted_for': True, 'no_duplicate_ids': True,
    'metric_count_invariants': True,
    'versions': {name: importlib.metadata.version(name) for name in
                 ['chgnet', 'torch', 'numpy', 'ase', 'scipy', 'pymatgen', 'spglib']},
    'submitted_evaluator_sha256': hashlib.sha256((root / 'evaluate_sun_submitted.py').read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256((root / 'PROTOCOL.md').read_bytes()).hexdigest()}, indent=2))
runpy.run_path(str(root / 'report.py'), run_name='__main__')
