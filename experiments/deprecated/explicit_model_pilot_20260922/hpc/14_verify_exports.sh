#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python -u - <<'PY'
import hashlib, json, sys
from pathlib import Path
import spglib
from pymatgen.io.cif import CifParser
root = Path.cwd()
sys.path.insert(0, str(root / 'codec'))
import program_codec as sp
import wyckoff_baseline as wp
from test_program import compare
selection = json.loads((root / 'followup/sampler_selection.json').read_text())
for name, expected in selection['checkpoint_sha256'].items():
    assert hashlib.sha256((root / f'runs/{name}/best.pt').read_bytes()).hexdigest() == expected
for name, expected in selection['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
projection = json.loads((root / 'followup/projection_results.json').read_text())
checks = []
for row in projection['results']:
    rep, seed = row['representation'], row['seed']
    folder = root / f'followup/projected/{rep}_{seed}'
    programs = {r['id']: r['program'] for r in map(json.loads, (folder / 'programs.jsonl').read_text().splitlines())}
    for case in row['passed_examples']:
        path = folder / f"runs/{rep}_{seed}/samples/{case['id']}.cif"
        parsed = CifParser(str(path), frac_tolerance=0).parse_structures(primitive=False)[0]
        expected = (sp if rep == 'explicit' else wp).decode(programs[case['id']])
        error = compare(expected, parsed)
        group = spglib.get_symmetry_dataset((parsed.lattice.matrix, parsed.frac_coords, parsed.atomic_numbers), symprec=1e-3)
        assert int(group.number) == case['generated_sg']
        checks.append({'representation': rep, 'seed': seed, 'id': case['id'], 'error_A': error, 'detected_sg': int(group.number)})
result = {'checkpoints_unchanged': True, 'frozen_generation_code_unchanged': True, 'export_checks': checks}
(root / 'followup/export_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
PY
