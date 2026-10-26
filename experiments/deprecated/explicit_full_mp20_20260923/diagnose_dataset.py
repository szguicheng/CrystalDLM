"""Minimal reproduction for the observed Ho5Pt3 preprocessing failure."""
import csv
import json
import os
from pathlib import Path
import socket
import tempfile

import numpy as np
import spglib
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from prepare import INPUT, parse, wp, sp, compare, process

assert os.environ.get('SLURM_JOB_ID')
with (INPUT / 'train.csv').open() as file:
    row = next(r for r in csv.DictReader(file) if r['material_id'] == 'mp-1189141')
wp.reference_table()
results = []
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'case.cif'; path.write_text(row['cif'])
    raw = parse(path)
    cell = (raw.lattice.matrix, raw.frac_coords, raw.atomic_numbers)
    original = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    table = wp.reference_table()[int(original.number)]
    ds = spglib.get_symmetry_dataset(cell, hall_number=table['hall'], symprec=1e-3)
    for digits in [6, 8, 12, None]:
        points = ds.std_positions if digits is None else np.round(ds.std_positions, digits)
        structure = Structure(ds.std_lattice, ds.std_types, points % 1)
        CifWriter(structure, symprec=None, significant_figures=14).write_file(path)
        try:
            _, program, audit = wp.encode_cif(path)
            canonical = wp.decode(program)
            CifWriter(canonical, symprec=None, significant_figures=14).write_file(path)
            explicit, meta = sp.encode_cif(path)
            result = {'digits': digits, 'passed': True, 'wp_error_A': compare(parse(path), canonical),
                      'sp_error_A': compare(parse(path), sp.decode(explicit)), 'sg': meta['space_group']}
        except Exception as error:
            result = {'digits': digits, 'passed': False, 'error': f'{type(error).__name__}: {error}'}
        results.append(result)
    parsed = parse(path)
    cell2 = (parsed.lattice.matrix, parsed.frac_coords, parsed.atomic_numbers)
    detected2 = spglib.get_symmetry_dataset(cell2, symprec=1e-3)
    table2 = wp.reference_table()[int(detected2.number)]
    ds2 = spglib.get_symmetry_dataset(cell2, hall_number=table2['hall'], symprec=1e-3)
    diagnostics = []
    for rep in range(len(ds2.std_positions)):
        z = int(ds2.std_types[rep])
        images = (np.einsum('nij,j->ni', table2['rotations'], ds2.std_positions[rep]) + table2['translations']) % 1
        delta = images[:, None, :] - ds2.std_positions[None, :, :]; delta -= np.rint(delta)
        members = np.flatnonzero(np.any(np.max(np.abs(delta), axis=-1) < 1e-7, axis=0) & (ds2.std_types == z))
        if rep != min(members): continue
        points = ds2.std_positions[members]
        fits = {}
        for label, definition in table2['definitions'].items():
            if definition['multiplicity'] != len(members): continue
            matrix = definition['a'][:, definition['axes']]
            targets = (points[:, None, :] + wp.IMAGES[None, :, :] - definition['b']).reshape(-1, 3)
            values = targets @ np.linalg.pinv(matrix).T if matrix.shape[1] else np.zeros((len(targets), 0))
            errors = np.max(np.abs(values @ matrix.T - targets), axis=1)
            fits[label] = {'template': definition['template'], 'min_linear_residual': float(errors.min()),
                           'fit_result': wp.fit_parameters(points, definition)}
        diagnostics.append({'Z': z, 'rep': rep, 'members': members.tolist(), 'points': points.tolist(), 'fits': fits})
output = {'id': row['material_id'], 'formula': row['pretty_formula'], 'raw_sg': int(original.number),
          'host': socket.gethostname(), 'results': results, 'canonical_sg': int(detected2.number),
          'hall': table2['hall'], 'orbits': diagnostics}
original_wp, original_sp = wp.encode_cif, sp.encode_cif
trials = []
for tolerance in [1e-3, 5e-4, 1e-4, 1e-5]:
    wp.encode_cif = lambda path, symprec=1e-3: original_wp(path, symprec=tolerance)
    sp.encode_cif = lambda path, symprec=1e-3: original_sp(path, symprec=tolerance)
    trial = process(('train', row))
    trials.append({'encoder_symprec': tolerance, **{key: trial.get(key) for key in ['status', 'error', 'sg', 'atoms', 'oracle_errors_A']}})
wp.encode_cif, sp.encode_cif = original_wp, original_sp
output['tolerance_trials'] = trials
Path('data/precision_diagnosis.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output), flush=True)
