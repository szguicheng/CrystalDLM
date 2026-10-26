"""Derive intrinsic site point-group types from the pinned affine WP table.

Exact rational/integer stabilizer tests; no generic random coordinates and no
model-performance claim. Standard point-group classification is by spglib.
"""
import ast
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import spglib

sys.path.insert(0, '/scratch/chenggui2/CrystalDLM/gate1_20260908')
from count_vocab import coordinate_matrix, evaluate, exact_rank, read_groups


@lru_cache(None)
def affine(operation):
    matrix = coordinate_matrix(operation)
    offset = []
    for expression in operation.split(','):
        expression = re.sub(r'(\d)([xyz])', r'\1*\2', expression)
        offset.append(evaluate(ast.parse(expression.strip(), mode='eval'), (0, 0, 0)))
    return matrix, offset


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    input_dir = Path('/scratch/chenggui2/CrystalDLM/gate1_20260908/input')
    groups = read_groups(input_dir)
    raw = {}
    with (input_dir / 'wyckoff_list.csv').open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for index, value in reader:
            if index != '0':
                raw[int(index)] = ast.literal_eval(value)
    base = json.loads(Path('set_merge_cap256_audit/vocabulary.json').read_text())['base']
    identity = np.eye(3, dtype=np.int64)
    records, point_groups, group_types = [], {}, {}
    wp_id = 0
    for g, info in groups.items():
        general = [affine(op) for op in raw[g][0]]
        assert general[0][0] == [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
        assert general[0][1] == [0, 0, 0]
        assert all(x.denominator == 1 for matrix, _ in general for row in matrix for x in row)
        rotations = np.array([[[int(x) for x in row] for row in matrix] for matrix, _ in general])
        delta = rotations - identity
        general_order = len(np.unique(rotations, axis=0))
        assert len(general) % general_order == 0
        center = len(general) // general_order
        assert center in [1, 2, 3, 4]
        group_types[g] = {}
        for wp in info['positions']:
            wp_id += 1
            reference = base[str(wp_id)]
            assert (reference['sg'], reference['letter'], reference['multiplicity'], reference['dof']) == (g, wp['letter'], wp['multiplicity'], wp['dof'])
            matrix, offset = affine(wp['representative_map'])
            matrix_scale = math.lcm(*(x.denominator for row in matrix for x in row))
            integer_matrix = np.array([[int(x * matrix_scale) for x in row] for row in matrix])
            denominator = math.lcm(*(x.denominator for _, shift in general for x in shift),
                                   *(x.denominator for x in offset))
            translations = np.array([[int(x * denominator) for x in shift] for _, shift in general])
            integer_offset = np.array([int(x * denominator) for x in offset])
            fixed_linear = ((delta @ integer_matrix) == 0).all(axis=(1, 2))
            fixed_offset = ((delta @ integer_offset + translations) % denominator == 0).all(axis=1)
            stabilizer = rotations[fixed_linear & fixed_offset]
            assert len(np.unique(stabilizer, axis=0)) == len(stabilizer)
            assert len(stabilizer) * wp['multiplicity'] == len(general), (g, wp['letter'])
            difference = stabilizer - identity
            gram = np.einsum('nij,nik->jk', difference, difference)
            freedom = 3 - exact_rank([[Fraction(int(x)) for x in row] for row in gram])
            assert freedom == wp['dof'], (g, wp['letter'], freedom)
            symbol, number, _ = spglib.get_pointgroup(stabilizer)
            number = int(number)
            description = {'symbol': symbol.strip(), 'order': len(stabilizer), 'dof': freedom}
            if number in point_groups:
                assert point_groups[number] == description
            point_groups[number] = description
            assert general_order // len(stabilizer) == wp['multiplicity'] // center
            typed = {'multiplicity': wp['multiplicity'], 'primitive_multiplicity': wp['multiplicity'] // center,
                     'dof': freedom, 'letters': []}
            if number not in group_types[g]:
                group_types[g][number] = typed
            entry = group_types[g][number]
            assert all(entry[k] == typed[k] for k in ['multiplicity', 'primitive_multiplicity', 'dof'])
            entry['letters'].append(wp['letter'])
            records.append({'wp_id': wp_id, 'space_group': g, 'letter': wp['letter'],
                            'multiplicity': wp['multiplicity'], 'centering': center,
                            'primitive_multiplicity': wp['multiplicity'] // center,
                            'dof': freedom, 'point_group_type': number, **description})
    assert len(records) == 1731 and len(point_groups) == 32
    output = Path('site_point_groups')
    output.mkdir(exist_ok=True)
    (output / 'metadata.json').write_text(json.dumps({'positions': records, 'point_groups': point_groups,
                                                     'groups': group_types}, indent=2))
    mixed = []
    for g, types in group_types.items():
        by_mu = {}
        for number, item in types.items():
            by_mu.setdefault(item['primitive_multiplicity'], []).append(number)
        if any(len(v) > 1 for v in by_mu.values()):
            mixed.append(g)
    summary = {'wyckoff_positions': len(records), 'intrinsic_point_group_types': len(point_groups),
               'groups_with_distinct_site_types_at_same_multiplicity': mixed,
               'exact_affine_stabilizer_order_and_dof_checks': 'PASS',
               'model_wp_id_alignment': 'PASS', 'orientation_retained_only_by_WP_identity': True,
               'no_new_crystallographic_concept': True, 'model_utility_not_evaluated': True,
               'job_id': os.environ['SLURM_JOB_ID'], 'run_id': os.environ.get('CRYSTALDLM_RUN_ID'),
               'elapsed_seconds': time.perf_counter() - start,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
