"""NatureCry v0.1: controlled English over the existing reversible Wyckoff codec.

Only structural data are encoded. Grammar variants and free-form paraphrases
are outside this version; the fixed public Wyckoff table is the decoder dependency.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter
import spglib

BACKEND = Path(__file__).resolve().parents[1] / 'explicit_full_mp20_20260923/codec'
sys.path.insert(0, str(BACKEND))
import wyckoff_baseline as wyckoff
from test_program import parse


def encode_cif(path, symprec=1e-3):
    source = parse(path)
    if not source.is_ordered:
        raise ValueError('NatureCry v0.1 requires ordered, fully occupied sites')
    full, _, audit = wyckoff.encode_cif(path, symprec=symprec)
    fields = full.splitlines()
    cell = fields[3].split()[1:]
    frame = next((x.split()[1:] for x in fields if x.startswith('FRAME ')), None)
    digits = next(x.split()[1] for x in fields if x.startswith('ROUND '))
    sg, hall = audit['space_group'], audit['hall_number']
    group = spglib.get_spacegroup_type(hall)
    text = [
        'NatureCry version 0.1.',
        f'The input unit cell contains {len(source)} fully occupied atoms with composition {source.composition.formula}.',
        f'The symmetry model uses space group {group.international_short} (number {sg}), Hall number {hall}, and Hall symbol "{group.hall_symbol}".',
        f'The input lattice has lengths a = {cell[0]} angstrom, b = {cell[1]} angstrom, c = {cell[2]} angstrom, and angles alpha = {cell[3]} degrees, beta = {cell[4]} degrees, gamma = {cell[5]} degrees.',
        'All atomic positions use fractional coordinates and are periodic modulo one.',
    ]
    if frame:
        text.append('The standard fractional frame satisfies q_standard = P q_input + s.')
        for i, name in enumerate(['first', 'second', 'third']):
            text.append(f'The {name} row of P is ({", ".join(frame[3*i:3*i+3])}).')
        text.append(f'The origin shift s is ({", ".join(frame[9:])}).')
    else:
        text.append('The input and standard fractional coordinate frames are identical.')
    text.extend([
        f'Round input-frame fractional coordinates to {digits} decimal places before applying precision corrections.',
        f'The structure is described by {len(audit["nodes"])} independent Wyckoff orbits.',
    ])
    for i, node in enumerate(audit['nodes'], 1):
        parameters = ', '.join(f'{axis} = {value}' for axis, value in zip(node['free_axes'], node['parameters']))
        detail = f'The independent parameters are {parameters}.' if parameters else 'It has no free positional parameters.'
        text.append(f'The orbit O{i} is occupied by {node["element"]} with occupancy 1. Its Wyckoff position is {node["label"]}, with multiplicity {int(node["label"][:-1])}. Its representative fractional coordinates follow ({node["template"]}) in the standard frame. {detail}')
    residual_index = next(i for i, x in enumerate(fields) if x.startswith('RESIDUALS '))
    count = int(fields[residual_index].split()[1])
    text.append(f'We retain {count} precision corrections in the input frame.')
    for row in fields[residual_index + 1:-1]:
        index, *delta = row.split()
        text.append(f'The generated atom with zero-based index {index} receives the fractional-coordinate correction ({", ".join(delta)}).')
    text.extend([
        'Generate each orbit with the declared Hall setting, transform into the input cell, round coordinates, and apply precision corrections.',
        'End of NatureCry description.',
    ])
    audit.update(input_atoms=len(source), input_formula=source.composition.formula)
    return '\n'.join(text) + '\n', audit


def decode(text):
    lines = text.splitlines()
    cursor = 0

    def take(pattern):
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError('Incomplete NatureCry description')
        match = re.fullmatch(pattern, lines[cursor])
        if not match:
            raise ValueError(f'Unexpected or conflicting sentence {cursor + 1}: {lines[cursor]}')
        cursor += 1
        return match.groups()

    def vector(value):
        return [float(x.strip()) for x in value.split(',')]

    take(r'NatureCry version 0\.1\.')
    atoms, formula = take(r'The input unit cell contains (\d+) fully occupied atoms with composition (.+)\.')
    symbol, sg, hall, hall_symbol = take(r'The symmetry model uses space group (\S+) \(number (\d+)\), Hall number (\d+), and Hall symbol "(.+)"\.')
    sg, hall = int(sg), int(hall)
    table = wyckoff.reference_table()[sg]
    group = spglib.get_spacegroup_type(hall)
    if hall != table['hall'] or symbol != group.international_short or hall_symbol != group.hall_symbol:
        raise ValueError('Inconsistent space-group/Hall setting')
    cell = take(r'The input lattice has lengths a = (\S+) angstrom, b = (\S+) angstrom, c = (\S+) angstrom, and angles alpha = (\S+) degrees, beta = (\S+) degrees, gamma = (\S+) degrees\.')
    values = list(map(float, cell))
    if not all(np.isfinite(values)) or min(values[:3]) <= 0 or not all(0 < a < 180 for a in values[3:]):
        raise ValueError('Invalid lattice parameters')
    program = ['WYCKOFF1', f'SG {sg}', f'HALL {hall}', 'CELL ' + ' '.join(cell)]
    take(r'All atomic positions use fractional coordinates and are periodic modulo one\.')
    if lines[cursor] == 'The input and standard fractional coordinate frames are identical.':
        take(re.escape(lines[cursor]))
    else:
        take(r'The standard fractional frame satisfies q_standard = P q_input \+ s\.')
        matrix = []
        for name in ['first', 'second', 'third']:
            row, = take(f'The {name} row of P is \\((.+)\\)\\.')
            row = vector(row)
            if len(row) != 3:
                raise ValueError('Frame rows must have three components')
            matrix.extend(row)
        shift, = take(r'The origin shift s is \((.+)\)\.')
        shift = vector(shift)
        if len(shift) != 3 or not np.isfinite(matrix + shift).all() or abs(np.linalg.det(np.array(matrix).reshape(3, 3))) < 1e-12:
            raise ValueError('Invalid fractional-frame transform')
        program.append('FRAME ' + ' '.join(map(str, matrix + shift)))
    digits, = take(r'Round input-frame fractional coordinates to (\d+) decimal places before applying precision corrections\.')
    count, = take(r'The structure is described by (\d+) independent Wyckoff orbits\.')
    program.extend([f'ROUND {digits}', f'SITES {count}'])
    for i in range(1, int(count) + 1):
        element, label, multiplicity, template, detail = take(rf'The orbit O{i} is occupied by (\S+) with occupancy 1\. Its Wyckoff position is (\S+), with multiplicity (\d+)\. Its representative fractional coordinates follow \((.+)\) in the standard frame\. (.+)')
        definition = table['definitions'][label]
        if int(multiplicity) != definition['multiplicity'] or template != definition['template']:
            raise ValueError('Wyckoff label, multiplicity, and coordinate constraints disagree')
        axes = ['xyz'[a] for a in definition['axes']]
        parameters = []
        if axes:
            match = re.fullmatch(r'The independent parameters are (.+)\.', detail)
            if not match:
                raise ValueError('Missing independent positional parameters')
            assignments = [x.split(' = ') for x in match[1].split(', ')]
            if [x[0] for x in assignments] != axes:
                raise ValueError('Wrong positional parameter names')
            parameters = [float(x[1]) for x in assignments]
            if not np.isfinite(parameters).all():
                raise ValueError('Nonfinite positional parameters')
        elif detail != 'It has no free positional parameters.':
            raise ValueError('A fixed position cannot declare free parameters')
        program.append(' '.join([element, label, *map(str, parameters)]))
    corrections, = take(r'We retain (\d+) precision corrections in the input frame\.')
    program.append(f'RESIDUALS {corrections}')
    corrected = set()
    for _ in range(int(corrections)):
        index, delta = take(r'The generated atom with zero-based index (\d+) receives the fractional-coordinate correction \((.+)\)\.')
        delta = vector(delta)
        if int(index) >= int(atoms) or index in corrected or len(delta) != 3 or not np.isfinite(delta).all():
            raise ValueError('Invalid precision-correction record')
        corrected.add(index)
        program.append(' '.join([index, *map(str, delta)]))
    take(r'Generate each orbit with the declared Hall setting, transform into the input cell, round coordinates, and apply precision corrections\.')
    take(r'End of NatureCry description\.')
    if cursor != len(lines):
        raise ValueError('Trailing sentences are outside NatureCry v0.1')
    structure = wyckoff.decode('\n'.join(program + ['END']))
    if len(structure) != int(atoms) or structure.composition != Composition(formula):
        raise ValueError('Declared atom count or composition disagrees with the decoded structure')
    return structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['encode', 'decode', 'decode-directory'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'encode':
        args.output.write_text(encode_cif(args.input)[0])
    elif args.command == 'decode':
        CifWriter(decode(args.input.read_text()), symprec=None, significant_figures=14).write_file(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        failures = {}
        for path in sorted(args.input.glob('*.txt')):
            try:
                CifWriter(decode(path.read_text()), symprec=None, significant_figures=14).write_file(args.output / (path.stem + '.cif'))
            except Exception as error:
                failures[path.stem] = repr(error)
        (args.output / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
        print(json.dumps({'texts_received': len(list(args.input.glob('*.txt'))), 'decode_failures': failures}))


if __name__ == '__main__':
    main()
