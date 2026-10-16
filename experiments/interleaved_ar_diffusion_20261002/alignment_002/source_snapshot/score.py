"""Audit the unmodified PLaID representation and score verifiable tasks."""
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np
from pymatgen.core import Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'work/PLaID'))
from utils import parse_fn_wyckoff


@lru_cache(None)
def group(symbol):
    return Group(symbol)


def parse(text):
    lines = text.strip().splitlines()
    if len(lines) < 6:
        raise ValueError('missing header or orbit records')
    if not re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', lines[0]):
        raise ValueError('invalid cell formula')
    if not lines[1].startswith('Spacegroup: '):
        raise ValueError('invalid spacegroup header')
    g = group(lines[1].split(': ', 1)[1])
    if not lines[2].startswith('abc: ') or not lines[3].startswith('angles: '):
        raise ValueError('invalid lattice header')
    lengths = [float(x) for x in lines[2].split()[1:]]
    angles = [float(x) for x in lines[3].split()[1:]]
    if len(lengths) != 3 or len(angles) != 3 or not np.isfinite(lengths + angles).all():
        raise ValueError('invalid lattice values')
    match = re.fullmatch(r'Sites \((\d+)\)', lines[4])
    if not match:
        raise ValueError('invalid Sites header')
    sites = []
    for line in lines[5:]:
        element, x, y, z, label = line.split()
        if not Element.is_valid_symbol(element) or not re.fullmatch(r'\d+[a-z]', label):
            raise ValueError('invalid orbit species or label')
        coords = [float(x), float(y), float(z)]
        if not np.isfinite(coords).all():
            raise ValueError('invalid fractional coordinate')
        wp = g.get_wyckoff_position(label)
        if wp.get_label() != label:
            raise ValueError('Wyckoff multiplicity does not match label')
        sites.append(dict(element=element, coords=coords, label=label))
    counts = Counter()
    for site in sites:
        counts[site['element']] += int(site['label'][:-1])
    return dict(formula=lines[0], counts=dict(counts), sg=g.number,
                lengths=lengths, angles=angles, atoms=sum(counts.values()),
                declared_atoms=int(match[1]), orbits=len(sites), sites=sites)


def audit(text, condition=None):
    reasons = []
    result = dict(valid=False, reasons=reasons)
    try:
        p = parse(text)
        result['facts'] = p
        if Composition(p['formula']).get_el_amt_dict() != p['counts']:
            reasons.append('formula_header_mismatch')
        if p['declared_atoms'] != p['atoms']:
            reasons.append('sites_header_mismatch')
        if condition:
            if Composition(condition['formula']).get_el_amt_dict() != p['counts']:
                reasons.append('composition_condition_mismatch')
            if int(condition['sg']) != p['sg']:
                reasons.append('space_group_condition_mismatch')
        if min(p['lengths']) <= 0 or not all(0 < v < 180 for v in p['angles']):
            reasons.append('invalid_lattice')
        a, b, c = p['lengths']
        alpha, beta, gamma = p['angles']
        kind = group(p['sg']).lattice_type
        close = lambda x, y, eps: abs(x-y) <= eps + 1e-10
        if kind in ('orthorhombic', 'tetragonal', 'cubic'):
            if not all(close(v, 90, .01) for v in p['angles']):
                reasons.append('lattice_symmetry_mismatch')
        if kind in ('hexagonal', 'trigonal'):
            if not (close(a, b, .01) and close(alpha, 90, .01) and
                    close(beta, 90, .01) and close(gamma, 120, .01)):
                reasons.append('lattice_symmetry_mismatch')
        if kind in ('tetragonal', 'cubic') and not close(a, b, .01):
            reasons.append('lattice_symmetry_mismatch')
        if kind == 'cubic' and not close(a, c, .01):
            reasons.append('lattice_symmetry_mismatch')
        if kind == 'monoclinic' and not (close(alpha, 90, .01) and close(gamma, 90, .01)):
            reasons.append('lattice_symmetry_mismatch')
        errors = []
        fixed_labels = []
        for site in p['sites']:
            wp = group(p['sg']).get_wyckoff_position(site['label'])
            delta = wp.ops[0].operate(site['coords']) - site['coords']
            errors.append(float(np.abs(delta - np.round(delta)).max()))
            if wp.get_dof() == 0:
                fixed_labels.append(site['label'])
        result['max_fractional_constraint_error'] = max(errors)
        if max(errors) > 1e-3 + 1e-10:
            reasons.append('wyckoff_coordinate_constraint')
        if len(set(fixed_labels)) != len(fixed_labels):
            reasons.append('duplicate_fixed_orbit')
        if reasons:
            return result
        structure = parse_fn_wyckoff(text, to_cif=False)
        result['decoded_atoms'] = len(structure)
        if structure.volume <= 0 or not np.isfinite(structure.lattice.matrix).all():
            reasons.append('invalid_decoded_lattice')
        if structure.composition.get_el_amt_dict() != p['counts']:
            reasons.append('decoded_composition_mismatch')
        neighbors = structure.get_all_neighbors(.5)
        overlaps = [float(n.nn_distance) for row in neighbors for n in row if n.nn_distance < .5-1e-10]
        if overlaps:
            reasons.append('atomic_overlap')
            result['minimum_overlap_distance_A'] = min(overlaps)
        detected = SpacegroupAnalyzer(structure, symprec=.01, angle_tolerance=5).get_space_group_number()
        result['detected_sg'] = int(detected)
        if detected != p['sg']:
            reasons.append('decoded_space_group_mismatch')
        shifts = []
        for site in p['sites']:
            distances = [structure.lattice.get_distance_and_image(site['coords'], atom.frac_coords)[0]
                         for atom in structure if atom.specie.symbol == site['element']]
            shifts.append(min(distances))
        result['decoder_max_representative_shift_A'] = float(max(shifts))
        result['valid'] = not reasons
    except Exception as error:
        # A failed candidate stays in the evaluation denominator.
        reasons.append(f'{type(error).__name__}: {error}')
    return result


def edit(text, scale):
    parse(text)
    lines = text.strip().splitlines()
    values = [Decimal(v) * Decimal(str(scale)) for v in lines[2].split()[1:]]
    lines[2] = 'abc: ' + ' '.join(str(v.quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)) for v in values)
    return '\n'.join(lines)


def answer(text):
    p = parse(text)
    a, b, c = p['lengths']
    return (f"The space group is {p['sg']}. The composition is {p['formula']}. "
            f"The cell contains {p['atoms']} atoms in {p['orbits']} independent orbits. "
            f'The lattice lengths are a={a:.2f}, b={b:.2f}, c={c:.2f} angstroms.')


def grade_answer(response, text):
    p = parse(text)
    patterns = {
        'sg': r'(?:space\s*group)\s*(?:number\s*)?(?:is\s*|[:=]\s*)?(\d+)',
        'formula': r'(?:composition|formula)\s*(?:is\s*|[:=]\s*)?((?:[A-Z][a-z]?\d*)+)',
        'atoms': r'(\d+)\s+atoms\b',
        'orbits': r'(\d+)\s+(?:independent\s+)?orbits\b',
    }
    fields = {}
    for name, pattern in patterns.items():
        values = re.findall(pattern, response, re.I)
        if name == 'formula':
            try:
                fields[name] = bool(values) and all(Composition(v).get_el_amt_dict() == p['counts'] for v in values)
            except ValueError:
                fields[name] = False
        else:
            fields[name] = bool(values) and all(int(v) == p[name] for v in values)
    triple = re.findall(r'lattice\s+lengths\s*(?:are\s*|[:=]\s*)([\d.]+)\s*,\s*([\d.]+)\s*,?\s*(?:and\s*)?([\d.]+)', response, re.I)
    for i, name in enumerate('abc'):
        values = re.findall(rf'\b{name}\s*[:=]\s*(-?\d+(?:\.\d+)?)', response, re.I)
        values += [row[i] for row in triple]
        fields[name] = bool(values) and all(abs(float(v)-p['lengths'][i]) <= .005 + 1e-10 for v in values)
    return dict(success=all(fields.values()), fields=fields)


def grade_edit(output, source, scale):
    try:
        observed, target = parse(output), parse(edit(source, scale))
        fields = {key: observed[key] == target[key]
                  for key in ['counts', 'sg', 'lengths', 'angles', 'sites', 'declared_atoms']}
        checked = audit(output)
        return dict(success=all(fields.values()) and checked['valid'], fields=fields, audit=checked)
    except Exception as error:
        return dict(success=False, error=f'{type(error).__name__}: {error}')
