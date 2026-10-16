"""Freeze 100 inputs, run two independent text-only decoders, retain all failures."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from naturecry_codec import encode_cif, parse, wyckoff

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE_SHA = '133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(a, b):
    assert len(a) == len(b), 'atom count differs'
    assert a.composition == b.composition, 'composition differs'
    lattice_error = float(np.max(np.abs(a.lattice.matrix - b.lattice.matrix)))
    assert lattice_error < 1e-8, f'lattice error {lattice_error} A'
    same = np.array(a.atomic_numbers)[:, None] == np.array(b.atomic_numbers)[None, :]
    distances = a.lattice.get_all_distances(a.frac_coords, b.frac_coords)
    i, j = linear_sum_assignment(np.where(same, distances, np.inf))
    residual = a.frac_coords[i] - b.frac_coords[j]
    residual -= np.rint(residual)
    coordinate_error = float(np.linalg.norm(residual @ a.lattice.matrix, axis=1).max())
    assert coordinate_error < 1e-6, f'coordinate error {coordinate_error} A'
    return coordinate_error, lattice_error


def select():
    dataset = PROJECT / 'work/PLaID/data/basic/train.csv'
    assert sha(dataset.read_bytes()) == SOURCE_SHA
    with dataset.open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: sha(r['material_id'].encode()))
    first = PROJECT / 'research/plaid_20260930/released_0_decoded.cif'
    candidates = [dict(id='anchor_Tb8Mg4', cif=first.read_text(), source=str(first.relative_to(PROJECT)), kind='released_generation')]
    released = PROJECT / 'work/PLaID/results/qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.3_esen_ehull_results_combined_sun.csv'
    with released.open() as f:
        for index, row in enumerate(csv.DictReader(f)):
            if index == 1400:
                candidates.append(dict(id='anchor_Li8V4Ni4O16', cif=row['cif'], source=str(released.relative_to(PROJECT)), source_row_zero_based=index, kind='released_generation'))
                break
    assert len(candidates) == 2
    groups = set()
    for row in rows:
        group = int(float(row['spacegroup.number']))
        if group not in groups:
            candidates.append(dict(id=row['material_id'], cif=row['cif'], source=str(dataset.relative_to(PROJECT)), dataset_space_group=group, kind='MP20_train'))
            groups.add(group)
        if len(candidates) == 100:
            break
    assert len(candidates) == 100 and len(groups) == 98
    return candidates


def decode_directory(source, target, out):
    command = [sys.executable, str(ROOT / 'naturecry_codec.py'), 'decode-directory', str(source), str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / (target.name + '_decoder.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    return json.loads((target / 'failures.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'run_001')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ['inputs', 'texts', 'decoded', 'reencoded', 'decoded_second', 'audits']:
        (out / name).mkdir()
    start = time.perf_counter()
    selected = select()
    manifest = dict(rule='Two released-generation anchors plus first-by-SHA256 entry of 98 distinct dataset space groups. No replacement on failure.',
                    dataset_sha256=SOURCE_SHA, selected=100,
                    cases=[{**{k: v for k, v in r.items() if k != 'cif'}, 'input_cif_sha256': sha(r['cif'].encode())} for r in selected])
    (out / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records = []
    for index, row in enumerate(selected, 1):
        mid = row['id']
        path = out / 'inputs' / (mid + '.cif')
        path.write_text(row['cif'])
        record = {k: v for k, v in row.items() if k != 'cif'}
        record['stage'] = 'encode'
        try:
            began = time.perf_counter()
            text, audit = encode_cif(path, symprec=1e-3)
            (out / 'texts' / (mid + '.txt')).write_text(text)
            (out / 'audits' / (mid + '.json')).write_text(json.dumps(audit, indent=2) + '\n')
            original = parse(path)
            record.update(status='encoded', formula=original.composition.reduced_formula,
                          space_group=audit['space_group'], hall=audit['hall_number'], atoms=len(original),
                          orbits=len(audit['nodes']), free_coordinate_dof=sum(len(n['parameters']) for n in audit['nodes']),
                          residual_atoms=audit['residual_atoms'], has_frame_transform=bool(audit['frame_tokens']),
                          uncorrected_input_error_A=audit['uncorrected_input_error_A'],
                          natural_bytes=len(text.encode()), natural_whitespace_words=len(text.split()),
                          input_cif_bytes=len(row['cif'].encode()), encode_seconds=time.perf_counter()-began)
        except Exception as error:
            record.update(status='fail', error=repr(error))
            (out / 'audits' / (mid + '_failure.txt')).write_text(traceback.format_exc())
        records.append(record)
        if index % 10 == 0:
            print(json.dumps({'encoded_attempts': index, 'encoding_failures': sum(r['status'] == 'fail' for r in records)}), flush=True)
        (out / 'records.json').write_text(json.dumps(records, indent=2) + '\n')

    failures = decode_directory(out / 'texts', out / 'decoded', out)
    for record in records:
        if record['status'] == 'fail':
            continue
        mid = record['id']
        record['stage'] = 'decode'
        try:
            if mid in failures:
                raise ValueError(failures[mid])
            original = parse(out / 'inputs' / (mid + '.cif'))
            rebuilt = parse(out / 'decoded' / (mid + '.cif'))
            record['stage'] = 'first_round_compare'
            error, lattice_error = compare(original, rebuilt)
            record.update(first_coordinate_error_A=error, first_lattice_error_A=lattice_error)
            record['stage'] = 'reencode'
            text, _ = encode_cif(out / 'decoded' / (mid + '.cif'), symprec=1e-3)
            (out / 'reencoded' / (mid + '.txt')).write_text(text)
            record['texts_identical'] = text == (out / 'texts' / (mid + '.txt')).read_text()
            record['status'] = 'reencoded'
        except Exception as error:
            record.update(status='fail', error=repr(error))
    second_failures = decode_directory(out / 'reencoded', out / 'decoded_second', out)
    for record in records:
        if record['status'] == 'fail':
            continue
        mid = record['id']
        record['stage'] = 'second_round_decode_compare'
        try:
            if mid in second_failures:
                raise ValueError(second_failures[mid])
            error, lattice_error = compare(parse(out / 'inputs' / (mid + '.cif')), parse(out / 'decoded_second' / (mid + '.cif')))
            record.update(status='pass', second_coordinate_error_A=error, second_lattice_error_A=lattice_error)
        except Exception as error:
            record.update(status='fail', error=repr(error))

    passed = [r for r in records if r['status'] == 'pass']
    encoded = [r for r in records if 'natural_bytes' in r]
    summary = dict(selected=100, passed=len(passed), failed=100-len(passed), symprec_A=1e-3,
                   atom_coordinate_tolerance_A=1e-6, lattice_component_tolerance_A=1e-8,
                   seconds=time.perf_counter()-start, datasets=Counter(r['kind'] for r in records),
                   detected_space_groups=sorted({r['space_group'] for r in encoded}),
                   cases_with_frame_transform=sum(r['has_frame_transform'] for r in encoded),
                   cases_with_precision_corrections=sum(r['residual_atoms'] > 0 for r in encoded),
                   corrected_atoms=sum(r['residual_atoms'] for r in encoded),
                   byte_totals={k: sum(r[k] for r in encoded) for k in ['natural_bytes', 'input_cif_bytes']},
                   versions={p: importlib.metadata.version(p) for p in ['numpy', 'pymatgen', 'spglib', 'sympy', 'scipy']},
                   decoder_inputs='Only NatureCry text directory + output directory; no input CIF/data path.',
                   scope='Controlled English v0.1, ordered fully occupied structures; no model, no free-form paraphrase, no CIF metadata recovery.')
    if passed:
        summary.update(max_coordinate_error_A=max(max(r['first_coordinate_error_A'], r['second_coordinate_error_A']) for r in passed),
                       max_lattice_error_A=max(max(r['first_lattice_error_A'], r['second_lattice_error_A']) for r in passed),
                       texts_identical=sum(r['texts_identical'] for r in passed))
    paths = [ROOT / 'naturecry_codec.py', ROOT / 'benchmark.py', ROOT / 'PROTOCOL.md', wyckoff.FIXTURE]
    paths += [Path(wyckoff.__file__), Path(wyckoff.__file__).with_name('program_codec.py'), Path(wyckoff.__file__).with_name('audit_wyckoff_families.py')]
    summary['dependency_sha256'] = {str(p.relative_to(PROJECT)): sha(p.read_bytes()) for p in paths}
    (out / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
