"""Frozen 100-case LLM roundtrip; first attempts only, with frames and precision."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
import spglib

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'experiments/naturecry_llm_pilot_20261002'))
from pilot import chat, ENCODER_PROMPT as PILOT_ENCODER_PROMPT, BASE, MODEL, wyckoff, parse, compare

ENCODER_PROMPT = PILOT_ENCODER_PROMPT + '''
This cohort also includes nonstandard input cells and small departures from an ideal symmetry backbone. Describe the original input cell, not just the conventional symmetry cell. If frames differ, explain the supplied relation q_standard = P q_input + s and its exact matrix/origin values in normal prose so the original cell can be recovered. Preserve the declared input composition, which may differ from the conventional-cell composition. Explain the rounding precision before corrections when needed. Every listed positional correction is necessary: identify its element and pre-correction input fractional position, and state its displacement. Corrections preserve the real input geometry rather than implying it is an exact idealized space-group structure. Convey this information in a coherent scientific narrative, not a JSON appendix or atom table.'''

DECODER_PROMPT = '''You are a blind semantic decoder of a natural-language crystal description. The user message is your only source of material-specific information. Understand the spatial narrative and produce a construction plan; do not guess missing independent numbers or look up a material. Return one JSON object only, without Markdown fences.
For sufficient information use:
{"status":"ok","space_group_number":integer,"hall_symbol":"Hall symbol stated in the text","lattice":[input_a,input_b,input_c,input_alpha,input_beta,input_gamma],"composition":{"Element":input_cell_atom_count},"orbits":[{"element":"Element","wyckoff":"multiplicity_and_letter","representative":[standard_fractional_x,standard_fractional_y,standard_fractional_z]}],"frame_transform":{"matrix":[[p11,p12,p13],[p21,p22,p23],[p31,p32,p33]],"origin_shift":[s1,s2,s3]},"round_digits":integer_or_null,"precision_corrections":[{"element":"Element","at":[input_fractional_x_before_correction,input_fractional_y_before_correction,input_fractional_z_before_correction],"delta":[fractional_dx,fractional_dy,fractional_dz]}]}
Lengths use angstroms, angles use degrees. The specified lattice and composition refer to the original input cell. Orbit representatives refer to the standard Hall coordinate frame. Copy the Hall symbol as a mathematical identifier rather than guessing a public-table integer number. Any equivalent representative of a declared orbit is allowed. Distinct independent orbits of the same element remain separate.
The geometry engine expands the public Hall symmetry orbits, applies q_input = inverse(P)(q_standard - s), selects images in the input cell, rounds fractional coordinates if round_digits is given, and applies precision corrections at their identified input-frame atom positions. If the text explicitly uses identical frames, supply identity P and zero shift. If no rounding is specified, use null. If no corrections are specified, use an empty array. All sites have full occupancy.
Numerical relations in the prose should be resolved into numbers in the plan. Do not silently replace the original geometry with an idealized one. For missing required information return {"status":"insufficient_information","reason":"..."}; for contradictory constraints return {"status":"conflict","reason":"..."}. Your input contains no source CIF or hidden data. Public crystallographic definitions are allowed; material-specific external retrieval is not.'''


def prepare_case(path):
    source = parse(path)
    full, _, audit = wyckoff.encode_cif(path, symprec=1e-3)
    rows = full.splitlines()
    frame = next((list(map(float, x.split()[1:])) for x in rows if x.startswith('FRAME ')), None)
    transform = dict(matrix=np.array(frame[:9]).reshape(3, 3).tolist(), origin_shift=frame[9:]) if frame else dict(matrix=np.eye(3).tolist(), origin_shift=[0, 0, 0])
    table = wyckoff.reference_table()[audit['space_group']]
    group = spglib.get_spacegroup_type(audit['hall_number'])
    orbits = []
    for node in audit['nodes']:
        definition = table['definitions'][node['label']]
        u = np.zeros(3)
        u[definition['axes']] = node['parameters']
        orbits.append(dict(element=node['element'], wyckoff=node['label'], representative=(definition['a'] @ u + definition['b']).tolist(),
                           multiplicity=definition['multiplicity'], coordinate_relation=node['template'],
                           independent_parameters=dict(zip(node['free_axes'], node['parameters']))))
    index = next(i for i, x in enumerate(rows) if x.startswith('RESIDUALS '))
    backbone = wyckoff.decode('\n'.join(rows[:index] + ['RESIDUALS 0', 'END']))
    corrections = []
    for row in rows[index + 1:-1]:
        atom, *delta = row.split()
        corrections.append(dict(element=str(backbone[int(atom)].specie), at=backbone[int(atom)].frac_coords.tolist(), delta=list(map(float, delta))))
    composition = {Element.from_Z(z).symbol: n for z, n in Counter(source.atomic_numbers).items()}
    facts = dict(input_lattice_parameters=list(source.lattice.parameters), input_composition=composition, input_atoms=len(source),
                 space_group_number=audit['space_group'], space_group_symbol=group.international_short, hall_symbol=group.hall_symbol,
                 coordinate_reference='Fractional coordinates. Standard and input frames satisfy q_standard = P q_input + s.',
                 frame_transform=transform, round_digits=audit['round_digits'],
                 independent_orbits=orbits, precision_corrections=corrections)
    oracle = dict(status='ok', space_group_number=audit['space_group'], hall_symbol=group.hall_symbol,
                  lattice=list(source.lattice.parameters), composition=composition,
                  orbits=[{k: o[k] for k in ['element', 'wyckoff', 'representative']} for o in orbits],
                  frame_transform=transform, round_digits=audit['round_digits'], precision_corrections=corrections)
    return facts, oracle


def build_structure(plan):
    if plan.get('status') != 'ok':
        raise ValueError(f'Decoder status {plan.get("status")}: {plan.get("reason", "")}')
    sg = int(plan['space_group_number'])
    normalize = lambda s: ''.join(s.split()).replace('−', '-')
    halls = [h for h in range(1, 531) if spglib.get_spacegroup_type(h).number == sg and normalize(spglib.get_spacegroup_type(h).hall_symbol) == normalize(plan['hall_symbol'])]
    table = wyckoff.reference_table()[sg]
    if len(halls) != 1 or halls[0] != table['hall']:
        raise ValueError('Hall symbol/space-group setting mismatch')
    operations = spglib.get_symmetry_from_database(halls[0])
    nodes = []
    for orbit in plan['orbits']:
        definition = table['definitions'][orbit['wyckoff']]
        q = np.array(orbit['representative'], dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('Invalid orbit representative')
        images = (np.einsum('nij,j->ni', operations['rotations'], q) + operations['translations']) % 1
        unique = []
        for image in images:
            if not any(np.max(np.abs(image - old - np.rint(image - old))) < 1e-10 for old in unique):
                unique.append(image)
        parameters = wyckoff.fit_parameters(np.array(unique), definition)
        if len(unique) != definition['multiplicity'] or parameters is None:
            raise ValueError(f'Orbit coordinate conflicts with {orbit["wyckoff"]}')
        nodes.append((Element(orbit['element']).Z, orbit['wyckoff'], parameters))
    q, species = wyckoff.expand_standard(sg, nodes)
    frame = plan['frame_transform']
    matrix, shift = np.array(frame['matrix'], dtype=float), np.array(frame['origin_shift'], dtype=float)
    if matrix.shape != (3, 3) or shift.shape != (3,) or not np.isfinite(matrix).all() or not np.isfinite(shift).all() or abs(np.linalg.det(matrix)) < 1e-12 or np.max(np.abs(matrix)) > 4 or np.max(np.abs(shift)) > 4:
        raise ValueError('Invalid or unsupported input-frame transform')
    q, species = wyckoff.to_input_frame(q, species, matrix, shift)
    digits = plan.get('round_digits')
    if digits is not None:
        if not isinstance(digits, int) or not 0 <= digits <= 15:
            raise ValueError('Invalid rounding precision')
        q = np.round(q, digits) % 1
    cell = np.array(plan['lattice'], dtype=float)
    if cell.shape != (6,) or not np.isfinite(cell).all() or min(cell[:3]) <= 0 or not np.all((cell[3:] > 0) & (cell[3:] < 180)):
        raise ValueError('Invalid input lattice')
    lattice = Lattice.from_parameters(*cell)
    corrected = set()
    for correction in plan.get('precision_corrections', []):
        point, delta = np.array(correction['at'], dtype=float), np.array(correction['delta'], dtype=float)
        if point.shape != (3,) or delta.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(delta).all():
            raise ValueError('Invalid precision correction')
        d = q - point
        d -= np.rint(d)
        indices = np.flatnonzero((np.linalg.norm(d @ lattice.matrix, axis=1) < 1e-8) & (np.array(species) == Element(correction['element']).Z))
        if len(indices) != 1 or int(indices[0]) in corrected:
            raise ValueError('Precision correction has an ambiguous or repeated atom address')
        atom = int(indices[0])
        corrected.add(atom)
        q[atom] = (q[atom] + delta) % 1
    counts = {Element.from_Z(z).symbol: n for z, n in Counter(species).items()}
    if counts != plan['composition']:
        raise ValueError('Input-cell atom counts disagree with orbit/frame expansion')
    return Structure(lattice, species, q)


def process_case(case, key):
    record = dict(case=case.name, stage='encode', status='fail', blind_decoder=True)
    try:
        original = parse(case / 'input.cif')
        facts = json.loads((case / 'encoder_facts.json').read_text())
        description, enc = chat([dict(role='system', content=ENCODER_PROMPT), dict(role='user', content=json.dumps(facts, indent=2))], 0.4, key, case / 'encoder')
        (case / 'description.txt').write_text(description + '\n')
        record.update(encoder=enc, description_characters=len(description), stage='decode')
        response, dec = chat([dict(role='system', content=DECODER_PROMPT), dict(role='user', content=description)], 0, key, case / 'decoder')
        record['decoder'] = dec
        plan = json.loads(response)
        (case / 'decoded_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        record['stage'] = 'geometry_execution'
        rebuilt = build_structure(plan)
        CifWriter(rebuilt, symprec=None, significant_figures=14).write_file(case / 'decoded.cif')
        record['stage'] = 'original_structure_match'
        error, lattice_error = compare(original, parse(case / 'decoded.cif'))
        record.update(status='pass', coordinate_error_A=error, lattice_error_A=lattice_error)
    except Exception as error:
        record['error'] = str(error).replace(key, '[REDACTED]')
        record['failure_kind'] = 'transport_or_output_incomplete' if ('HTTP ' in str(error) or 'timed out' in str(error) or 'Incomplete model output' in str(error)) else 'model_or_structure_failure'
    (case / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'run_001')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get('SCIENCEPAL_API_KEY') or getpass.getpass('SciencePal API key: ')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = PROJECT / 'experiments/naturecry_roundtrip_20261001/run_001'
    selection = json.loads((frozen / 'selection.json').read_text())
    assert selection['selected'] == 100
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    snapshots = out / 'sources'
    snapshots.mkdir()
    (snapshots / 'cohort.py').write_bytes(Path(__file__).read_bytes())
    (snapshots / 'pilot.py').write_bytes((PROJECT / 'experiments/naturecry_llm_pilot_20261002/pilot.py').read_bytes())
    (out / 'prompts.json').write_text(json.dumps(dict(encoder=ENCODER_PROMPT, decoder=DECODER_PROMPT), indent=2) + '\n')
    cases = []
    for item in selection['cases']:
        case = out / item['id']
        case.mkdir()
        path = frozen / 'inputs' / (item['id'] + '.cif')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['input_cif_sha256']
        (case / 'input.cif').write_bytes(path.read_bytes())
        facts, oracle = prepare_case(path)
        compare(parse(path), build_structure(oracle))
        (case / 'encoder_facts.json').write_text(json.dumps(facts, indent=2) + '\n')
        cases.append(case)
    print('Frozen 100 cases; oracle geometry preflight passed; starting isolated LLM requests.', flush=True)
    started = time.perf_counter()
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(process_case, case, key): case for case in cases}
        for task in as_completed(tasks):
            record = task.result()
            records.append(record)
            (out / 'records.json').write_text(json.dumps(sorted(records, key=lambda r: r['case']), indent=2) + '\n')
            print(json.dumps(dict(completed=len(records), passed=sum(r['status']=='pass' for r in records), last_case=record['case'], last_status=record['status'], last_stage=record['stage'])), flush=True)
    usages = [r[k]['usage'] for r in records for k in ['encoder', 'decoder'] if k in r]
    summary = dict(provider=BASE, model=MODEL, selected=100, completed=len(records), passed=sum(r['status']=='pass' for r in records),
                   failed=sum(r['status']=='fail' for r in records), failures_by_stage=Counter(r['stage'] for r in records if r['status']=='fail'),
                   failures_by_kind=Counter(r['failure_kind'] for r in records if r['status']=='fail'),
                   seconds=time.perf_counter()-started, workers=args.workers, automatic_retries=0,
                   successful_model_requests=len(usages), total_tokens=sum(u['total_tokens'] for u in usages),
                   coordinate_tolerance_A=1e-6, lattice_tolerance_A=1e-8, original_structure_is_hard_requirement=True,
                   auxiliary_tests_run=False, scope='Frozen 100-case cohort; free LLM prose; isolated semantic decoder; public-table geometry with original frames and precision corrections.')
    passed = [r for r in records if r['status']=='pass']
    if passed:
        summary['max_coordinate_error_A'] = max(r['coordinate_error_A'] for r in passed)
        summary['max_lattice_error_A'] = max(r['lattice_error_A'] for r in passed)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
