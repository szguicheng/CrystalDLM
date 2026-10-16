"""Two-case SciencePal/Kimi natural-description and blind semantic-decoding pilot."""
import argparse
from collections import Counter
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
import requests
import spglib

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'experiments/naturecry_roundtrip_20261001'))
from benchmark import compare, parse
from naturecry_codec import wyckoff

BASE = 'https://llm.sciencepal.ai/v1'
MODEL = 'tokenhub/kimi-k3'
ENCODER_PROMPT = '''You are the verbalizer in a crystal-to-natural-language codec. Write an accurate, natural scientific construction narrative from the trusted crystal facts provided by the user. Another crystallographer must be able to reconstruct the same atomic geometry from your prose alone.
Choose your own wording, paragraph order, and spatial organization. Describe the three-dimensional arrangement, repetitions, symmetry relationships, planes and relative positions when supported by the facts. Integrate the necessary dimensions, coordinate reference, exact positional values, element identities and occupancies into the narrative. You may explain ratios or geometric relations, but do not remove necessary independent numbers. A scientific space-group/Hall name or number is allowed to identify the symmetry convention.
Your entire output must be ordinary natural-language prose. Do not output CIF, JSON, source code, a key-value block, a fixed per-orbit template, an appended atom-coordinate table, hidden encodings, or material IDs. Do not merely prepend English filler to every fact. Numerical coordinates and equations may appear within genuine explanatory sentences when necessary for precision.
Do not invent bonding, coordination, stability, material dimensionality, or chemistry claims: no such evidence is supplied. Preserve every independent positional value and the specified lattice. Explain all independent sublattices, including multiple inequivalent orbits of the same element. End with the completed description, not commentary about the task.'''

DECODER_PROMPT = '''You are a blind semantic decoder of natural-language crystal descriptions. The user message is your only source of material-specific information. Read its spatial meaning, identify the coordinate convention and symmetry, resolve numerical relations, and produce a finite crystal construction plan. Do not assume unstated positional values, look up a named material, or claim a complete reconstruction when required information is missing.
Return one JSON object only, without a Markdown fence. For a complete conventional-cell description use this schema:
{"status":"ok","space_group_number":integer,"hall_symbol":"literal Hall symbol specified in the prose","lattice":[a,b,c,alpha,beta,gamma],"composition":{"Element":integer_atom_count},"orbits":[{"element":"Element","wyckoff":"multiplicity_and_letter","representative":[fractional_x,fractional_y,fractional_z]}]}
Lengths are in angstroms and angles in degrees. Copy the Hall symbol from the description, preserving its mathematical meaning; the geometry engine resolves the public-table integer index, so do not guess a Hall number. Each representative may be any equivalent point of that Wyckoff orbit in the declared Hall setting. List each independent orbit once; multiple orbits of the same element remain separate. The geometry engine will apply all public Hall-setting symmetry operations, retain unique points per orbit, and write the full unit cell. Sites have full occupancy in this pilot.
Use the supplied meanings and numbers rather than matching any fixed wording. You may rely on public crystallographic definitions to identify a Wyckoff orbit or Hall convention. If the text does not determine the necessary lattice, independent positions or setting, return {"status":"insufficient_information","reason":"..."}; for incompatible constraints return {"status":"conflict","reason":"..."}. Do not output guessed coordinates to fill missing information.'''


def build_structure(plan):
    if plan.get('status') != 'ok':
        raise ValueError(f'Decoder status {plan.get("status")}: {plan.get("reason", "")}')
    sg = int(plan['space_group_number'])
    normalize = lambda symbol: ''.join(symbol.split()).replace('−', '-')
    matches = [h for h in range(1, 531) if normalize(spglib.get_spacegroup_type(h).hall_symbol) == normalize(plan['hall_symbol'])]
    if len(matches) != 1:
        raise ValueError('The supplied Hall symbol does not identify a unique public-table setting')
    hall = matches[0]
    table = wyckoff.reference_table()[sg]
    if hall != table['hall'] or spglib.get_spacegroup_type(hall).number != sg:
        raise ValueError('Space-group and public-table Hall setting disagree')
    cell = np.array(plan['lattice'], dtype=float)
    if cell.shape != (6,) or not np.isfinite(cell).all() or min(cell[:3]) <= 0 or not np.all((cell[3:] > 0) & (cell[3:] < 180)):
        raise ValueError('Invalid lattice')
    lattice = Lattice.from_parameters(*cell)
    positions, species = [], []
    operations = spglib.get_symmetry_from_database(hall)
    for orbit in plan['orbits']:
        label = orbit['wyckoff']
        definition = table['definitions'][label]
        point = np.array(orbit['representative'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid representative position')
        images = (np.einsum('nij,j->ni', operations['rotations'], point) + operations['translations']) % 1
        unique = []
        for q in images:
            if not any(np.max(np.abs(q - p - np.rint(q - p))) < 1e-10 for p in unique):
                unique.append(q)
        if len(unique) != definition['multiplicity'] or wyckoff.fit_parameters(np.array(unique), definition) is None:
            raise ValueError(f'Representative conflicts with declared Wyckoff orbit {label}')
        z = Element(orbit['element']).Z
        for q in unique:
            if any(z == old_z and np.max(np.abs(q - p - np.rint(q - p))) < 1e-10 for p, old_z in zip(positions, species)):
                raise ValueError('Independent orbits overlap')
            positions.append(q)
            species.append(z)
    structure = Structure(lattice, species, positions)
    counts = {Element.from_Z(z).symbol: n for z, n in Counter(species).items()}
    if counts != plan['composition']:
        raise ValueError('Orbit expansion disagrees with the declared composition')
    metric = lattice.matrix @ lattice.matrix.T
    if any(np.max(np.abs(r.T @ metric @ r - metric)) > 1e-7 for r in operations['rotations']):
        raise ValueError('Lattice metric conflicts with the declared symmetry')
    return structure


def facts_for(path, audit_path):
    source = parse(path)
    audit = json.loads(audit_path.read_text())
    if audit['frame_tokens'] or audit['residual_atoms']:
        raise ValueError('The two-case pilot requires an uncorrected standard-frame input')
    table = wyckoff.reference_table()[audit['space_group']]
    group = spglib.get_spacegroup_type(audit['hall_number'])
    orbits = []
    for node in audit['nodes']:
        d = table['definitions'][node['label']]
        u = np.zeros(3)
        u[d['axes']] = node['parameters']
        orbits.append(dict(element=node['element'], wyckoff=node['label'], multiplicity=d['multiplicity'],
                           representative=(d['a'] @ u + d['b']).tolist(), coordinate_relation=node['template'],
                           independent_parameters=dict(zip(node['free_axes'], node['parameters'])), occupancy=1))
    planes = []
    for axis, name in enumerate('abc'):
        values = sorted(set(np.round(source.frac_coords[:, axis] % 1, 10)))
        if len(values) <= 4:
            for value in values:
                sites = [site for site in source if abs((float(site.frac_coords[axis]) - value + 0.5) % 1 - 0.5) < 1e-9]
                planes.append(dict(axis=name, fractional_coordinate=float(value), atom_counts=dict(Counter(str(s.specie) for s in sites))))
    facts = dict(lattice_parameters=list(source.lattice.parameters), composition={Element.from_Z(z).symbol: n for z, n in Counter(source.atomic_numbers).items()},
                 space_group_number=audit['space_group'], space_group_symbol=group.international_short,
                 hall_number=audit['hall_number'], hall_symbol=group.hall_symbol,
                 coordinate_reference='Fractional coordinates along a,b,c; the standard Hall frame and original input frame are identical.',
                 independent_orbits=orbits, verified_atomic_plane_groups=planes)
    return source, facts


def chat(messages, temperature, key, prefix):
    payload = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=8192)
    prefix.with_suffix('.request.json').write_text(json.dumps(payload, indent=2) + '\n')
    start = time.perf_counter()
    response = requests.post(BASE + '/chat/completions', headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=180)
    raw = response.text.replace(key, '[REDACTED]')
    prefix.with_suffix('.response.json').write_text(raw)
    if response.status_code != 200:
        raise RuntimeError(f'HTTP {response.status_code}: {raw[:250]}')
    data = response.json()
    choice = data['choices'][0]
    content = choice['message'].get('content')
    if choice['finish_reason'] != 'stop' or not content:
        raise RuntimeError(f'Incomplete model output: finish_reason={choice["finish_reason"]}')
    receipt = dict(requested_model=MODEL, returned_model=data.get('model'), response_id=data.get('id'),
                   finish_reason=choice['finish_reason'], usage=data.get('usage'), seconds=time.perf_counter()-start)
    prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return content, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'run_001')
    parser.add_argument('--reuse-descriptions', type=Path, help='Explicit interface comparison using prior unchanged encoder outputs')
    args = parser.parse_args()
    key = os.environ.get('SCIENCEPAL_API_KEY') or getpass.getpass('SciencePal API key: ')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = PROJECT / 'experiments/naturecry_roundtrip_20261001/run_001'
    results = []
    for name in ['anchor_Tb8Mg4', 'anchor_Li8V4Ni4O16']:
        case = out / name
        case.mkdir()
        path = frozen / 'inputs' / (name + '.cif')
        original, facts = facts_for(path, frozen / 'audits' / (name + '.json'))
        (case / 'input.cif').write_text(path.read_text())
        (case / 'encoder_facts.json').write_text(json.dumps(facts, indent=2) + '\n')
        record = dict(case=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_formula=original.formula,
                      status='fail', stage='encode', blind_decoder=True)
        try:
            if args.reuse_descriptions:
                previous = args.reuse_descriptions.resolve() / name
                description = (previous / 'description.txt').read_text().rstrip('\n')
                enc = json.loads((previous / 'encoder.receipt.json').read_text())
                record['encoder_reused_from'] = str(previous.relative_to(PROJECT))
            else:
                print(f'{name}: encoder request', flush=True)
                description, enc = chat([dict(role='system', content=ENCODER_PROMPT),
                                         dict(role='user', content=json.dumps(facts, indent=2))], 0.4, key, case / 'encoder')
            (case / 'description.txt').write_text(description + '\n')
            record.update(encoder=enc, description_characters=len(description), stage='decode')
            print(f'{name}: isolated blind decoder request', flush=True)
            answer, dec = chat([dict(role='system', content=DECODER_PROMPT), dict(role='user', content=description)],
                               0, key, case / 'decoder')
            record['decoder'] = dec
            plan = json.loads(answer)
            (case / 'decoded_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
            record['stage'] = 'geometry_execution'
            rebuilt = build_structure(plan)
            CifWriter(rebuilt, symprec=None, significant_figures=14).write_file(case / 'decoded.cif')
            record['stage'] = 'original_structure_match'
            coordinate_error, lattice_error = compare(original, parse(case / 'decoded.cif'))
            record.update(status='pass', coordinate_error_A=coordinate_error, lattice_error_A=lattice_error,
                          output_atoms=len(rebuilt), output_formula=rebuilt.formula)
        except Exception as error:
            record['error'] = str(error).replace(key, '[REDACTED]')
        results.append(record)
        (out / 'records.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    summary = dict(provider=BASE, requested_model=MODEL, selected=2, passed=sum(r['status']=='pass' for r in results),
                   failed=sum(r['status']=='fail' for r in results), coordinate_tolerance_A=1e-6, lattice_tolerance_A=1e-8,
                   hard_requirement='Reconstructed crystal matches the original input structure.',
                   auxiliary_tests_run=False, automatic_semantic_or_transport_retries=0,
                   encoder_descriptions_reused=bool(args.reuse_descriptions), decoder_interface='Literal Hall symbol; public-table index resolution',
                   source='Two previously discussed released-generation anchors; no 100-case expansion.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
