"""CIF -> standard-cell element/Wyckoff occupation records on Slurm."""
import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import spglib
from pymatgen.core import Structure

GROUPS = None
TABLE = None
COUNT_CACHE = None
GATE1 = Path('/scratch/chenggui2/CrystalDLM/gate1_20260908')
sys.path.insert(0, str(GATE1))
from count_vocab import coefficients, read_groups


def initialize():
    global GROUPS, TABLE, COUNT_CACHE
    GROUPS = read_groups(GATE1 / 'input')
    TABLE = {g: {p['letter']: p for p in x['positions']} for g, x in GROUPS.items()}
    COUNT_CACHE = {}


def process(payload):
    index, row, split = payload
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            structure = Structure.from_str(row['cif'], fmt='cif')
        assert structure.is_ordered, 'disordered input'
        numbers = [int(site.specie.Z) for site in structure]
        cell = (structure.lattice.matrix, structure.frac_coords, numbers)
        standardized = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False,
                                               symprec=.1, angle_tolerance=5.)
        assert standardized is not None, 'standardization failed'
        lattice, positions, species = standardized
        data = spglib.get_symmetry_dataset(standardized, symprec=.1, angle_tolerance=5.)
        assert data is not None, 'symmetry detection failed'
        sg = int(data.number)
        original_counts, counts = Counter(numbers), Counter(int(x) for x in species)
        scale = Fraction(len(species), len(numbers))
        assert set(counts) == set(original_counts)
        assert all(Fraction(counts[e], original_counts[e]) == scale for e in counts), 'composition changed'
        orbits = []
        for representative in sorted(set(int(x) for x in data.equivalent_atoms)):
            indices = np.flatnonzero(data.equivalent_atoms == representative)
            letter = data.wyckoffs[representative]
            z = int(species[representative])
            assert all(int(species[i]) == z for i in indices), 'mixed orbit'
            assert all(data.wyckoffs[i] == letter for i in indices), 'inconsistent orbit letters'
            definition = TABLE[sg][letter]
            assert len(indices) == definition['multiplicity'], ('setting/multiplicity mismatch', sg, letter, len(indices), definition['multiplicity'])
            orbits.append({'Z': z, 'letter': letter, 'multiplicity': len(indices),
                           'dof': definition['dof'], 'site_symmetry': data.site_symmetry_symbols[representative],
                           'representative_fractional': [float(x) for x in positions[representative]]})
        orbits.sort(key=lambda x: (x['Z'], x['letter'], x['representative_fractional']))
        all_fixed = [x['letter'] for x in orbits if x['dof'] == 0]
        assert len(set(all_fixed)) == len(all_fixed), 'repeated fixed Wyckoff position'
        elements = []
        for z, n in sorted(counts.items()):
            profile = Counter(o['letter'] for o in orbits if o['Z'] == z)
            assert sum(TABLE[sg][letter]['multiplicity'] * k for letter, k in profile.items()) == n
            key = (sg, n)
            if key not in COUNT_CACHE:
                COUNT_CACHE[key] = coefficients(GROUPS[sg]['positions'], n)[n]
            # Decode the symbolic profile, independently of the input orbit ordering.
            decoded = sorted((z, letter) for letter, k in profile.items() for _ in range(k))
            target = sorted((o['Z'], o['letter']) for o in orbits if o['Z'] == z)
            assert decoded == target, 'symbolic roundtrip failed'
            elements.append({'Z': z, 'N': n, 'profile': sorted(profile.items()),
                             'candidate_count': COUNT_CACHE[key], 'orbit_count': sum(profile.values())})
        return {'ok': True, 'record': {'material_id': row['material_id'], 'split': split,
                'source_index': index, 'space_group': sg, 'hall_number': int(data.hall_number),
                'choice': data.choice, 'source_space_group': int(float(row['spacegroup.number'])),
                'original_atom_count': len(numbers), 'conventional_atom_count': len(species),
                'elements': elements, 'orbits': orbits, 'lattice': lattice.tolist(),
                'band_gap': float(row['band_gap']), 'formation_energy': float(row['formation_energy_per_atom']),
                'warnings': sorted(set(str(w.message)[:400] for w in caught))}}
    except Exception as exc:
        return {'ok': False, 'failure': {'material_id': row['material_id'], 'split': split,
                'source_index': index, 'type': type(exc).__name__, 'reason': str(exc)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('processed'))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm compute node.'
    start = time.perf_counter()
    path = Path('input') / f'{args.split}.csv'
    manifest = json.loads(Path('input/manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][args.split]['sha256']
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    success = failure = 0
    mismatches = 0
    warnings_count = Counter()
    failures = []
    elements_total = multi_orbit_elements = orbits_total = 0
    candidate_histogram = Counter()
    group_counts = Counter()
    with (args.output / f'{args.split}.jsonl').open('w') as output, \
         ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        payloads = ((i, row, args.split) for i, row in enumerate(rows))
        for i, result in enumerate(pool.map(process, payloads, chunksize=32)):
            if result['ok']:
                record = result['record']
                output.write(json.dumps(record, separators=(',', ':')) + '\n')
                success += 1
                mismatches += record['space_group'] != record['source_space_group']
                group_counts[record['space_group']] += 1
                orbits_total += len(record['orbits'])
                for e in record['elements']:
                    elements_total += 1
                    multi_orbit_elements += e['orbit_count'] > 1
                    candidate_histogram[e['candidate_count']] += 1
                warnings_count.update(record['warnings'])
            else:
                failures.append(result['failure'])
                failure += 1
            if (i + 1) % 1000 == 0:
                print(json.dumps({'processed': i + 1, 'success': success, 'failure': failure,
                                  'elapsed_seconds': time.perf_counter() - start}), flush=True)
    summary = {'split': args.split, 'limit': args.limit, 'source_rows': len(rows), 'success': success,
               'failures': failures, 'symmetry_differs_from_source_metadata': mismatches,
               'element_slots': elements_total, 'orbit_slots': orbits_total,
               'multi_orbit_elements': multi_orbit_elements,
               'candidate_histogram': dict(sorted(candidate_histogram.items())),
               'structures_by_group': dict(sorted(group_counts.items())),
               'warnings': dict(warnings_count),
               'symprec_A': .1, 'angle_tolerance_deg': 5.,
               'host': socket.gethostname(), 'job_id': os.environ['SLURM_JOB_ID'],
               'workers': args.workers, 'elapsed_seconds': time.perf_counter() - start,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'data_manifest': manifest}
    (args.output / f'{args.split}_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ['candidate_histogram', 'structures_by_group', 'data_manifest', 'warnings', 'failures']}), flush=True)
    if failure:
        print(f'{failure} failed records are preserved in the summary; investigate before model training.', flush=True)


if __name__ == '__main__':
    main()
