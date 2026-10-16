"""Train-only CIF pilot for a space-group edge-orbit codec on Slurm."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
import spglib
from pymatgen.core import Structure
from symmetry_graph_codec import encode, decode, edge_key, match_sites, wrap


def process(row, radius):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source = Structure.from_str(row['cif'], fmt='cif')
    assert source.is_ordered
    cell = spglib.standardize_cell((source.lattice.matrix, source.frac_coords, source.atomic_numbers),
                                   to_primitive=False, no_idealize=False, symprec=.1, angle_tolerance=5.)
    assert cell is not None
    lattice, positions, species = cell
    positions = wrap(positions)
    symmetry = spglib.get_symmetry_dataset((lattice, positions, species), symprec=.1, angle_tolerance=5.)
    assert symmetry is not None
    crystal = Structure(lattice, species, positions)
    centers, targets, images, distances = crystal.get_neighbor_list(radius, numerical_tol=1e-8)
    edges = {edge_key(i, j, image) for i, j, image, d in zip(centers, targets, images, distances) if d > 1e-8}
    encoded, orbit_sizes = encode(positions, species, symmetry.rotations, symmetry.translations,
                                 symmetry.equivalent_atoms, symmetry.wyckoffs, edges, lattice,
                                 symmetry.number, symmetry.hall_number)
    # JSON roundtrip and independent atom expansion: original full sites/graph
    # are not supplied to the decoder.
    encoded = json.loads(json.dumps(encoded))
    decoded_positions, decoded_species, decoded_edges = decode(encoded)
    assert len(decoded_positions) == len(positions)
    mapping, shifts = match_sites(positions, species, decoded_positions, decoded_species)
    assert len(set(mapping.tolist())) == len(positions)
    expected = {edge_key(mapping[i], mapping[j], shifts[j] + np.asarray(n) - shifts[i]) for i, j, *n in edges}
    assert expected == decoded_edges, f'edge roundtrip differs: {len(expected ^ decoded_edges)}'
    maximum_coordinate_error = float(np.max(np.abs(positions - decoded_positions[mapping] - shifts)))
    assert maximum_coordinate_error < 1e-6
    primitive = spglib.standardize_cell((lattice, positions, species), to_primitive=True,
                                       no_idealize=True, symprec=1e-5)
    assert primitive is not None
    plattice, ppositions, pspecies = primitive
    pcrystal = Structure(plattice, pspecies, wrap(ppositions))
    pi, pj, pn, pd = pcrystal.get_neighbor_list(radius, numerical_tol=1e-8)
    pedges = {edge_key(i, j, n) for i, j, n, d in zip(pi, pj, pn, pd) if d > 1e-8}
    assert len(positions) % len(ppositions) == 0
    centering = len(positions) // len(ppositions)
    assert len(edges) == len(pedges) * centering, 'primitive/conventional radius graph counts differ'
    return {'material_id': row['material_id'], 'space_group': int(symmetry.number),
            'atoms': len(positions), 'atom_orbits': len(encoded['header']['nodes']),
            'undirected_periodic_edges': len(edges), 'edge_orbits': len(encoded['edge_tokens']),
            'primitive_atoms': len(ppositions), 'primitive_periodic_edges': len(pedges),
            'conventional_to_primitive_factor': centering,
            'symmetry_operations_in_explicit_header': len(symmetry.rotations),
            'edge_orbit_sizes': orbit_sizes, 'maximum_coordinate_error_fractional': maximum_coordinate_error,
            'original_atoms': len(source), 'radius_A': radius}, encoded


def main():
    assert os.environ.get('SLURM_JOB_ID')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--radius', type=float, default=4.5)
    args = parser.parse_args()
    start = time.perf_counter()
    source = Path('input/train.csv')
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    rows = sorted(rows, key=lambda r: hashlib.sha256(r['material_id'].encode()).hexdigest())[:args.limit]
    output = Path(f'symmetry_graph_audit/train_{args.limit}')
    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for i, row in enumerate(rows):
        try:
            result, encoded = process(row, args.radius)
            results.append(result)
            (output / f"{row['material_id']}.json").write_text(json.dumps(encoded))
        except Exception as error:
            failures.append({'material_id': row['material_id'], 'error_type': type(error).__name__, 'error': str(error)})
        if (i + 1) % 8 == 0:
            print(json.dumps({'processed': i + 1, 'success': len(results), 'failures': len(failures),
                              'seconds': time.perf_counter() - start}), flush=True)
    summary = {'selected_train_structures': len(rows), 'success': len(results), 'failures': failures,
               'total_atoms': sum(r['atoms'] for r in results), 'total_atom_orbits': sum(r['atom_orbits'] for r in results),
               'total_periodic_edges': sum(r['undirected_periodic_edges'] for r in results),
               'total_edge_orbits': sum(r['edge_orbits'] for r in results),
               'total_primitive_atoms': sum(r['primitive_atoms'] for r in results),
               'total_primitive_periodic_edges': sum(r['primitive_periodic_edges'] for r in results),
               'space_groups': sorted({r['space_group'] for r in results}),
               'radius_A': args.radius, 'graph_definition': 'Undirected radius-neighbor graph, not a chemical-bond inference',
               'geometry_scope': 'Idealized conventional cell; ASU coordinates and explicit operation table retained',
               'not_a_coordinate_free_decoder': True, 'model_benefit_not_tested': True,
               'record_count_reduction_not_serialized_token_compression': True,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'code_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['symmetry_graph_codec.py','audit_symmetry_graph.py']},
               'job_id': os.environ['SLURM_JOB_ID'], 'run_id': os.environ.get('CRYSTALDLM_RUN_ID'),
               'seconds': time.perf_counter() - start}
    (output / 'records.json').write_text(json.dumps(results, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
