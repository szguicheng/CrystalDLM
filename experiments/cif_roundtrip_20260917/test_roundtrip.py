"""One end-to-end test: real CIF -> independent decoder -> CIF -> encoder."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
CLI = ROOT / 'cif_codec.py'


def compact(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True).encode()


def parse(path):
    return CifParser(str(path), frac_tolerance=0).parse_structures(primitive=False)[0]


def compare(a, b):
    assert len(a) == len(b), 'atom count differs'
    assert a.composition == b.composition, 'composition differs'
    lattice_error = float(np.max(np.abs(a.lattice.matrix - b.lattice.matrix)))
    assert lattice_error < 1e-8, f'lattice differs by {lattice_error} A'
    distances = a.lattice.get_all_distances(a.frac_coords, b.frac_coords)
    same = np.array(a.atomic_numbers)[:, None] == np.array(b.atomic_numbers)[None, :]
    rows, columns = linear_sum_assignment(np.where(same, distances, 1e6))
    assert np.array_equal(rows, np.arange(len(a)))
    shifts = np.rint(a.frac_coords - b.frac_coords[columns]).astype(int)
    residual = (a.frac_coords-b.frac_coords[columns]-shifts)@a.lattice.matrix
    maximum = float(np.linalg.norm(residual,axis=1).max())
    assert maximum < 1e-6, f'coordinate error {maximum} A exceeds 1e-6 A'
    return columns, shifts, maximum, lattice_error


def cli(*args):
    r = subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='results')
    parser.add_argument('--symprec',type=float,default=1e-7)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run scientific checks on a Slurm node'
    assert CLI.exists(), 'CIF encoder/decoder entry point is not implemented yet'
    from cif_codec import decode_bundle, neighbors
    from symmetry_graph_codec import edge_key
    from discrete_graph_prototype import integer_tree, expand
    began = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_bytes = Path(args.csv).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == '133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d'
    with open(args.csv) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: hashlib.sha256(r['material_id'].encode()).hexdigest())
    selected, groups = [], set()
    for row in rows:
        group = int(float(row['spacegroup.number']))
        if group not in groups:
            selected.append(row)
            groups.add(group)
        if len(selected) == 10:
            break
    manifest = [{'id': r['material_id'], 'formula': r['pretty_formula'], 'dataset_space_group': int(float(r['spacegroup.number'])), 'cif_sha256': hashlib.sha256(r['cif'].encode()).hexdigest()} for r in selected]
    (out/'selection.json').write_text(json.dumps({'rule':'Sort all training material IDs by SHA256, take first entry of each previously unseen dataset space group, stop at 10. No replacement on failure.', 'source_sha256': hashlib.sha256(source_bytes).hexdigest(), 'materials':manifest}, indent=2))
    results = []
    for row in selected:
        mid = row['material_id']
        case = out / mid
        case.mkdir(exist_ok=True)
        src, encoded, decoded, reencoded = [case/name for name in ['input.cif','encoded.json','decoded.cif','reencoded.json']]
        src.write_text(row['cif'])
        record = {'id':mid, 'formula':row['pretty_formula'], 'dataset_space_group':int(float(row['spacegroup.number']))}
        try:
            original = parse(src)
            cli('encode', src, encoded, '--symprec', args.symprec)
            bundle = json.loads(encoded.read_text())
            prototype, geometry = bundle['prototype'], bundle['geometry']
            assert integer_tree(prototype), 'continuous coordinates leaked into discrete prototype'
            assert set(geometry) == {'lattice','free_vectors'}
            # The decoder subprocess receives only encoded JSON, never the original CIF.
            cli('decode', encoded, decoded)
            rebuilt = parse(decoded)
            mapping, shifts, coord_err, lattice_err = compare(original, rebuilt)
            expanded, decoded_edges = decode_bundle(bundle)
            m0, s0, expansion_error, _ = compare(original, expanded)
            original_edges = neighbors(original, bundle['radius_A'])
            expected = {edge_key(m0[i], m0[j], np.asarray(n)+s0[j]-s0[i]) for i,j,*n in original_edges}
            assert expected == decoded_edges, f'edge loss/extra: {len(expected ^ decoded_edges)}'
            reread_edges = neighbors(rebuilt, bundle['radius_A'])
            expected_cif = {edge_key(mapping[i],mapping[j],np.asarray(n)+shifts[j]-shifts[i]) for i,j,*n in original_edges}
            assert expected_cif == reread_edges, 'radius graph changed after CIF output'
            # Re-encode the emitted CIF and decode again to check both directions.
            cli('encode', decoded, reencoded, '--symprec', args.symprec)
            again = json.loads(reencoded.read_text())
            rebuilt_again, edges_again = decode_bundle(again)
            m1,s1,second_error,_ = compare(rebuilt, rebuilt_again)
            expected_again = {edge_key(m1[i],m1[j],np.asarray(n)+s1[j]-s1[i]) for i,j,*n in reread_edges}
            assert expected_again == edges_again
            assert bundle['prototype']['space_group'] == again['prototype']['space_group']
            # Geometry ablation: another allowed free-coordinate choice keeps the
            # symbolic graph but generally changes the material geometry.
            u = np.array(geometry['free_vectors'])
            p0,z0,e0,c0 = expand(prototype,u)
            u2 = u + np.random.default_rng(17).normal(0,.013,u.shape)
            p2,z2,e2,c2 = expand(prototype,u2)
            assert e0 == e2 and c0 == c2 and np.array_equal(z0,z2)
            geometry_change = float(np.max(np.linalg.norm((p2-p0)@np.array(geometry['lattice']),axis=1)))
            free_dof = sum(n['dof'] for n in prototype['nodes'])
            assert (geometry_change > 1e-8) == (free_dof > 0)
            # Changing the lattice demonstrates non-uniqueness for fixed sites too.
            scaled = {**bundle,'geometry':{**geometry,'lattice':(np.array(geometry['lattice'])*1.01).tolist()}}
            scaled_crystal, scaled_edges = decode_bundle(scaled)
            assert abs(scaled_crystal.volume / expanded.volume - 1.01**3) < 1e-10
            assert scaled_edges == decoded_edges
            try:
                decode_bundle({'prototype':prototype})
            except ValueError as error:
                assert 'geometry' in str(error) or 'coordinates' in str(error)
            else:
                raise AssertionError('Decoder silently invented missing geometry')
            # Geometry is in site constraints + continuous values, not in edges.
            edge_free={**bundle,'prototype':{**prototype,'edge_tokens':[]}}
            edge_free_crystal, edge_free_edges=decode_bundle(edge_free)
            compare(expanded,edge_free_crystal)
            assert not edge_free_edges
            plain = {'lattice':original.lattice.matrix.tolist(),'species':list(original.atomic_numbers),'fractional':original.frac_coords.tolist()}
            graph = {**plain,'radius_A':bundle['radius_A'],'edges':[list(e) for e in sorted(original_edges)]}
            asu = {'prototype':{k:v for k,v in prototype.items() if k!='edge_tokens'},'geometry':geometry}
            (case/'prototype.json').write_bytes(compact(prototype))
            (case/'geometry.json').write_bytes(compact(geometry))
            (case/'full_graph.json').write_bytes(compact(graph))
            byte_sets = {'input_cif':src.read_bytes(),'bundle':encoded.read_bytes(),'structure_json':compact(plain),'full_graph_json':compact(graph),'asu_geometry_json':compact(asu),'prototype':compact(prototype),'geometry':compact(geometry)}
            sizes={k:{'bytes':len(v),'gzip_bytes':len(gzip.compress(v,mtime=0))} for k,v in byte_sets.items()}
            n_atoms,n_nodes,n_edges,n_tokens=len(original),len(prototype['nodes']),len(original_edges),len(prototype['edge_tokens'])
            record.update(status='pass',space_group=prototype['space_group'],hall_number=prototype['hall_number'],atoms=n_atoms,site_records=n_nodes,undirected_edges=n_edges,edge_records=n_tokens,operations=len(prototype['operations']),free_coordinate_dof=free_dof,
                maximum_coordinate_error_A=coord_err,maximum_lattice_entry_error_A=lattice_err,expansion_error_A=expansion_error,second_round_error_A=second_error,
                edge_symmetric_difference=0,site_record_reduction=1-n_nodes/n_atoms,edge_record_reduction=1-n_tokens/n_edges if n_edges else None,
                combined_record_reduction=1-(n_nodes+n_tokens)/(n_atoms+n_edges),sizes=sizes,
                prototype_only_unique_geometry=False,free_parameter_ablation_displacement_A=geometry_change,lattice_scale_ablation=1.01,missing_geometry_rejected=True,coordinates_unchanged_without_edge_records=True,
                reencoded_bytes_identical=encoded.read_bytes()==reencoded.read_bytes())
        except Exception as error:
            record.update(status='fail',error=repr(error))
            (case/'failure.txt').write_text(traceback.format_exc())
        results.append(record)
        (out/'records.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(record),flush=True)
    successful=[r for r in results if r['status']=='pass']
    versions={name:importlib.metadata.version(name) for name in ['numpy','pymatgen','spglib','scipy']}
    summary={'selected':len(selected),'passed':len(successful),'failed':len(results)-len(successful),'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'seconds':time.perf_counter()-began,'versions':versions,
        'symprec_A':args.symprec,'radius_A':4.5,'coordinate_tolerance_A':1e-6,'input_cell_preserved':True,'idealization':False,
        'scope':'Ordered fully occupied structures. Structural CIF data only; textual metadata/labels are not preserved. Edge records describe a radius graph, not chemical bonds. No learned encoder/decoder.',
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    if successful:
        summary['totals']={k:sum(r[k] for r in successful) for k in ['atoms','site_records','undirected_edges','edge_records','free_coordinate_dof']}
        summary['byte_totals']={k:{metric:sum(r['sizes'][k][metric] for r in successful) for metric in ['bytes','gzip_bytes']} for k in successful[0]['sizes']}
        summary['max_coordinate_error_A']=max(r['maximum_coordinate_error_A'] for r in successful)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    assert len(successful)==10, 'Some of the ten frozen CIF inputs failed; see records.json'


if __name__=='__main__':
    main()
