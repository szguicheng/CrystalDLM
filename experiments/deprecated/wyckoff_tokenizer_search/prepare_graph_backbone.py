"""Prepare independent primitive and orbit graph paths for backbone parity."""
import json
import os
from pathlib import Path
import time
import numpy as np
import spglib
import torch
from pymatgen.core import Structure
from symmetry_graph_codec import decode, wrap, edge_key, action_tables, transform


def one(encoded):
    x, z, _ = decode(encoded)
    lattice = np.asarray(encoded['header']['lattice'])
    lattice, x, z = spglib.standardize_cell((lattice, x, z), to_primitive=True,
                                           no_idealize=True, symprec=1e-5)
    x = wrap(x)
    sym = spglib.get_symmetry_dataset((lattice, x, z), symprec=1e-5)
    assert sym.number == encoded['header']['space_group']
    reps = sorted(set(sym.equivalent_atoms.tolist()))
    oi = np.array([reps.index(i) for i in sym.equivalent_atoms])
    q = x[reps]
    rotations, translations = sym.rotations, sym.translations
    perm, offsets = action_tables(x, z, rotations, translations)
    choices = [int(np.flatnonzero(perm[:, reps[oi[j]]] == j)[0]) for j in range(len(x))]
    a = rotations[choices]
    b = np.array([translations[g] - offsets[g, reps[oi[j]]] for j, g in enumerate(choices)])
    assert np.allclose(np.einsum('nij,nj->ni', a, q[oi]) + b, x, atol=1e-6)
    projection = np.array([rotations[perm[:, r] == r].mean(0) for r in reps])
    assert np.allclose(projection @ projection, projection, atol=1e-8)
    crystal = Structure(lattice, z, x)
    src, dst, images, distances = crystal.get_neighbor_list(4.5, numerical_tol=1e-8)
    edges = {edge_key(i, j, n) for i, j, n, d in zip(src, dst, images, distances) if d > 1e-8}
    directed = [(i, j, *n) for i, j, *n in sorted(edges)]
    directed += [(j, i, *(-np.array(n))) for i, j, *n in sorted(edges)]
    coarse = []
    remaining = set(edges)
    while remaining:
        e = min(remaining)
        orbit = {transform(e, g, rotations, perm, offsets) for g in range(len(rotations))}
        assert orbit <= edges
        remaining.difference_update(orbit)
        arcs = [(i, j, np.array(n)) for i, j, *n in orbit]
        arcs += [(j, i, -np.array(n)) for i, j, *n in orbit]
        for source_orbit in sorted({int(oi[e[0]]), int(oi[e[1]])}):
            group = [(i, j, n) for i, j, n in arcs if i == reps[source_orbit]]
            assert group
            targets = {int(oi[j]) for i, j, n in group}
            assert len(targets) == 1
            _, j, n = group[0]
            coarse.append({'src': source_orbit, 'dst': targets.pop(), 'A': a[j], 'b': b[j] + n,
                           'sumA': sum(a[jj] for _, jj, _ in group),
                           'sumb': sum(b[jj] + nn for _, jj, nn in group), 'count': len(group)})
    assert sum(c['count'] * int((oi == c['src']).sum()) for c in coarse) == len(directed)
    return dict(lattice=lattice, q=q, projection=projection, Z=z[reps], orbit=oi,
                A=a, b=b, reps=reps, multiplicity=np.bincount(oi), edges=directed, coarse=coarse)


def pack(records):
    out = {k: [] for k in ['lattice','q','projection','Z','orbit','A','b','reps','multiplicity',
                           'asu_graph','full_graph','src','dst','image','cs','ct','cA','cb','sumA','sumb','count']}
    ns = nf = 0
    for gi, r in enumerate(records):
        out['lattice'].append(r['lattice'])
        for k in ['q','projection','Z','A','b','multiplicity']:
            out[k].extend(r[k])
        out['orbit'].extend(r['orbit'] + ns)
        out['reps'].extend(np.array(r['reps']) + nf)
        out['asu_graph'].extend([gi] * len(r['q']))
        out['full_graph'].extend([gi] * len(r['orbit']))
        for i, j, *n in r['edges']:
            out['src'].append(i + nf); out['dst'].append(j + nf); out['image'].append(n)
        for c in r['coarse']:
            out['cs'].append(c['src'] + ns); out['ct'].append(c['dst'] + ns)
            for k, origin in [('cA','A'),('cb','b'),('sumA','sumA'),('sumb','sumb'),('count','count')]:
                out[k].append(c[origin])
        ns += len(r['q']); nf += len(r['orbit'])
    integers = {'Z','orbit','reps','asu_graph','full_graph','src','dst','cs','ct'}
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.long if k in integers else torch.float64)
            for k, v in out.items()}


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    folder = Path('symmetry_graph_audit/train_512')
    rows = json.loads((folder / 'records.json').read_text())
    prepared = []
    for i, row in enumerate(rows):
        prepared.append(one(json.loads((folder / f"{row['material_id']}.json").read_text())))
        if (i + 1) % 64 == 0:
            print(json.dumps({'prepared': i + 1, 'seconds': time.perf_counter() - start}), flush=True)
    out = Path('graph_backbone'); out.mkdir(exist_ok=True)
    torch.save({'all': pack(prepared), 'small': pack(prepared[:32]),
                'ids': [r['material_id'] for r in rows]}, out / 'data.pt')
    print(json.dumps({'status': 'COMPLETED','structures': len(prepared),'job_id': os.environ['SLURM_JOB_ID'],
                      'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    main()
