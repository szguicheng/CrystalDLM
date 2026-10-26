"""Spatial edge-orbit codec with explicit ASU geometry and operation table.

Labels (source orbit, target orbit, operation id, integer image) follow the
operation+translation semantics used for CIF bond endpoints. This prototype
requires continuous ASU coordinates; it is not a coordinate-free graph decoder.
"""
import numpy as np


def wrap(x):
    x = np.asarray(x, dtype=float) % 1.
    x[np.abs(x - 1.) < 1e-9] = 0.
    x[np.abs(x) < 1e-9] = 0.
    return x


def edge_key(i, j, shift):
    n = tuple(int(v) for v in shift)
    return min((int(i), int(j), *n), (int(j), int(i), *(-v for v in n)))


def match_sites(query, types, positions, species):
    delta = np.asarray(query)[:, None, :] - positions[None, :, :]
    residual = delta - np.rint(delta)
    matching = (np.abs(residual).max(-1) < 1e-6) & (np.asarray(types)[:, None] == species[None, :])
    assert np.all(matching.sum(-1) == 1), 'site matching is not unique'
    ids = matching.argmax(-1)
    shifts = np.rint(np.asarray(query) - positions[ids]).astype(int)
    return ids, shifts


def action_tables(positions, species, rotations, translations):
    permutations, offsets = [], []
    for r, t in zip(rotations, translations):
        ids, shifts = match_sites(positions @ r.T + t, species, positions, species)
        assert len(set(ids.tolist())) == len(positions), 'operation is not a permutation'
        permutations.append(ids)
        offsets.append(shifts)
    return np.asarray(permutations), np.asarray(offsets)


def transform(edge, operation, rotations, permutations, offsets):
    i, j, *n = edge
    shift = offsets[operation, j] + rotations[operation] @ n - offsets[operation, i]
    return edge_key(permutations[operation, i], permutations[operation, j], shift)


def encode(positions, species, rotations, translations, equivalent_atoms, letters, edges, lattice, sg, hall):
    positions = wrap(positions)
    representatives = sorted(set(int(i) for i in equivalent_atoms))
    orbit_lookup = {rep: o for o, rep in enumerate(representatives)}
    orbit_ids = np.array([orbit_lookup[int(i)] for i in equivalent_atoms])
    permutations, offsets = action_tables(positions, species, rotations, translations)
    operation_to_atom = [int(np.flatnonzero(permutations[:, representatives[orbit_ids[j]]] == j)[0])
                         for j in range(len(positions))]
    remaining, tokens = set(edges), []
    edge_orbit_sizes = []
    while remaining:
        representative_edge = min(remaining)
        orbit = {transform(representative_edge, g, rotations, permutations, offsets) for g in range(len(rotations))}
        assert orbit <= edges, 'neighbor graph is not closed under symmetry'
        remaining.difference_update(orbit)
        candidates = []
        i, j, *n = representative_edge
        for source, target, image in [(i, j, np.asarray(n)), (j, i, -np.asarray(n))]:
            oi, oj = int(orbit_ids[source]), int(orbit_ids[target])
            for g in np.flatnonzero(permutations[:, source] == representatives[oi]):
                target_atom = int(permutations[g, target])
                image_after = offsets[g, target] + rotations[g] @ image - offsets[g, source]
                h = operation_to_atom[target_atom]
                translation = image_after - offsets[h, representatives[oj]]
                candidates.append((oi, oj, h, *(int(x) for x in translation)))
        tokens.append(min(candidates))
        edge_orbit_sizes.append(len(orbit))
    assert len(set(tokens)) == len(tokens), 'edge orbit token collision'
    header = {'space_group': int(sg), 'hall_number': int(hall), 'lattice': np.asarray(lattice).tolist(),
              'rotations': rotations.tolist(), 'translations': translations.tolist(),
              'nodes': [{'Z': int(species[i]), 'wyckoff_letter': letters[i], 'fractional': positions[i].tolist()}
                        for i in representatives]}
    return {'header': header, 'edge_tokens': sorted(tokens)}, edge_orbit_sizes


def decode(encoded):
    header = encoded['header']
    rotations = np.asarray(header['rotations'], dtype=int)
    translations = np.asarray(header['translations'], dtype=float)
    positions, species, representatives = [], [], []
    for node in header['nodes']:
        q = np.asarray(node['fractional'])
        start = len(positions)
        for r, t in zip(rotations, translations):
            x = wrap(r @ q + t)
            if any(np.max(np.abs((x - p) - np.rint(x - p))) < 1e-6 for p in positions[start:]):
                continue
            positions.append(x)
            species.append(node['Z'])
        ids, _ = match_sites([q], [node['Z']], np.asarray(positions), np.asarray(species))
        representatives.append(int(ids[0]))
    positions, species = np.asarray(positions), np.asarray(species)
    permutations, offsets = action_tables(positions, species, rotations, translations)
    edges = set()
    for oi, oj, operation, *translation in encoded['edge_tokens']:
        source = representatives[oi]
        target = permutations[operation, representatives[oj]]
        image = offsets[operation, representatives[oj]] + translation
        edge = edge_key(source, target, image)
        edges.update(transform(edge, g, rotations, permutations, offsets) for g in range(len(rotations)))
    return positions, species, edges
