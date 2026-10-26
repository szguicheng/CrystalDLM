"""Exact site-point-group inventory audit. No held-out model evaluation."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch


def entropy(counts):
    total = sum(counts.values())
    return -sum(n / total * math.log2(n / total) for n in counts.values()) if total else 0.


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1)
    start = time.perf_counter()
    source = Path('multiplicity_candidates/data.pt')
    data = torch.load(source, map_location='cpu')
    meta_path = Path('site_point_groups/metadata.json')
    meta = json.loads(meta_path.read_text())
    positions = {p['wp_id']: p for p in meta['positions']}
    inventories = [()]
    lookup = {(): 0}
    profile_inventory, budgets = [], []
    by_group_type = defaultdict(dict)
    for p in positions.values():
        key = (p['space_group'], p['point_group_type'])
        value = (p['multiplicity'], p['dof'])
        if key in by_group_type:
            assert by_group_type[key] == value
        by_group_type[key] = value
    for g, ids, counts in zip(data['profile_sg'].tolist(), data['profile_wp_ids'].tolist(), data['profile_counts'].tolist()):
        inventory = Counter()
        atoms = dof = orbits = 0
        for wp, count in zip(ids, counts):
            if count:
                p = positions[wp]
                assert p['space_group'] == g
                inventory[p['point_group_type']] += count
                atoms += p['multiplicity'] * count
                dof += p['dof'] * count
                orbits += count
        key = tuple(sorted(inventory.items()))
        if key not in lookup:
            lookup[key] = len(inventories)
            inventories.append(key)
        profile_inventory.append(lookup[key])
        assert sum(by_group_type[g, h][0] * n for h, n in key) == atoms
        assert sum(by_group_type[g, h][1] * n for h, n in key) == dof
        budgets.append((atoms, dof, orbits))
    stats, examples = [], []
    residual = torch.zeros(len(profile_inventory), dtype=torch.long)
    mappings = []
    for ci, candidates in enumerate(data['context_candidates'].tolist()):
        ids = [p for p in candidates if p >= 0]
        by_inventory, by_mu = defaultdict(list), defaultdict(list)
        for p in ids:
            by_inventory[profile_inventory[p]].append(p)
            by_mu[int(data['profile_partition_id'][p])].append(p)
        assert all(len({int(data['profile_partition_id'][p]) for p in members}) == 1 for members in by_inventory.values())
        for members in by_inventory.values():
            for rank, p in enumerate(members):
                residual[p] = rank
                assert members[int(residual[p])] == p
        refined = any(len({profile_inventory[p] for p in members}) > 1 for members in by_mu.values())
        geometry = any(len({budgets[p][1] for p in members}) > 1 for members in by_mu.values())
        stats.append({'context': ci, 'G': data['contexts'][ci][0], 'N': data['contexts'][ci][1],
                      'profiles': len(ids), 'mu_inventories': len(by_mu), 'site_inventories': len(by_inventory),
                      'site_refines_mu': refined, 'mu_hides_dof': geometry,
                      'max_residual_profiles': max(map(len, by_inventory.values()))})
        mappings.append({str(k): v for k, v in by_inventory.items()})
        if geometry and len(examples) < 12:
            for members in by_mu.values():
                if len({budgets[p][1] for p in members}) > 1:
                    chosen = [members[0], next(p for p in members if budgets[p][1] != budgets[members[0]][1])]
                    examples.append({'G': data['contexts'][ci][0], 'N': data['contexts'][ci][1], 'profiles': [
                        {'id': p, 'wp_counts': [(positions[w]['letter'], n) for w, n in zip(data['profile_wp_ids'][p].tolist(), data['profile_counts'][p].tolist()) if n],
                         'site_inventory': inventories[profile_inventory[p]], 'coordinate_dof': budgets[p][1]} for p in chosen]})
                    break
    # Only training labels enter entropy/coverage analysis; replication is untouched.
    freq = {kind: defaultdict(Counter) for kind in ['profile', 'mu', 'site']}
    total = refined = geometry = 0
    for r in data['splits']['train']:
        for z, ci, p in zip(r['Z'], r['context'], r['profile_id']):
            # ci identifies G,N directly.
            key = (ci, z)
            freq['profile'][key][p] += 1
            freq['mu'][key][int(data['profile_partition_id'][p])] += 1
            freq['site'][key][profile_inventory[p]] += 1
            total += 1
            refined += stats[ci]['site_refines_mu']
            geometry += stats[ci]['mu_hides_dof']
    entropies = {kind: sum(sum(c.values()) * entropy(c) for c in groups.values()) / total for kind, groups in freq.items()}
    output = Path('site_inventories')
    output.mkdir(exist_ok=True)
    summary = {'profiles': len(profile_inventory), 'contexts': len(stats),
               'unique_site_inventories_excluding_pad': len(inventories) - 1,
               'max_site_inventories_per_context': max(s['site_inventories'] for s in stats),
               'contexts_site_refines_mu': sum(s['site_refines_mu'] for s in stats),
               'contexts_mu_hides_coordinate_dof': sum(s['mu_hides_dof'] for s in stats),
               'train_elements': total, 'train_elements_in_refined_contexts': refined,
               'train_elements_in_dof_ambiguous_mu_contexts': geometry,
               'train_empirical_entropy_bits_given_context_element': entropies,
               'site_inventory_plus_residual_roundtrip': len(profile_inventory),
               'site_inventory_determines_atom_orbit_dof_budget': 'PASS',
               'scope': 'Complete candidates within prior cap256 G,N contexts; train labels only',
               'novelty_and_model_benefit': 'NOT ESTABLISHED',
               'job_id': os.environ['SLURM_JOB_ID'], 'run_id': os.environ.get('CRYSTALDLM_RUN_ID'),
               'elapsed_seconds': time.perf_counter() - start,
               'source_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    torch.save({'profile_inventory_id': torch.tensor(profile_inventory), 'profile_residual_rank': residual,
                'profile_budgets': torch.tensor(budgets)}, output / 'indices.pt')
    for name, content in [('summary', summary), ('definitions', inventories), ('contexts', stats), ('residual_members', mappings), ('examples', examples)]:
        (output / f'{name}.json').write_text(json.dumps(content, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
