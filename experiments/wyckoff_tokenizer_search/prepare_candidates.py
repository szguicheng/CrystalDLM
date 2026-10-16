"""Prepare a complete contextual profile vocabulary for bounded pilot contexts."""
import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, '/scratch/chenggui2/CrystalDLM/gate1_20260908')
from count_vocab import read_groups, coefficients


def enumerate_exact(positions, total):
    @lru_cache(None)
    def ways(index, remaining):
        if index == len(positions):
            return int(remaining == 0)
        p = positions[index]
        limit = remaining // p['multiplicity']
        if p['dof'] == 0:
            limit = min(limit, 1)
        return sum(ways(index + 1, remaining - k * p['multiplicity']) for k in range(limit + 1))

    def visit(index, remaining, prefix):
        if index == len(positions):
            if remaining == 0:
                yield tuple(prefix)
            return
        p = positions[index]
        limit = remaining // p['multiplicity']
        if p['dof'] == 0:
            limit = min(limit, 1)
        for k in range(limit + 1):
            residual = remaining - k * p['multiplicity']
            if ways(index + 1, residual):
                yield from visit(index + 1, residual, prefix + [k])

    return list(visit(0, total, []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=int, default=256)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    root = Path('processed')
    output = Path(f'candidates_cap{args.cap}')
    output.mkdir(exist_ok=True)
    groups = read_groups(Path('/scratch/chenggui2/CrystalDLM/gate1_20260908/input'))
    raw = {split: [json.loads(line) for line in (root / f'{split}.jsonl').open()]
           for split in ['train', 'val', 'test']}
    all_contexts = sorted({(r['space_group'], e['N']) for data in raw.values()
                           for r in data for e in r['elements']})
    context_counts = {(g, n): coefficients(groups[g]['positions'], n)[n] for g, n in all_contexts}
    good_contexts = [c for c in all_contexts if 0 < context_counts[c] <= args.cap]
    context_id = {c: i for i, c in enumerate(good_contexts)}
    metadata = [[0, 0, 0]]
    wp_ids = {}
    for g, group in groups.items():
        wp_ids[g] = []
        for i, p in enumerate(group['positions']):
            wp_ids[g].append(len(metadata))
            metadata.append([p['multiplicity'], p['dof'], i])
    profile_counts, profile_wp_ids, profile_sg = [], [], []
    context_candidates = torch.full((len(good_contexts), args.cap), -1, dtype=torch.long)
    index_by_context = {}
    for context, index in context_id.items():
        g, n = context
        profiles = enumerate_exact(groups[g]['positions'], n)
        assert len(profiles) == context_counts[context]
        index_by_context[context] = {profile: j for j, profile in enumerate(profiles)}
        for j, profile in enumerate(profiles):
            pid = len(profile_counts)
            context_candidates[index, j] = pid
            profile_counts.append(list(profile) + [0] * (27 - len(profile)))
            profile_wp_ids.append(wp_ids[g] + [0] * (27 - len(profile)))
            profile_sg.append(g)
    prepared = {}
    analysis = {}
    train_patterns = Counter()
    for split, records in raw.items():
        examples = []
        rejected = []
        profile_hist = Counter()
        for r in records:
            sg = r['space_group']
            if any((sg, e['N']) not in context_id for e in r['elements']):
                rejected.append(r['material_id'])
                continue
            zs, ns, contexts, pids, targets = [], [], [], [], []
            for e in r['elements']:
                context = (sg, e['N'])
                counts = dict(e['profile'])
                profile = tuple(counts.get(p['letter'], 0) for p in groups[sg]['positions'])
                target = index_by_context[context][profile]
                ctx = context_id[context]
                pid = int(context_candidates[ctx, target])
                assert pid >= 0
                zs.append(e['Z']); ns.append(e['N']); contexts.append(ctx)
                pids.append(pid); targets.append(target)
                profile_hist[pid] += 1
                if split == 'train':
                    train_patterns[pid] += 1
            examples.append({'id': r['material_id'], 'sg': sg, 'Z': zs, 'N': ns,
                             'context': contexts, 'profile_id': pids, 'target': targets})
        prepared[split] = examples
        all_element_count = sum(len(r['elements']) for r in records)
        all_orbit_count = sum(len(r['orbits']) for r in records)
        analysis[split] = {'preprocessed_rows': len(records), 'retained_rows': len(examples),
                           'excluded_by_candidate_cap': len(rejected), 'excluded_ids': rejected,
                           'original_element_slots': all_element_count, 'original_orbit_slots': all_orbit_count,
                           'multi_orbit_elements': sum(e['orbit_count'] > 1 for r in records for e in r['elements']),
                           'candidate_threshold_element_coverage': {
                               str(t): sum(e['candidate_count'] <= t for r in records for e in r['elements']) / all_element_count
                               for t in [64, 256, 1024, 4096, 10000]},
                           'unique_observed_profiles_in_scope': len(profile_hist),
                           'in_scope_element_slots': sum(profile_hist.values()),
                           'unseen_positive_profile_slots_vs_train': sum(count for pid, count in profile_hist.items() if pid not in train_patterns)}
    max_elements = max(len(r['Z']) for records in prepared.values() for r in records)
    max_atoms = max(n for records in prepared.values() for r in records for n in r['N'])
    profile_counts = torch.tensor(profile_counts, dtype=torch.long)
    profile_wp_ids = torch.tensor(profile_wp_ids, dtype=torch.long)
    # Geometry-independent reconstruction checks for every allowed candidate.
    meta = torch.tensor(metadata, dtype=torch.long)
    for (g, n), ctx in context_id.items():
        ids = context_candidates[ctx]
        ids = ids[ids >= 0]
        assert torch.all((profile_counts[ids] * meta[profile_wp_ids[ids], 0]).sum(1) == n)
        fixed = meta[profile_wp_ids[ids], 1] == 0
        assert torch.all(profile_counts[ids][fixed] <= 1)
    package = {'profile_counts': profile_counts, 'profile_wp_ids': profile_wp_ids,
               'profile_sg': torch.tensor(profile_sg, dtype=torch.long),
               'wp_metadata': meta, 'context_candidates': context_candidates,
               'contexts': good_contexts, 'splits': prepared, 'max_elements': max_elements,
               'max_atoms': max_atoms, 'cap': args.cap, 'train_profile_counts': dict(train_patterns)}
    torch.save(package, output / 'data.pt')
    summary = {'scope': 'Given conventional-cell composition and space group; all element contexts must have <= cap candidates.',
               'not_full_dataset_model_coverage': True, 'candidate_cap': args.cap,
               'all_observed_contexts': len(all_contexts), 'contexts_in_scope': len(good_contexts),
               'complete_theoretical_candidates': len(profile_counts),
               'candidate_generation_uses_target_profiles': False,
               'candidate_enumeration_truncated': False,
               'candidate_vocabulary_depends_only_on_G_N_and_Wyckoff_table': True,
               'max_elements': max_elements, 'max_element_atom_count': max_atoms,
               'splits': analysis, 'host': socket.gethostname(), 'job_id': os.environ['SLURM_JOB_ID'],
               'elapsed_seconds': time.perf_counter() - start,
               'package_sha256': hashlib.sha256((output / 'data.pt').read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'splits'}), flush=True)
    for split, data in analysis.items():
        print(split, json.dumps({k: v for k, v in data.items() if k != 'excluded_ids'}), flush=True)


if __name__ == '__main__':
    main()
