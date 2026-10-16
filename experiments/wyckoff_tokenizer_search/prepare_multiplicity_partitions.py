"""Audit a crystallographic multiplicity-partition / WP-residual factorization.

All candidates are theoretical and complete within the existing retained G,N
contexts. No validation/test frequencies are used to define the representation.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import spglib
import torch
from torch.nn import functional as F


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1)
    start = time.perf_counter()
    source = Path('candidates_cap256/data.pt')
    data = torch.load(source, map_location='cpu')
    centering = {}
    symbols = {}
    factors = {'P': 1, 'A': 2, 'B': 2, 'C': 2, 'I': 2, 'F': 4, 'R': 3}
    for hall in range(1, 531):
        kind = spglib.get_spacegroup_type(hall)
        if kind.number not in centering:
            symbols[kind.number] = kind.international_short
            centering[kind.number] = factors[kind.international_short[0]]
    assert len(centering) == 230
    partitions, lookup, profile_partition = [()], {(): 0}, []
    for g, counts, wpids in zip(data['profile_sg'].tolist(), data['profile_counts'].tolist(), data['profile_wp_ids'].tolist()):
        c = centering[g]
        histogram = Counter()
        for wp, n in zip(wpids, counts):
            if wp:
                multiplicity = int(data['wp_metadata'][wp, 0])
                assert multiplicity % c == 0
                primitive = multiplicity // c
                assert primitive <= 48
                if n:
                    histogram[primitive] += n
        key = tuple(sorted(histogram.items()))
        if key not in lookup:
            lookup[key] = len(partitions)
            partitions.append(key)
        profile_partition.append(lookup[key])
    histograms = torch.zeros(len(partitions), 49, dtype=torch.long)
    for i, value in enumerate(partitions):
        for m, count in value:
            histograms[i, m] = count
    partition_ids = torch.tensor(profile_partition)
    contexts, local_maps = [], []
    residual_rank = torch.zeros(len(profile_partition), dtype=torch.long)
    context_stats = []
    for context, candidates in enumerate(data['context_candidates']):
        ids = candidates[candidates >= 0]
        pids = partition_ids[ids]
        available = sorted(set(pids.tolist()))
        local_lookup = {pid: i for i, pid in enumerate(available)}
        local = torch.tensor([local_lookup[p] for p in pids.tolist()])
        contexts.append(torch.tensor(available))
        local_maps.append(F.pad(local, (0, len(candidates) - len(local)), value=-1))
        group, atoms = data['contexts'][context]
        for pid in available:
            assert sum(m * n for m, n in partitions[pid]) * centering[group] == atoms
            members = ids[pids == pid]
            residual_rank[members] = torch.arange(len(members))
            for r, original in enumerate(members.tolist()):
                assert int(members[int(residual_rank[original])]) == original
        context_stats.append({'context': context, 'sg': group, 'N_conventional': atoms,
                              'N_primitive': atoms // centering[group], 'profiles': len(ids),
                              'partitions': len(available),
                              'largest_residual_class': max(Counter(pids.tolist()).values())})
    maximum = max(map(len, contexts))
    data['primitive_partition_counts'] = histograms
    data['profile_partition_id'] = partition_ids
    data['profile_residual_rank'] = residual_rank
    data['context_partition_ids'] = torch.stack([F.pad(x, (0, maximum - len(x)), value=-1) for x in contexts])
    data['candidate_partition_local'] = torch.stack(local_maps)
    data['centering_factors'] = torch.tensor([0] + [centering[g] for g in range(1, 231)])
    stats = {}
    for split, records in data['splits'].items():
        total = ambiguous = parent_ambiguous = changed = 0
        for r in records:
            structure_has_parent_choice = False
            for context in r['context']:
                entry = context_stats[context]
                total += 1
                ambiguous += entry['profiles'] > 1
                parent_ambiguous += entry['partitions'] > 1
                structure_has_parent_choice |= entry['partitions'] > 1
            changed += structure_has_parent_choice
        stats[split] = {'structures': len(records), 'elements': total,
                        'ambiguous_profile_elements': ambiguous,
                        'ambiguous_partition_elements': parent_ambiguous,
                        'structures_with_partition_choice': changed}
    output = Path('multiplicity_candidates')
    output.mkdir(exist_ok=True)
    torch.save(data, output / 'data.pt')
    summary = {'unique_primitive_partitions_excluding_pad': len(partitions) - 1,
               'total_profiles': len(partition_ids), 'contexts': len(contexts),
               'maximum_partitions_per_context': maximum, 'statistics': stats,
               'centering_normalization_check': 'PASS for every tabulated WP used by all candidates',
               'partition_plus_residual_candidate_roundtrip': len(partition_ids),
               'fit_uses_label_frequencies': False, 'model_utility_not_tested': True,
               'candidate_cap256_scope': True, 'job_id': os.environ['SLURM_JOB_ID'],
               'elapsed_seconds': time.perf_counter() - start,
               'source_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'context_statistics.json').write_text(json.dumps(context_stats, indent=2))
    (output / 'partition_definitions.json').write_text(json.dumps(partitions, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
