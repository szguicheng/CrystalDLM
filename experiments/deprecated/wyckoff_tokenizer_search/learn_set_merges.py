"""Train-only unordered same-element Wyckoff merges; lossless primitive fallback."""
import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/scratch/chenggui2/CrystalDLM/gate1_20260908')
from count_vocab import read_groups


def apply_rule(bag, a, b, c):
    count = bag[a] // 2 if a == b else min(bag[a], bag[b])
    if count:
        bag[a] -= count
        bag[b] -= count
        bag[c] += count
        for old in {a, b}:
            if bag[old] == 0:
                del bag[old]


def encode(primitives, rules):
    bag = Counter(primitives)
    for a, b, c in rules:
        apply_rule(bag, a, b, c)
    return tuple(sorted(bag.elements()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merges', type=int, default=256)
    parser.add_argument('--min-support', type=int, default=20)
    parser.add_argument('--max-orbits', type=int, default=4)
    parser.add_argument('--fit-candidate-cap', type=int)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    groups = read_groups(Path('/scratch/chenggui2/CrystalDLM/gate1_20260908/input'))
    base = {}
    info = {}
    for g, group in groups.items():
        for wp in group['positions']:
            idx = len(base) + 1
            base[(g, wp['letter'])] = idx
            info[idx] = {'sg': g, 'letter': wp['letter'], 'multiplicity': wp['multiplicity'], 'dof': wp['dof']}
    payload = {idx: (idx,) for idx in info}
    reverse = {value: key for key, value in payload.items()}
    data = {split: [json.loads(line) for line in Path(f'processed/{split}.jsonl').open()]
            for split in ['train', 'val', 'test']}
    def primitives(record, element):
        return tuple(sorted(base[(record['space_group'], letter)]
                            for letter, count in element['profile'] for _ in range(count)))
    fit_records = [r for r in data['train'] if args.fit_candidate_cap is None or
                   all(e['candidate_count'] <= args.fit_candidate_cap for e in r['elements'])]
    frequencies = Counter(primitives(r, e) for r in fit_records for e in r['elements'])
    bags = [(Counter(key), weight) for key, weight in frequencies.items()]
    rules, support = [], []
    for _ in range(args.merges):
        pairs = Counter()
        for bag, weight in bags:
            ids = sorted(bag)
            for i, a in enumerate(ids):
                for b in ids[i:]:
                    available = bag[a] // 2 if a == b else min(bag[a], bag[b])
                    if not available or len(payload[a]) + len(payload[b]) > args.max_orbits:
                        continue
                    key = tuple(sorted(payload[a] + payload[b]))
                    if key in reverse:
                        continue
                    pairs[(a, b)] += weight * available
        if not pairs:
            break
        pair = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[pair] < args.min_support:
            break
        a, b = pair
        c = len(payload) + 1
        payload[c] = tuple(sorted(payload[a] + payload[b]))
        assert len({info[x]['sg'] for x in payload[c]}) == 1
        reverse[payload[c]] = c
        rules.append((a, b, c)); support.append(pairs[pair])
        for bag, _ in bags:
            apply_rule(bag, a, b, c)
    output = Path('set_merge_audit' if args.fit_candidate_cap is None else f'set_merge_cap{args.fit_candidate_cap}_audit')
    output.mkdir(exist_ok=True)
    evaluations = {}
    for budget in sorted({0, min(64, len(rules)), min(128, len(rules)), len(rules)}):
        chosen = rules[:budget]
        cache = {}
        statistics = {}
        for split, records in data.items():
            before = after = rle = changed = bags_count = 0
            for r in records:
                for e in r['elements']:
                    raw = primitives(r, e)
                    if raw not in cache:
                        tokens = encode(raw, chosen)
                        decoded = tuple(sorted(x for t in tokens for x in payload[t]))
                        assert decoded == raw
                        assert encode(list(reversed(raw)), chosen) == tokens
                        cache[raw] = tokens
                    tokens = cache[raw]
                    before += len(raw); after += len(tokens); rle += len(set(raw))
                    changed += len(tokens) < len(raw); bags_count += 1
            statistics[split] = {'structures': len(records), 'element_bags': bags_count,
                                 'orbit_units': before, 'set_merge_units': after,
                                 'run_length_WP_count_units': rle,
                                 'reduction_vs_orbit_units': 1 - after / before,
                                 'reduction_vs_run_length_units': 1 - after / rle,
                                 'compressed_element_bags': changed,
                                 'roundtrip_and_reversed_order_checks': 'PASS'}
        evaluations[budget] = statistics
    vocab = {'base': {str(k): v for k, v in info.items()},
             'payload': {str(k): list(v) for k, v in payload.items()},
             'rules': rules, 'support': support,
             'fit_split': 'train only', 'same_element_only': True,
             'fit_candidate_cap': args.fit_candidate_cap, 'fit_structures': len(fit_records),
             'max_orbits_per_token': args.max_orbits, 'minimum_train_support': args.min_support,
             'base_vocab_complete_for_230_groups': True,
             'handles_equivalent_origin_choices': False}
    (output / 'vocabulary.json').write_text(json.dumps(vocab, indent=2))
    top = []
    for rule, count in zip(rules, support):
        c = rule[2]
        top.append({'code': c, 'space_group': info[payload[c][0]]['sg'],
                    'wyckoff_letters': [info[x]['letter'] for x in payload[c]],
                    'atom_count': sum(info[x]['multiplicity'] for x in payload[c]),
                    'support_at_merge': count, 'only_repeated_same_WP': len(set(payload[c])) == 1})
    summary = {'learned_merges': len(rules), 'train_unique_element_profiles': len(frequencies),
               'evaluations': evaluations, 'merges': top,
               'elapsed_seconds': time.perf_counter() - start, 'job_id': os.environ['SLURM_JOB_ID'],
               'host': socket.gethostname(), 'model_performance_not_evaluated': True,
               'unit_counts_are_not_bit_rates_or_GPU_speed': True,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'merges'}), flush=True)


if __name__ == '__main__':
    main()
