"""Learn disjoint WP-count block merges, keeping RLE as the exact fallback.

Unlike orbit-level merges, this can never increase a profile's RLE word count.
All learning uses only the shared model training subset.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F


def encode(sequence, rules):
    bag = set(sequence)
    assert len(bag) == len(sequence), 'RLE has one record per distinct WP.'
    for a, b, c in rules:
        if a in bag and b in bag:
            bag.remove(a); bag.remove(b); bag.add(c)
    return tuple(sorted(bag))


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    source = Path('word_candidates/data.pt')
    data = torch.load(source, map_location='cpu')
    counts = data['word_counts'].tolist()
    wpids = data['word_wp_ids'].tolist()
    payload = [tuple((w, n) for w, n in zip(ws, ns) if n) for ws, ns in zip(wpids, counts)]
    reverse = {v: k for k, v in enumerate(payload)}
    assert len(reverse) == len(payload)
    original_size = len(payload)
    initial = [tuple(w for w in row if w) for row in data['profile_words']['rle'].tolist()]
    frequencies = Counter(pid for r in data['splits']['train'] for pid in r['profile_id'])
    bags = [(set(initial[pid]), frequency) for pid, frequency in frequencies.items()]
    rules, supports = [], []
    for _ in range(256):
        pairs = Counter()
        for bag, frequency in bags:
            ids = sorted(bag)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if len(payload[a]) + len(payload[b]) > 3:
                        continue
                    pairs[a, b] += frequency
        if not pairs:
            break
        a, b = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[a, b] < 20:
            break
        merged = tuple(sorted(payload[a] + payload[b]))
        assert len({w for w, n in merged}) == len(merged)
        if merged in reverse:
            c = reverse[merged]
        else:
            c = len(payload)
            reverse[merged] = c
            payload.append(merged)
            d = dict(merged)
            wpids.append(wpids[a])
            counts.append([d.get(w, 0) for w in wpids[a]])
            assert tuple((w, n) for w, n in zip(wpids[-1], counts[-1]) if n) == merged
        rules.append((a, b, c)); supports.append(pairs[a, b])
        for bag, _ in bags:
            if a in bag and b in bag:
                bag.remove(a); bag.remove(b); bag.add(c)
    encoded = []
    for raw in initial:
        words = encode(raw, rules)
        decoded = tuple(sorted(item for w in words for item in payload[w]))
        expected = tuple(sorted(item for w in raw for item in payload[w]))
        assert decoded == expected
        assert encode(tuple(reversed(raw)), rules) == words
        assert len(words) <= len(raw)
        encoded.append(words)
    longest = max(map(len, encoded))
    data['profile_words']['rlemerge'] = torch.tensor([list(r) + [0] * (longest - len(r)) for r in encoded])
    data['word_counts'] = torch.tensor(counts)
    data['word_wp_ids'] = torch.tensor(wpids)
    statistics = {}
    old_lengths = (data['profile_words']['setmerge'] > 0).sum(-1).tolist()
    for split, records in data['splits'].items():
        ids = [pid for r in records for pid in r['profile_id']]
        before = sum(len(initial[pid]) for pid in ids)
        after = sum(len(encoded[pid]) for pid in ids)
        statistics[split] = {'structures': len(records), 'elements': len(ids), 'rle_words': before,
                             'count_block_words': after, 'reduction_vs_rle': 1 - after / before,
                             'old_setmerge_longer_than_rle_elements': sum(old_lengths[pid] > len(initial[pid]) for pid in ids),
                             'new_longer_than_rle_elements': sum(len(encoded[pid]) > len(initial[pid]) for pid in ids),
                             'changed_elements': sum(len(encoded[pid]) < len(initial[pid]) for pid in ids)}
    output = Path('count_merge_candidates')
    output.mkdir(exist_ok=True)
    torch.save(data, output / 'data.pt')
    vocab = {'rules': rules, 'train_support': supports, 'payload': payload, 'max_distinct_wps': 3,
             'fit_training_structures': len(data['splits']['train']), 'fit_uses_validation_or_test_frequencies': False,
             'base_is_rle': True, 'fallback_never_longer_than_rle': True}
    (output / 'vocabulary.json').write_text(json.dumps(vocab, indent=2))
    summary = {'learned_rules': len(rules), 'new_word_definitions': len(payload) - original_size,
               'all_candidate_roundtrips_and_reverse_order': 'PASS', 'candidate_count': len(encoded),
               'statistics': statistics, 'max_words_per_profile': longest,
               'job_id': os.environ['SLURM_JOB_ID'], 'elapsed_seconds': time.perf_counter() - start,
               'source_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
