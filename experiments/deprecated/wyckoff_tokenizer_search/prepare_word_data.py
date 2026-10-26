"""Build shared word definitions and three reversible candidate decompositions."""
import json
import os
import time
from pathlib import Path

import torch

from learn_set_merges import encode


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    data = torch.load('candidates_cap256/data.pt', map_location='cpu')
    vocabulary = json.loads(Path('set_merge_cap256_audit/vocabulary.json').read_text())
    assert vocabulary['fit_structures'] == len(data['splits']['train'])
    payloads = {int(k): tuple(v) for k, v in vocabulary['payload'].items()}
    definitions = [()]
    reverse = {(): 0}
    def register(payload):
        payload = tuple(payload)
        if payload not in reverse:
            reverse[payload] = len(definitions)
            definitions.append(payload)
        return reverse[payload]
    for i in sorted(payloads):
        register(payloads[i])
    wpids_by_group = {}
    for g, row in zip(data['profile_sg'].tolist(), data['profile_wp_ids'].tolist()):
        wpids_by_group[g] = row
    # Word definitions outside observed G,N still come from the same fixed table.
    import sys
    sys.path.insert(0, '/scratch/chenggui2/CrystalDLM/gate1_20260908')
    from count_vocab import read_groups
    groups = read_groups(Path('/scratch/chenggui2/CrystalDLM/gate1_20260908/input'))
    idx = 1
    for g, group in groups.items():
        ids = list(range(idx, idx + len(group['positions'])))
        idx += len(ids)
        wpids_by_group[g] = ids + [0] * (27 - len(ids))
    representations = {name: [] for name in ['unigram', 'rle', 'setmerge']}
    for counts, wpids in zip(data['profile_counts'].tolist(), data['profile_wp_ids'].tolist()):
        raw = tuple(sorted(w for w, n in zip(wpids, counts) for _ in range(n)))
        unigram = [register((w,)) for w in raw]
        rle = [register((w,) * n) for w, n in zip(wpids, counts) if n]
        merged = [register(payloads[w]) for w in encode(raw, vocabulary['rules'])]
        for name, words in [('unigram', unigram), ('rle', rle), ('setmerge', merged)]:
            assert tuple(sorted(w for word in words for w in definitions[word])) == raw
            representations[name].append(words)
    word_counts, word_wp_ids = [], []
    from collections import Counter
    for payload in definitions:
        if not payload:
            word_counts.append([0] * 27); word_wp_ids.append([0] * 27)
            continue
        g = vocabulary['base'][str(payload[0])]['sg']
        ids = wpids_by_group[g]
        counts = Counter(payload)
        word_counts.append([counts[w] for w in ids])
        word_wp_ids.append(ids)
    data['word_counts'] = torch.tensor(word_counts, dtype=torch.long)
    data['word_wp_ids'] = torch.tensor(word_wp_ids, dtype=torch.long)
    data['profile_words'] = {}
    for name, rows in representations.items():
        longest = max(map(len, rows))
        data['profile_words'][name] = torch.tensor([r + [0] * (longest - len(r)) for r in rows], dtype=torch.long)
    data['word_vocabulary_fit'] = {'training_structures': vocabulary['fit_structures'],
                                  'learned_merges': len(vocabulary['rules']),
                                  'uses_validation_or_test_frequencies': False}
    output = Path('word_candidates')
    output.mkdir(exist_ok=True)
    torch.save(data, output / 'data.pt')
    summary = {'word_definitions': len(definitions), 'profile_candidates': len(representations['rle']),
               'max_words_per_candidate': {k: max(map(len, v)) for k, v in representations.items()},
               'all_candidate_roundtrips': 'PASS', 'fit': data['word_vocabulary_fit'],
               'elapsed_seconds': time.perf_counter() - start, 'job_id': os.environ['SLURM_JOB_ID']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
