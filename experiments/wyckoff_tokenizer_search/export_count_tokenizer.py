"""Export the frozen codec and audit it on all successfully processed structures."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from count_block_tokenizer import CountBlockTokenizer


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start = time.perf_counter()
    vocabulary = json.loads(Path('count_merge_candidates/vocabulary.json').read_text())
    base = json.loads(Path('set_merge_cap256_audit/vocabulary.json').read_text())['base']
    payload = vocabulary['payload']
    def fields(word):
        return tuple(sorted((base[str(wp)]['letter'], n) for wp, n in payload[word]))
    positions, order, rules = {}, {}, {}
    for info in base.values():
        g = str(info['sg'])
        positions.setdefault(g, {})[info['letter']] = {'multiplicity': info['multiplicity'], 'dof': info['dof']}
    for word, value in enumerate(payload):
        if value:
            g = str(base[str(value[0][0])]['sg'])
            order.setdefault(g, []).append({'id': word, 'records': fields(word)})
    for a, b, c in vocabulary['rules']:
        g = str(base[str(payload[c][0][0])]['sg'])
        rules.setdefault(g, []).append([fields(a), fields(b), fields(c)])
    spec = {'format': 'wyckoff-count-block-v1', 'positions': positions,
            'rules_by_group': rules, 'word_order_by_group': order,
            'fit_training_structures': vocabulary['fit_training_structures'],
            'max_distinct_wps_per_block': 3, 'learned_rule_count': len(vocabulary['rules']),
            'counts_are_independent_orbits_not_expanded_atoms': True,
            'geometry_encoded': False, 'equivalent_origin_quotient': False,
            'fallback': 'one typed (WP letter, count) record; no learned rule required'}
    output = Path('portable_tokenizer')
    output.mkdir(exist_ok=True)
    target = output / 'count_block_tokenizer.json'
    target.write_text(json.dumps(spec, indent=2))
    tokenizer = CountBlockTokenizer(target)
    statistics = {}
    for split in ['train', 'val', 'test']:
        structures = elements = before = after = 0
        for line in Path(f'processed/{split}.jsonl').open():
            record = json.loads(line)
            g = record['space_group']
            structures += 1
            for element in record['elements']:
                profile = dict(element['profile'])
                tokens = tokenizer.encode(g, profile)
                assert tokenizer.decode(tokens) == profile
                assert tokenizer.atom_count(g, tokens) == element['N']
                assert tokenizer.encode(g, dict(reversed(list(profile.items())))) == tokens
                assert len(tokens) <= len(profile)
                elements += 1; before += len(profile); after += len(tokens)
        statistics[split] = {'structures': structures, 'elements': elements, 'rle_units': before,
                             'count_block_units': after, 'reduction_vs_rle': 1 - after / before,
                             'roundtrip_atom_count_order_and_nonexpansion': 'PASS'}
    data = torch.load('count_merge_candidates/data.pt', map_location='cpu')
    for pid, g in enumerate(data['profile_sg'].tolist()):
        profile = {base[str(wp)]['letter']: n for wp, n in zip(data['profile_wp_ids'][pid].tolist(), data['profile_counts'][pid].tolist()) if n}
        actual = tokenizer.encode(g, profile)
        expected = tuple(fields(word) for word in data['profile_words']['rlemerge'][pid].tolist() if word)
        assert actual == expected, (pid, actual, expected)
    outside = tokenizer.encode(1, {'a': 37})
    assert tokenizer.decode(outside) == {'a': 37} and tokenizer.atom_count(1, outside) == 37
    summary = {'statistics': statistics, 'exact_frozen_model_token_sequence_parity': len(data['profile_sg']),
               'unseen_count_typed_fallback_check': 'PASS: SG1 a*37',
               'outside_candidate_cap_codec_supported': True,
               'outside_candidate_cap_model_not_evaluated': True,
               'vocabulary_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'job_id': os.environ['SLURM_JOB_ID'], 'elapsed_seconds': time.perf_counter() - start}
    (output / 'audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
