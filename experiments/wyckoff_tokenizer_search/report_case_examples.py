"""Descriptive, pre-specified selection of labelled-protostructure failures."""
import json
import os
from pathlib import Path

import torch
from pymatgen.core import Element


def main():
    assert os.environ.get('SLURM_JOB_ID')
    folder = Path('frozen_word_evaluation/test')
    seeds = [17, 29, 43]
    predictions = {}
    for seed in seeds:
        for variant in ['rle', 'rlemerge']:
            predictions[seed, variant] = {r['id']: r for r in
                map(json.loads, (folder / f'{variant}_seed{seed}.jsonl').read_text().splitlines())}
    data = torch.load('count_merge_candidates/data.pt', map_location='cpu')
    records = {r['id']: r for r in data['splits']['test']}
    base = json.loads(Path('set_merge_cap256_audit/vocabulary.json').read_text())['base']
    def profile(pid):
        return [{'wp': base[str(wp)]['letter'], 'orbit_count': count,
                 'multiplicity': base[str(wp)]['multiplicity']} for wp, count in
                zip(data['profile_wp_ids'][pid].tolist(), data['profile_counts'][pid].tolist()) if count]
    groups = {'rle_correct_all_seeds_block_wrong_all_seeds': [],
              'block_correct_all_seeds_rle_wrong_all_seeds': [],
              'both_wrong_all_seeds': []}
    for material in sorted(records):
        a = [predictions[s, 'rle'][material]['exact'] for s in seeds]
        b = [predictions[s, 'rlemerge'][material]['exact'] for s in seeds]
        name = None
        if all(a) and not any(b):
            name = 'rle_correct_all_seeds_block_wrong_all_seeds'
        elif all(b) and not any(a):
            name = 'block_correct_all_seeds_rle_wrong_all_seeds'
        elif not any(a) and not any(b):
            name = 'both_wrong_all_seeds'
        if name:
            groups[name].append(material)
    report = {}
    for name, ids in groups.items():
        examples = []
        for material in ids[:3]:
            record = records[material]
            examples.append({'material_id': material, 'space_group': record['sg'],
                             'composition_conventional': {Element.from_Z(z).symbol: n for z, n in zip(record['Z'], record['N'])},
                             'true_profiles': [profile(pid) for pid in record['profile_id']],
                             'predictions': {f'{v}_seed{s}': [profile(pid) for pid in predictions[s, v][material]['predicted_profile_ids']]
                                             for s in seeds for v in ['rle', 'rlemerge']}})
        report[name] = {'count': len(ids), 'selection': 'First three ascending material IDs in the stated class', 'examples': examples}
    output = Path('final_report')
    output.mkdir(exist_ok=True)
    (output / 'case_examples.json').write_text(json.dumps({'classes': report,
        'errors_are_labelled_protostructure_mismatches_not_physical_instability': True,
        'no_causal_failure_explanation_inferred': True, 'job_id': os.environ['SLURM_JOB_ID']}, indent=2))
    print(json.dumps({'case_class_counts': {k: v['count'] for k, v in report.items()}, 'job_id': os.environ['SLURM_JOB_ID']}), flush=True)


if __name__ == '__main__':
    main()
