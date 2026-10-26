"""Fix the unused validation remainder by ID, without evaluating model outputs."""
import hashlib
import json
import os
from pathlib import Path
import torch


def main():
    assert os.environ.get('SLURM_JOB_ID')
    data_path = Path('multiplicity_candidates/data.pt')
    data = torch.load(data_path, map_location='cpu')
    val = sorted(data['splits']['val'], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    development = [r['id'] for r in val[:2048]]
    replication = [r['id'] for r in val[2048:]]
    train = {r['id'] for r in data['splits']['train']}
    original_test = {r['id'] for r in data['splits']['test']}
    assert len(set(development + replication)) == len(val)
    assert not set(development).intersection(replication)
    assert not set(replication).intersection(train | original_test)
    manifest = {'development_ids': development, 'replication_ids': replication,
                'development_count': len(development), 'replication_count': len(replication),
                'selection': 'SHA256(material_id) ascending; first2048 development, remainder replication',
                'replication_model_scores_not_read': True,
                'prior_exposure': 'Original train/val/test codec and integrity statistics were inspected; only first2048 validation IDs used for prior model evaluation',
                'original_test_already_used_for_previous_count_block_version': True,
                'source_data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'job_id': os.environ['SLURM_JOB_ID']}
    output = Path('multiplicity_candidates/replication_manifest.json')
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('_ids')}), flush=True)


if __name__ == '__main__':
    main()
