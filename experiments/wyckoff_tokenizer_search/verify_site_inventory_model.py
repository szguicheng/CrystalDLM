"""CPU preflight inside an allocated node; no training or held-out evaluation."""
import hashlib
import json
import os
from pathlib import Path
import torch
from site_inventory_model import SiteInventoryModel
from train_profile_model import collate


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2)
    data = torch.load('multiplicity_candidates/data.pt', map_location='cpu')
    types = torch.zeros(len(data['wp_metadata']), dtype=torch.long)
    for p in json.loads(Path('site_point_groups/metadata.json').read_text())['positions']:
        types[p['wp_id']] = p['point_group_type']
    data['wp_site_type'] = types
    torch.manual_seed(17)
    a = SiteInventoryModel(data, 'aggregate').eval()
    torch.manual_seed(17)
    b = SiteInventoryModel(data, 'records').eval()
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    indices = torch.load('site_inventories/indices.pt', map_location='cpu')
    grouped, differing = {}, 0
    with torch.no_grad():
        for lo in range(0, len(data['profile_counts']), 256):
            ids = torch.arange(lo, min(lo + 256, len(data['profile_counts'])))
            assert torch.equal(a.encode_profiles(ids), b.encode_profiles(ids))
            ca, cb = a.encode_unaggregated(ids), b.encode_unaggregated(ids)
            for i, pid in enumerate(ids.tolist()):
                counts = data['profile_counts'][pid]
                occupied = counts > 0
                hs = types[data['profile_wp_ids'][pid][occupied]]
                if len(hs.unique()) == len(hs):
                    assert torch.allclose(ca[i], cb[i], atol=2e-6)
                differing += not torch.allclose(ca[i], cb[i], atol=2e-6)
                key = int(indices['profile_inventory_id'][pid])
                if key in grouped:
                    assert torch.allclose(ca[i], grouped[key], atol=2e-6)
                else:
                    grouped[key] = ca[i].clone()
        assert differing > 0
        batch = collate(data['splits']['train'][:16], torch.device('cpu'))
        hidden = batch['Z'] > 0
        changed = {k: v.clone() for k, v in batch.items()}
        changed['profile_id'] = a.context_candidates[batch['context']][..., 0]
        for model in (a, b):
            original, _ = model(batch, hidden)
            other, _ = model(changed, hidden)
            assert torch.allclose(original, other, atol=1e-6)
            assert torch.allclose(original.exp().sum(-1), torch.ones_like(original[..., 0]), atol=1e-6)
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'run_id': os.environ.get('CRYSTALDLM_RUN_ID'),
              'parameter_count_each': sum(p.numel() for p in a.parameters()),
              'identical_initial_parameters': True, 'fine_branch_parity_all_profiles': len(data['profile_counts']),
              'inventory_invariance': True, 'single_record_per_type_parity': True,
              'profiles_with_changed_coarse_features': differing, 'hidden_target_invariance': True,
              'source_hashes': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                ['site_inventory_model.py', 'train_site_inventory_model.py', 'verify_site_inventory_model.py']}}
    Path('site_inventories/model_preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
