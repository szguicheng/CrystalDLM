"""Replay the frozen CPU corruption stream under the training Torch version."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('native_mask_replay.json')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run inside a compute allocation'
    assert not OUT.exists(), 'Do not overwrite native replay evidence'
    status = json.loads((ROOT / 'pilot_001/diffusion_b32/status.json').read_text())
    receipt = json.loads((ROOT / 'pilot_001/training_exposure.json').read_text())
    assert torch.__version__ == status['packages']['torch'] == '2.2.0+cu121'
    events = receipt['events']
    digest = hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == receipt['events_sha256'] == status['task_exposure_sha256']
    rng = torch.Generator().manual_seed(1017)
    counts = defaultdict(lambda: dict(examples=0, full_k32=0, first_block_fully_masked={str(b): 0 for b in (8, 16, 32, 64)}))
    steps = []
    for event in events:
        if event['task'] not in ('G', 'E'):
            continue
        k = int(torch.randint(1, 33, (1,), generator=rng))
        mask = torch.rand(2560, generator=rng) < k / 32
        steps.append(k)
        row = counts[event['task'], event['id']]
        row['examples'] += 1
        row['full_k32'] += k == 32
        for b in (8, 16, 32, 64):
            row['first_block_fully_masked'][str(b)] += bool(mask[:b].all())
    # Memory mapping avoids reading model/optimizer tensors for this RNG check.
    saved = torch.load(str(ROOT / 'pilot_001/diffusion_b32/checkpoint.pt'), mmap=True, map_location='cpu', weights_only=True)
    checkpoint_rng = saved['corruption_rng']
    assert torch.equal(rng.get_state(), checkpoint_rng), 'Replay differs from actual checkpoint RNG state'
    result = dict(torch_version=torch.__version__, seed=1017, events_sha256=digest,
                  k_sequence_sha256=hashlib.sha256(json.dumps(steps, separators=(',', ':')).encode()).hexdigest(),
                  corruption_rng_matches_checkpoint=True,
                  per_case=[dict(task=key[0], id=key[1], **value) for key, value in sorted(counts.items())])
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_case'}), flush=True)


if __name__ == '__main__':
    main()
