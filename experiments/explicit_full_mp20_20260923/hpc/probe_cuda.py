"""Probe only the single GPU visible inside this Slurm allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get('SLURM_JOB_ID')
uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
assert len(uuids) == 1, uuids
records = []
for name, visible in [('scheduler_index', os.environ.get('CUDA_VISIBLE_DEVICES', '')), ('allocated_uuid', uuids[0])]:
    result = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__); print(torch.cuda.device_count()); print(torch.ones(1,device="cuda")); print(torch.cuda.get_device_name())'],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': visible}, text=True, capture_output=True)
    records.append({'mode': name, 'visible_device': visible, 'exit_code': result.returncode,
                    'stdout': result.stdout, 'stderr': result.stderr})
Path('hpc/gpu_probe.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records), flush=True)
