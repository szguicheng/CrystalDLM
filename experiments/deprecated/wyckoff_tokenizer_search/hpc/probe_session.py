"""Verify the allocated compute surface without dumping the full environment."""
import json
import os
import socket
import torch

assert os.environ.get('SLURM_JOB_ID')
assert torch.cuda.is_available()
value = torch.ones(16, device='cuda').sum().item()
assert value == 16
print(json.dumps({'job_id': os.environ['SLURM_JOB_ID'], 'run_id': os.environ['CRYSTALDLM_RUN_ID'],
                  'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(),
                  'torch': torch.__version__, 'gpu_kernel_check': 'PASS'}), flush=True)
