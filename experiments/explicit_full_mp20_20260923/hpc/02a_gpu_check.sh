#!/bin/bash
set -euo pipefail
hostname
env | grep -E 'CUDA|SLURM_.*GPU|SLURM_JOB_ID'
nvidia-smi --query-gpu=index,uuid,compute_mode,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
cat /proc/self/cgroup
