#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python -u sampling_followup.py --count 64
