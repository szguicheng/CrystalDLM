#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
$PY -u evaluate.py generate
$PY -u evaluate.py score
