#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
$PY -u train.py preflight
for SEED in 17 29; do
    for REPRESENTATION in explicit wyckoff; do
        $PY -u train.py train --representation "$REPRESENTATION" --seed "$SEED" --steps 4000
    done
done
