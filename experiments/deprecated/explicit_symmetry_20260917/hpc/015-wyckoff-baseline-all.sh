#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_wyckoff_baseline.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python audit_wyckoff_baseline.py
