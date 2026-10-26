#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_origin.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_program.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python audit_corpus.py
