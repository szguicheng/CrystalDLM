#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_roundtrip.py --csv /scratch/chenggui2/CrystalDLM/tokenizer_search_20260908/input/train.csv --out results
