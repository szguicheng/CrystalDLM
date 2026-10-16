#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python prepare_full_prototype_corpus.py --split train --workers 4
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python prepare_full_prototype_corpus.py --split val --workers 4
