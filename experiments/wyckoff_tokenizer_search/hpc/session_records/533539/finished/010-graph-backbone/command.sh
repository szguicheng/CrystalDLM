#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python prepare_graph_backbone.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python benchmark_graph_backbone.py
