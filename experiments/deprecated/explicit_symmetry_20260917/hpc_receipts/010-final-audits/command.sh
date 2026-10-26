#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_family_blocks.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_origin.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python test_program.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python audit_corpus.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python audit_spacegroups.py
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python audit_wyckoff_families.py
