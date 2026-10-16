#!/bin/bash
set -euo pipefail
for variant in aggregate records; do
  /home/chenggui2/scratch/anaconda3/envs/matgl/bin/python train_site_inventory_model.py --variant "$variant" --seed 17 --steps 16 --val-limit 64 --eval-every 16 --smoke
done
