#!/bin/bash
set -euo pipefail
for variant in aggregate records; do
  /home/chenggui2/scratch/anaconda3/envs/matgl/bin/python train_site_inventory_model.py --variant "$variant" --seed 17 --steps 10000 --val-limit 2048 --eval-every 1000
done
