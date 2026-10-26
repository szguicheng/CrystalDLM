#!/bin/bash
#SBATCH --job-name=b4-native-score
#SBATCH --partition=tiny
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/chenggui2/CrystalDLM/experiments/interleaved_relational_scaled32_20261006/eval_full_history_b4_001/b4_heldout_native_score_replay_%j.out
#SBATCH --error=/scratch/chenggui2/CrystalDLM/experiments/interleaved_relational_scaled32_20261006/eval_full_history_b4_001/b4_heldout_native_score_replay_%j.out

set -euo pipefail
cd /scratch/chenggui2/CrystalDLM
/scratch/chenggui2/CrystalDLM/experiments/interleaved_ar_diffusion_20261002/env/bin/python \
  /scratch/chenggui2/CrystalDLM/experiments/interleaved_relational_scaled32_20261006/b4_heldout_native_score_replay.py
