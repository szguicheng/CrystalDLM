#!/bin/bash
#SBATCH --job-name=rel-repair-controls
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/controls-%j.out
#SBATCH --error=experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/controls-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# One allocation: independent processes and fresh initializations, no shared weights.
bash experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/commit_probe.slurm
bash experiments/interleaved_relational_b16_20261006/pilot_001.slurm
bash experiments/interleaved_relational_joint_20261006/pilot_001.slurm
