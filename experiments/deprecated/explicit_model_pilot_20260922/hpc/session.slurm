#!/bin/bash
#SBATCH --job-name=explicit-model-pilot
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/session-%j.out
#SBATCH --error=logs/session-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CRYSTAL_DIR="$PWD/compute_allocations/$SLURM_JOB_ID"
CRYSTAL_SOCKET="crystaldlm-$SLURM_JOB_ID"
mkdir -p "$CRYSTAL_DIR"
trap 'tmux -L "$CRYSTAL_SOCKET" kill-server 2>/dev/null || true' EXIT TERM INT
tmux -L "$CRYSTAL_SOCKET" -f /dev/null new-session -d -s work -c "$PWD" "/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python hpc/queue_worker.py $CRYSTAL_DIR"
while tmux -L "$CRYSTAL_SOCKET" has-session -t work 2>/dev/null; do sleep 2; done
