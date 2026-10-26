#!/bin/bash
#SBATCH --job-name=crystaldlm-session
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
CRYSTALDLM_ALLOCATION="$SLURM_SUBMIT_DIR/compute_allocations/$SLURM_JOB_ID"
CRYSTALDLM_SOCKET="crystaldlm-$SLURM_JOB_ID"
mkdir -p "$CRYSTALDLM_ALLOCATION"
chmod 700 "$CRYSTALDLM_ALLOCATION"
trap 'tmux -L "$CRYSTALDLM_SOCKET" kill-server 2>/dev/null || true' EXIT TERM INT
printf -v CRYSTALDLM_COMMAND '%q ' /home/chenggui2/scratch/anaconda3/bin/python3 \
  "$SLURM_SUBMIT_DIR/hpc/queue_worker.py" "$CRYSTALDLM_ALLOCATION"
tmux -L "$CRYSTALDLM_SOCKET" -f /dev/null new-session -d -s work -c "$SLURM_SUBMIT_DIR" "$CRYSTALDLM_COMMAND"
while tmux -L "$CRYSTALDLM_SOCKET" has-session -t work 2>/dev/null; do
  sleep 2
done
