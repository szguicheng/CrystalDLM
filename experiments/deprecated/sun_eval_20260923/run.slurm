#!/bin/bash
#SBATCH --job-name=crystal-sun-pilot
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CRYSTAL_SOCKET="sun-$SLURM_JOB_ID"
trap 'tmux -L "$CRYSTAL_SOCKET" kill-server 2>/dev/null || true' EXIT TERM INT
tmux -L "$CRYSTAL_SOCKET" -f /dev/null new-session -d -s work -c "$PWD" "bash -c './env/bin/python -u evaluate_sun.py > run.log 2>&1; echo \$? > exit_code.txt'"
while tmux -L "$CRYSTAL_SOCKET" has-session -t work 2>/dev/null; do sleep 2; done
test -f exit_code.txt
exit "$(cat exit_code.txt)"
