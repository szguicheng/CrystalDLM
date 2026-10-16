#!/bin/bash
#SBATCH --job-name=crystaldlm-compact-10k
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
if [[ "${SEED:-17}" == 29 ]]; then
  VARIANTS="rlemerge rle"
else
  VARIANTS="rle rlemerge"
fi
for variant in $VARIANTS; do
  "$PYTHON" direct_words.py --variant "$variant" --data-path count_merge_candidates/data.pt --run-root compact_10k_runs --compact --seed "${SEED:-17}" --steps 10000 --eval-every 1000
done
"$PYTHON" compare_direct_words.py --root compact_10k_runs --method rlemerge --seed "${SEED:-17}"
