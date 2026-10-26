#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=experiments/design_understanding_full_20261025
python3 "$stage/annotate.py"
python3 "$stage/prepare.py" finalize
python3 "$stage/validate.py"
rsync -az "$stage/" LiuGangGPU:/home/liugang/src/crystalDLM/experiments/design_understanding_full_20261025/
ssh LiuGangGPU 'cd /home/liugang/src/crystalDLM && source research/liugang_setup_20261015/runtime.sh && /home/liugang/software/miniforge3/condabin/mamba run --prefix /home/liugang/src/crystalDLM/envs/crystaldlm python experiments/design_understanding_full_20261025/validate.py'
