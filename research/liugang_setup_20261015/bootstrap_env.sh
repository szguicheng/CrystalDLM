#!/usr/bin/env bash
set -euo pipefail
source /home/liugang/src/crystalDLM/research/liugang_setup_20261015/runtime.sh
export CONDA_REGISTER_ENVS=false PYTHONDONTWRITEBYTECODE=1
/home/liugang/software/miniforge3/condabin/conda create \
  --solver libmamba \
  --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python=3.11 pip -y \
  --override-channels -c conda-forge
