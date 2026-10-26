#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export CONDA_REGISTER_ENVS=false PYTHONDONTWRITEBYTECODE=1
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
ENV_PREFIX="$CRYSTAL_PROJECT/envs/grace"
OUT="$CRYSTAL_PROJECT/experiments/structure_request_eval_20261017"
trap 'code=$?; echo "$code" > "$OUT/grace-install-exit.txt"' EXIT
/home/liugang/software/miniforge3/condabin/conda create --solver libmamba --prefix "$ENV_PREFIX" python=3.11 pip -y --override-channels -c conda-forge
"$MAMBA" run --prefix "$ENV_PREFIX" python -m pip install --no-cache-dir 'tensorpotential==0.6.2' 'tensorflow[and-cuda]==2.20.0' 'tf-keras>=2.20,<2.21' 'numpy==2.2.6' 'pymatgen==2025.10.7' 'ase==3.29.0' 'pyxtal==1.0.8'
"$MAMBA" run --prefix "$ENV_PREFIX" python -m pip freeze > "$OUT/grace-environment.txt"
