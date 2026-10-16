#!/usr/bin/env bash
set -euo pipefail
source /home/liugang/src/crystalDLM/research/liugang_setup_20261015/runtime.sh
cd "$CRYSTAL_PROJECT"
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/crystaldlm"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
"$MAMBA" run --prefix "$PREFIX" python -m pip install --no-index --no-deps \
  "$CRYSTAL_PROJECT"/.cache/wheelhouse/*.whl
"$MAMBA" run --prefix "$PREFIX" python -m pip install --no-cache-dir \
  torch==2.6.0
"$MAMBA" run --prefix "$PREFIX" python -m pip install --no-cache-dir \
  transformers==4.57.6 peft==0.18.0 'accelerate>=1.0,<2' \
  numpy==2.2.6 pymatgen==2025.10.7 pyxtal==1.0.8
"$MAMBA" run --prefix "$PREFIX" python -m pip check
"$MAMBA" run --prefix "$PREFIX" python -m pip freeze > research/liugang_setup_20261015/pip-freeze.txt
"$MAMBA" run --prefix "$PREFIX" python -c 'import torch,transformers,peft,accelerate; print({"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,"accelerate":accelerate.__version__,"cuda_runtime":torch.version.cuda})'
