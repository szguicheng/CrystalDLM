#!/usr/bin/env bash
set -euo pipefail
CRYSTAL_PROJECT=/home/liugang/src/crystalDLM
source "$CRYSTAL_PROJECT/research/liugang_setup_20261015/runtime.sh"
CRYSTAL_PREFIX="${CRYSTAL_ENV_PREFIX:-$CRYSTAL_PROJECT/envs/crystaldlm}"
# Activation scripts override inherited cache paths; set them after activation too.
exec /home/liugang/software/miniforge3/condabin/mamba run --prefix "$CRYSTAL_PREFIX" env \
  MAMBA_ROOT_PREFIX="$CRYSTAL_PROJECT/.mamba" CONDA_ENVS_PATH="$CRYSTAL_PROJECT/envs" \
  CONDA_PKGS_DIRS="$CRYSTAL_PROJECT/.cache/mamba/pkgs" CONDA_REGISTER_ENVS=false \
  PIP_CACHE_DIR="$CRYSTAL_PROJECT/.cache/pip" HF_HOME="$CRYSTAL_PROJECT/.cache/huggingface" \
  TORCH_HOME="$CRYSTAL_PROJECT/.cache/torch" XDG_CACHE_HOME="$CRYSTAL_PROJECT/.cache" \
  XDG_DATA_HOME="$CRYSTAL_PROJECT/.local/share" XDG_CONFIG_HOME="$CRYSTAL_PROJECT/.config" \
  TRANSFORMERS_CACHE="$CRYSTAL_PROJECT/.cache/huggingface/hub" \
  TMPDIR="$CRYSTAL_PROJECT/tmp" TMP="$CRYSTAL_PROJECT/tmp" TEMP="$CRYSTAL_PROJECT/tmp" \
  MPLCONFIGDIR="$CRYSTAL_PROJECT/.cache/matplotlib" CUDA_CACHE_PATH="$CRYSTAL_PROJECT/.cache/cuda" \
  TORCH_EXTENSIONS_DIR="$CRYSTAL_PROJECT/.cache/torch_extensions" \
  PYTHONDONTWRITEBYTECODE=1 python "$@"
