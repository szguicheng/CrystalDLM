#!/usr/bin/env bash
# Source this file for project-scoped training and package operations.
export CRYSTAL_PROJECT=/home/liugang/src/crystalDLM
export MAMBA_ROOT_PREFIX="$CRYSTAL_PROJECT/.mamba"
export CONDA_ENVS_PATH="$CRYSTAL_PROJECT/envs"
export CONDA_PKGS_DIRS="$CRYSTAL_PROJECT/.cache/mamba/pkgs"
export PIP_CACHE_DIR="$CRYSTAL_PROJECT/.cache/pip"
export HF_HOME="$CRYSTAL_PROJECT/.cache/huggingface"
export TORCH_HOME="$CRYSTAL_PROJECT/.cache/torch"
export XDG_CACHE_HOME="$CRYSTAL_PROJECT/.cache"
export XDG_DATA_HOME="$CRYSTAL_PROJECT/.local/share"
export XDG_CONFIG_HOME="$CRYSTAL_PROJECT/.config"
export TMPDIR="$CRYSTAL_PROJECT/tmp"
export MPLCONFIGDIR="$CRYSTAL_PROJECT/.cache/matplotlib"
export CUDA_CACHE_PATH="$CRYSTAL_PROJECT/.cache/cuda"
export TORCH_EXTENSIONS_DIR="$CRYSTAL_PROJECT/.cache/torch_extensions"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$TMPDIR" \
  "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
