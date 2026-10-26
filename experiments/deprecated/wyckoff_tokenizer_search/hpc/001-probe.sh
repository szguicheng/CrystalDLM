#!/bin/bash
set -euo pipefail
tmux -L "crystaldlm-$SLURM_JOB_ID" list-sessions -F '#{session_name} attached=#{session_attached} windows=#{session_windows}'
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python hpc/probe_session.py
