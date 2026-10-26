#!/bin/bash
exec ssh -o ControlMaster=auto -o ControlPersist=7200 \
  -o ControlPath=/tmp/crystaldlm-burgundy-%C \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ConnectTimeout=10 Burgundy "$@"
