#!/bin/bash
exec rsync -e 'ssh -o ControlMaster=auto -o ControlPersist=7200 -o ControlPath=/tmp/crystaldlm-burgundy-%C -o ConnectTimeout=10' "$@"
