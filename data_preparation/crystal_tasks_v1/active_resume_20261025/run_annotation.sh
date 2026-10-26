#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 125
caffeinate -i python3 annotate.py >> logs/annotation_20261026T0500Z.log 2>&1
status=$?
printf '%s\n' "$status" > logs/annotation_20261026T0500Z.exit
exit "$status"
