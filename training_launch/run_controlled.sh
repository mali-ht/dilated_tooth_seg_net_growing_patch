#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
recipe=${1:?Usage: run_controlled.sh RECIPE [training arguments]}
shift
cmd=(python3 -u -m experiments.controlled.train --recipe "$recipe" "$@")
for arg in "$@"; do
    if [[ "$arg" == --dry-run || "$arg" == --help ]]; then
        exec "${cmd[@]}"
    fi
done
mkdir -p logs/experiments/controlled
log="logs/experiments/controlled/${recipe}_$(date +%Y%m%d_%H%M%S)_$$.log"
printf -v command_line '%q ' "${cmd[@]}"
# Real terminal for Lightning's bar; -e propagates training failure through tee.
script -qec "$command_line" /dev/null 2>&1 | tee "$log"
