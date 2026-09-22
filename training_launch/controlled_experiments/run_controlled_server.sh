#!/usr/bin/env bash
# Explicit foreground orchestrator: one serial lane per selected GPU.
# Usage: bash training_launch/controlled_experiments/run_controlled_server.sh 0,1,2,3 [training arguments]
set -euo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IFS=, read -r -a devices <<< "${1:?Supply GPU IDs, e.g. 0,1,2,3}"
shift
if (( ${#devices[@]} < 1 || ${#devices[@]} > 4 )); then
    echo 'Select between one and four GPUs.' >&2; exit 2
fi
declare -A seen=()
for gpu in "${devices[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen[$gpu]:-}" ]]; then
        echo 'GPU IDs must be distinct nonnegative integers.' >&2; exit 2
    fi
    seen[$gpu]=1
done
for arg in "$@"; do
    if [[ "$arg" == --device || "$arg" == --device=* ]]; then
        echo 'Choose GPU IDs using the first argument, not --device.' >&2; exit 2
    fi
done
recipes=(A_inverse_sweep B_sqrt_sweep C_inverse_targeted D_sqrt_targeted)
# Serial dry-run prints all commands without allocating GPUs or child sessions.
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then
        for i in "${!recipes[@]}"; do
            gpu=${devices[$((i % ${#devices[@]}))]}
            bash "$dir/run_controlled.sh" "${recipes[$i]}" --device "$gpu" "$@"
        done
        exit
    fi
done
command -v setsid >/dev/null
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    wait || true
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
for lane in "${!devices[@]}"; do
    # Each lane has its own process group so Ctrl+C also stops its workers/tee.
    setsid bash -c '
        set -euo pipefail
        dir=$1; lane=$2; lanes=$3; gpu=$4; shift 4
        recipes=(A_inverse_sweep B_sqrt_sweep C_inverse_targeted D_sqrt_targeted)
        for ((i=lane; i<4; i+=lanes)); do
            bash "$dir/run_controlled.sh" "${recipes[$i]}" --device "$gpu" "$@"
        done
    ' _ "$dir" "$lane" "${#devices[@]}" "${devices[$lane]}" "$@" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
