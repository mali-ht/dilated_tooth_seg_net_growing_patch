#!/usr/bin/env bash
# Four runs sequentially, attached to this terminal. Stops on the first failure.
set -euo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for recipe in A_inverse_sweep B_sqrt_sweep C_inverse_targeted D_sqrt_targeted; do
    bash "$dir/run_controlled.sh" "$recipe" "$@"
done
