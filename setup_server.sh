#!/usr/bin/env bash
# =============================================================================================
# Environment setup for dilated_tooth_seg_net.
#
# DEFAULT MODE TOUCHES NOTHING OUTSIDE ONE CONDA ENVIRONMENT.
# It does not install or modify the NVIDIA driver, system CUDA, nvcc, or conda itself; it does
# not run sudo, apt, `conda init` or `conda config`; it does not edit ~/.bashrc or ~/.condarc.
# It reuses whatever conda is already on the machine and creates a single new environment.
# Everything it creates lives in that one directory and is removed by deleting it.
#
#   bash setup_server.sh --check     # inspect and report ONLY - makes no changes at all
#   bash setup_server.sh             # create the environment (default, non-invasive)
#   bash setup_server.sh --help
#
# Run --check first on a machine you care about. It prints exactly what would be created.
#
# Opt-in flags for things that DO modify the system (never happen unless you pass them):
#   --with-driver     install an NVIDIA driver via apt (needs sudo + reboot)
#   --install-conda   install Miniconda, only if no existing conda was found
#
# Full rationale, troubleshooting and hardware caveats: Docs/SETUP.md
# =============================================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-dtsegnet}"
ENV_PREFIX=""                # explicit path for the env; auto-chosen if empty
CONDA_ROOT="${CONDA_ROOT:-}" # auto-detected if empty
PY_VERSION=3.10
CUDA_VERSION=12.1.1          # must stay consistent with requirements.txt's torch ...+cu121
GCC_VERSION=12               # CUDA 12.1's nvcc rejects gcc >= 13
MIN_DRIVER=525.60.13         # minimum driver for CUDA 12.1 runtime (CUDA 12.x minor-version compat)
MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECK_ONLY=0; WITH_DRIVER=0; INSTALL_CONDA=0; SYSTEM_CUDA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|--dry-run)  CHECK_ONLY=1 ;;
    --with-driver)      WITH_DRIVER=1 ;;
    --install-conda)    INSTALL_CONDA=1 ;;
    --system-cuda)      SYSTEM_CUDA=1 ;;
    --env-name)         ENV_NAME="${2:?--env-name needs a value}"; shift ;;
    --env-prefix)       ENV_PREFIX="${2:?--env-prefix needs a value}"; shift ;;
    --conda-root)       CONDA_ROOT="${2:?--conda-root needs a value}"; shift ;;
    --env-only|--skip-driver) : ;;   # accepted for backwards compatibility; now the default
    --help|-h) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# conda's generated activation scripts are NOT `set -u` safe - conda-forge's
# activate-gcc_linux-64.sh dereferences SYS_SYSROOT unconditionally, so sourcing it under nounset
# aborts with "SYS_SYSROOT: unbound variable". Drop nounset only around conda's own code.
conda_source()     { set +u; # shellcheck disable=SC1091
                     . "$CONDA_ROOT/etc/profile.d/conda.sh"; set -u; }
conda_activate()   { set +u; conda activate "$1"; set -u; }
conda_deactivate() { set +u; conda deactivate;    set -u; }

# NB: checks below capture output into variables instead of piping into `grep -q` / `head -1`.
# Under `set -o pipefail` those consumers exit at the first match and kill the producer with
# SIGPIPE (exit 141), so the pipeline reports FAILURE precisely when the pattern WAS found - the
# same pipefail trap documented in run_experiment_matrix.sh.

# ---------------------------------------------------------------------------------------------
# Phase 0 - preflight (read-only)
# ---------------------------------------------------------------------------------------------
log "Phase 0: preflight (read-only)"
[[ "$(uname -s)" == "Linux"  ]] || die "Linux-only (found $(uname -s))."
[[ "$(uname -m)" == "x86_64" ]] || die "x86_64-only; the pinned torch/CUDA wheels have no $(uname -m) build."
lspci_out="$(lspci 2>/dev/null || true)"
grep -qiE 'nvidia' <<<"$lspci_out" || warn "No NVIDIA GPU seen on the PCI bus."
info "host   : $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
info "cores  : $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GB"

# ---------------------------------------------------------------------------------------------
# Phase 1 - NVIDIA driver: INSPECT ONLY unless --with-driver
# ---------------------------------------------------------------------------------------------
log "Phase 1: NVIDIA driver"
if nvidia-smi >/dev/null 2>&1; then
  driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  info "found  : driver $driver_ver  ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
  info "         LEAVING IT ALONE - this script never modifies an existing driver."
  # torch 2.1.0+cu121 needs a driver new enough for the CUDA 12.1 runtime. Newer is always fine
  # (drivers are backwards compatible); older means torch cannot initialise CUDA at all.
  if [[ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$driver_ver" | sort -V | head -1)" != "$MIN_DRIVER" ]]; then
    warn "Driver $driver_ver is OLDER than $MIN_DRIVER, the minimum for the CUDA 12.1 runtime that
   torch 2.1.0+cu121 needs. torch will fail to initialise CUDA. Updating the driver is a system
   change this script deliberately will not make - talk to whoever administers the machine."
  fi
elif [[ $WITH_DRIVER -eq 1 ]]; then
  [[ $CHECK_ONLY -eq 1 ]] && { info "would install an NVIDIA driver via apt (--with-driver)"; } || {
    sb_state="$(mokutil --sb-state 2>/dev/null || true)"
    if grep -qi 'SecureBoot enabled' <<<"$sb_state"; then
      die "Secure Boot is ENABLED; DKMS-built NVIDIA modules are unsigned and will not load.
   Requires disabling Secure Boot or enrolling a MOK at the physical console. See Docs/SETUP.md."
    fi
    sudo -v || die "--with-driver needs sudo."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common pciutils
    drivers_out="$(ubuntu-drivers devices 2>/dev/null || true)"
    recommended="$(awk '/recommended/{print $3; exit}' <<<"$drivers_out")"
    branch="$(sed -E 's/^nvidia-driver-([0-9]+).*/\1/' <<<"${recommended:-}")"
    target=""
    if [[ -n "$branch" ]]; then
      for c in "nvidia-driver-${branch}-server-open" "nvidia-driver-${branch}-open" "$recommended"; do
        if apt-cache show "$c" >/dev/null 2>&1; then target="$c"; break; fi
      done
    fi
    [[ -n "$target" ]] || die "Could not determine a driver package; install one manually."
    info "installing: $target"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$target"
    dkms_out="$(sudo dkms status 2>/dev/null || true)"
    grep -q nvidia <<<"$dkms_out" || warn "No nvidia entry in 'dkms status' - module may not have built.
   Try: sudo apt-get install linux-headers-\$(uname -r)"
    log "REBOOT REQUIRED - then re-run this script (it will skip this phase)."
    exit 0
  }
else
  die "No working NVIDIA driver (nvidia-smi failed) and --with-driver was not passed.
   This script does not install drivers unless you explicitly ask it to."
fi

# ---------------------------------------------------------------------------------------------
# Phase 2 - GPU capability -> TORCH_CUDA_ARCH_LIST (read-only)
# ---------------------------------------------------------------------------------------------
log "Phase 2: GPU compute capability"
ARCH_LIST=""
if nvidia-smi >/dev/null 2>&1; then
  mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
  if [[ ${#caps[@]} -gt 0 ]]; then
    info "compute capability: ${caps[*]}"
    # CUDA 12.1 emits native SASS for sm_50..sm_90 only. Blackwell (RTX 50xx = sm_120,
    # B200 = sm_100) can run solely via PTX JIT, hence the '+PTX' suffix - slow to start and
    # not guaranteed. Warn rather than fail obscurely later.
    highest="$(printf '%s\n' "${caps[@]}" | sort -V | tail -1)"
    if [[ "$(printf '%s\n9.0\n' "$highest" | sort -V | tail -1)" != "9.0" && "$highest" != "9.0" ]]; then
      warn "Compute capability $highest is NEWER than CUDA ${CUDA_VERSION%.*} supports natively (max sm_90).
   The build falls back to PTX + driver JIT. For Blackwell, this repo's torch/CUDA pins are the
   wrong stack - see Docs/SETUP.md, 'GPUs newer than CUDA 12.1'."
    fi
    ARCH_LIST="$(printf '%s;' "${caps[@]}" | sed 's/;$//')+PTX"
    info "TORCH_CUDA_ARCH_LIST=$ARCH_LIST"
  fi
fi

# ---------------------------------------------------------------------------------------------
# Phase 3 - locate an EXISTING conda (install one only with --install-conda)
# ---------------------------------------------------------------------------------------------
log "Phase 3: conda"
if [[ -z "$CONDA_ROOT" ]]; then
  # Prefer conda's own idea of where it lives, then PATH, then the usual install locations.
  # This is what stops the script from installing a second conda next to an existing one.
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
    CONDA_ROOT="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(cd "$(dirname "$(command -v conda)")/.." && pwd)"
  else
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                /opt/conda /opt/miniconda3 /opt/anaconda3 /opt/miniforge3 \
                /usr/local/miniconda3 /usr/local/anaconda3 /usr/share/miniconda; do
      [[ -x "$cand/bin/conda" ]] && { CONDA_ROOT="$cand"; break; }
    done
  fi
fi

if [[ -n "$CONDA_ROOT" && -x "$CONDA_ROOT/bin/conda" ]]; then
  info "found  : $CONDA_ROOT  ($("$CONDA_ROOT/bin/conda" --version))"
  info "         REUSING IT - no conda is installed, upgraded or reconfigured."
elif [[ $INSTALL_CONDA -eq 1 ]]; then
  CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
  if [[ $CHECK_ONLY -eq 1 ]]; then
    info "would install Miniconda to $CONDA_ROOT (--install-conda)"
  else
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/miniconda.sh" "$MINICONDA_URL"
    bash "$tmp/miniconda.sh" -b -p "$CONDA_ROOT"
    info "installed: $("$CONDA_ROOT/bin/conda" --version)"
  fi
else
  die "No conda found, and --install-conda was not passed.
   Searched \$CONDA_EXE, \$PATH, ~/miniconda3, ~/anaconda3, ~/miniforge3, /opt/conda and others.
   If conda IS installed somewhere else, point at it:  --conda-root /path/to/conda
   Otherwise allow this script to install its own:      --install-conda"
fi

# Decide WHERE the env goes. On a shared server the conda install is often root-owned, so
# `conda create -n NAME` (which writes into <conda_root>/envs) would fail or need sudo. Fall back
# to the per-user env directory, which conda already searches by default, so `conda activate NAME`
# still works. Either way nothing outside this one directory is written.
if [[ -z "$ENV_PREFIX" ]]; then
  if [[ -w "$CONDA_ROOT/envs" ]]; then
    ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
    info "env dir: $ENV_PREFIX (conda install is user-writable)"
  else
    ENV_PREFIX="$HOME/.conda/envs/$ENV_NAME"
    info "env dir: $ENV_PREFIX (conda install is not user-writable, using your home)"
  fi
fi

# Warn about a globally-set LD_LIBRARY_PATH pointing at a system CUDA. It is not modified, but it
# can shadow the environment's own CUDA libraries at runtime and produce confusing version errors.
if [[ -n "${LD_LIBRARY_PATH:-}" ]] && grep -qi cuda <<<"${LD_LIBRARY_PATH:-}"; then
  warn "LD_LIBRARY_PATH references a system CUDA:
   ${LD_LIBRARY_PATH}
   Not changed by this script, but it can shadow the env's CUDA 12.1 libraries at runtime.
   If torch reports an unexpected CUDA version, unset it for the training shell."
fi

# ---------------------------------------------------------------------------------------------
# Phase 4 - the environment (the ONLY thing this script creates by default)
# ---------------------------------------------------------------------------------------------
if [[ $CHECK_ONLY -eq 1 ]]; then
  log "Check complete - NOTHING WAS CHANGED"
  cat <<MSG
    A normal run would create exactly one directory:

        $ENV_PREFIX

    containing python $PY_VERSION, $( [[ $SYSTEM_CUDA -eq 1 ]] && echo "no CUDA toolkit (--system-cuda: builds against the system nvcc)" || echo "an env-local CUDA $CUDA_VERSION toolkit + gcc $GCC_VERSION (~4 GB)" )
    and the pinned python dependencies, plus a compiled pointnet2_ops extension.

    It would NOT touch: the NVIDIA driver, system CUDA/nvcc, the conda installation itself,
    ~/.condarc, ~/.bashrc, or anything requiring sudo.
    To undo a real run afterwards:  rm -rf "$ENV_PREFIX"

    Run without --check to proceed.
MSG
  exit 0
fi

log "Phase 4: creating env at $ENV_PREFIX"
mkdir -p "$(dirname "$ENV_PREFIX")"
conda_source

# --override-channels -c conda-forge on EVERY command, rather than `conda config`, so the machine's
# existing channel configuration and ~/.condarc are left exactly as they are. It also avoids
# Anaconda's 'defaults' channel, whose Terms of Service carry commercial licensing conditions that
# an installer must not accept on the operator's behalf.
if [[ -d "$ENV_PREFIX" ]]; then
  info "env already exists, skipping creation"
else
  "$CONDA_ROOT/bin/conda" create -p "$ENV_PREFIX" --override-channels -c conda-forge \
    "python=$PY_VERSION" -y
fi

# The CUDA toolkit goes INSIDE the env, never system-wide:
#  - pointnet2_ops_lib is a CUDAExtension compiled from source, so nvcc must match the CUDA torch
#    was built against (12.1) or the extension is ABI-incompatible.
#  - An env-local toolkit shadows the system nvcc only while the env is activated, so the machine's
#    own CUDA install is untouched and other users are unaffected.
# --system-cuda skips this and builds against whatever nvcc is on PATH instead.
if [[ $SYSTEM_CUDA -eq 1 ]]; then
  command -v nvcc >/dev/null 2>&1 || die "--system-cuda given but no nvcc on PATH."
  sys_cuda="$(nvcc --version | awk '/release/{gsub(",","",$5); print $5}')"
  info "using system nvcc $sys_cuda (--system-cuda; env-local toolkit skipped)"
  [[ "${sys_cuda%%.*}" == "12" ]] || warn "System CUDA $sys_cuda is not 12.x; torch is built against 12.1.
   The extension may fail to build or be ABI-incompatible. Drop --system-cuda to use CUDA $CUDA_VERSION."
elif [[ -x "$ENV_PREFIX/bin/nvcc" ]]; then
  info "env-local nvcc already present, skipping toolkit install"
else
  "$CONDA_ROOT/bin/conda" install -p "$ENV_PREFIX" -y \
    --override-channels -c "nvidia/label/cuda-$CUDA_VERSION" -c conda-forge \
    cuda-toolkit "gxx_linux-64=$GCC_VERSION" "gcc_linux-64=$GCC_VERSION"
fi

conda_activate "$ENV_PREFIX"
[[ $SYSTEM_CUDA -eq 1 ]] || "$CONDA_ROOT/bin/conda" env config vars set -p "$ENV_PREFIX" \
  "CUDA_HOME=$ENV_PREFIX" >/dev/null
[[ -z "$ARCH_LIST" ]] || "$CONDA_ROOT/bin/conda" env config vars set -p "$ENV_PREFIX" \
  "TORCH_CUDA_ARCH_LIST=$ARCH_LIST" >/dev/null
conda_deactivate; conda_activate "$ENV_PREFIX"
info "nvcc: $(nvcc --version | awk '/release/{print $5,$6}')"
if [[ -n "${CC:-}" ]]; then cc_version="$("$CC" --version 2>&1)"; info "gcc : ${cc_version%%$'\n'*}"; fi

# ---------------------------------------------------------------------------------------------
# Phase 5 - python dependencies (inside the env only)
# ---------------------------------------------------------------------------------------------
log "Phase 5: pip requirements"
pip install --no-input -r "$REPO_DIR/requirements.txt"

# ---------------------------------------------------------------------------------------------
# Phase 6 - pointnet2_ops CUDA extension
# ---------------------------------------------------------------------------------------------
log "Phase 6: building pointnet2_ops (CUDA extension)"
# --no-build-isolation is REQUIRED: pointnet2_ops_lib/setup.py imports torch at module level, but
# a PEP 517 isolated build environment has no torch, so it fails with ModuleNotFoundError before
# compiling anything.
pip install --no-input --no-build-isolation "$REPO_DIR/pointnet2_ops_lib"

# ---------------------------------------------------------------------------------------------
# Phase 7 - verification
# ---------------------------------------------------------------------------------------------
log "Phase 7: verifying"
python - <<'PYCHECK'
import sys
import torch

ok = True
def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"    [{'OK ' if cond else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

check("python 3.10", sys.version_info[:2] == (3, 10), sys.version.split()[0])
check("torch 2.1.0+cu121", torch.__version__.startswith("2.1.0+cu121"), torch.__version__)
check("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    check("GPU visible", True, torch.cuda.get_device_name(0))
    a = torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
    check("GPU matmul", torch.isfinite(a @ a).all().item())

import numpy, numba, lightning              # noqa: E402
check("numpy 1.24.4 (torch ABI)", numpy.__version__ == "1.24.4", numpy.__version__)
check("numba importable", True, numba.__version__)
check("lightning importable", True, lightning.__version__)

# The real test: the custom CUDA kernels this repo cannot train without. torch.cuda.is_available()
# still returns True on a machine where these were built for the wrong arch and every launch fails.
from pointnet2_ops import pointnet2_utils as pu   # noqa: E402
if torch.cuda.is_available():
    xyz = torch.randn(2, 4096, 3, device="cuda").contiguous()
    idx = pu.furthest_point_sample(xyz, 512); torch.cuda.synchronize()
    check("pointnet2 CUDA kernel (FPS)", tuple(idx.shape) == (2, 512)
          and all(len(set(r.tolist())) == 512 for r in idx))

print()
print("    RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
PYCHECK

log "Done - the only thing created was $ENV_PREFIX"
cat <<MSG
    Activate it with:

        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        conda activate "$ENV_PREFIX"

    (No shell profile was modified. If this machine's conda is already initialised in your
    shell, plain 'conda activate $ENV_NAME' works too.)

    To remove everything this script created:  rm -rf "$ENV_PREFIX"
MSG
