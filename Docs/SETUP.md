# Environment setup (GPU server)

How to get `dilated_tooth_seg_net` training on a fresh Ubuntu GPU machine, and why the stack is
pinned the way it is. Written 2026-08-20 after building it end-to-end on a 12-core / RTX 3090
Ubuntu 24.04 box; every command and failure mode below was reproduced on that machine.

For the model, the dataset layout and the original paper, see [README.md](README.md).

---

## TL;DR

```bash
git clone <this repo> && cd dilated_tooth_seg_net
bash setup_server.sh --check   # report what would happen; changes NOTHING
bash setup_server.sh           # create the environment
```

`setup_server.sh` is idempotent — re-running it skips whatever is already in place, so it is safe
after a failure and safe on a half-configured machine.

---

## Installing on a machine you don't administer

**By default this script touches nothing outside a single conda environment.** That is the normal
case for a shared or already-provisioned GPU server, where the driver, CUDA and conda are managed
by somebody else and must not be disturbed.

It does **not**:

- install, upgrade or reconfigure the NVIDIA driver, system CUDA, or `nvcc`
- install or upgrade conda — it *finds and reuses* the conda already on the machine
- run `sudo` or `apt`, or require any privilege beyond writing one directory
- run `conda init` or `conda config`, or edit `~/.bashrc`, `~/.condarc`, or any shell profile

It creates exactly one directory (the environment) and nothing else. Undo the whole thing with
`rm -rf <env path>`; the path is printed at the end of every run.

Run `--check` first. It performs the full inspection, prints the environment path it would create
and what would go in it, then exits without writing anything:

```bash
bash setup_server.sh --check
```

Two flags opt in to system changes. **Neither ever happens unless you pass it explicitly**, and
without them the script fails with an explanatory message rather than doing something invasive:

| Flag | What it enables |
|---|---|
| `--with-driver` | Install an NVIDIA driver via apt. Needs sudo and a reboot. |
| `--install-conda` | Install Miniconda, and only if no existing conda was found anywhere. |

Other useful flags:

| Flag | Purpose |
|---|---|
| `--conda-root PATH` | Use a conda the auto-detection missed (it searches `$CONDA_EXE`, `$PATH`, `~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `~/mambaforge`, `/opt/conda`, and more). |
| `--env-prefix PATH` | Put the environment somewhere specific — e.g. a scratch volume rather than `$HOME`. |
| `--env-name NAME` | Name the environment (default `dtsegnet`). |
| `--system-cuda` | Build against the machine's existing `nvcc` instead of installing an env-local CUDA 12.8 (saves ~4 GB; see the caveat below). |

### Where the environment is placed

If the conda installation is user-writable, the environment goes in its usual `envs/` directory,
so plain `conda activate dtsegnet` works. If conda is root-owned — common for `/opt/conda` on a
shared server — writing there would need sudo, so the script falls back to `~/.conda/envs/`, which
conda already searches by default. Either way the name still resolves and nothing shared is
modified.

### Should you use `--system-cuda`?

Probably not, even though the machine already has CUDA. The env-local toolkit exists because
`pointnet2_ops_lib` is compiled from source and `nvcc` must match the CUDA that torch was built
against (**12.8**). A server's system CUDA is rarely exactly 12.8 — the Blackwell box this was
deployed to ships 13.2 — and a mismatch produces either a build failure or, worse, an extension
that builds cleanly and then misbehaves at runtime.

The env-local toolkit is not a system change: it is inside the environment directory and shadows
the system `nvcc` **only while that environment is activated**. Other users and other projects see
nothing different. The cost is about 4 GB of disk.

Use `--system-cuda` only if disk is genuinely tight and the system CUDA is 12.x; the script warns
if it is not.

---

## What gets installed, and why each version is forced

| Layer | Version | Where | Why this exact version |
|---|---|---|---|
| NVIDIA driver | *existing one reused* | system | Only installed with `--with-driver`; see [Open vs. proprietary](#open-vs-proprietary-kernel-modules) |
| conda | *existing one reused* | wherever it already is | Only installed with `--install-conda`, and only if none was found |
| Python | **3.11** | conda env `dtsegnet` | Supported by every pin below; 3.13 (Ubuntu 24.04's `python3`) is ahead of numba's support window |
| CUDA toolkit | **12.8.1** | *inside* the conda env | Must match torch's `+cu128` to compile `pointnet2_ops_lib` |
| gcc / g++ | **13** | *inside* the conda env | CUDA 12.8's `nvcc` accepts gcc ≤ 14 |
| torch | `2.8.0+cu128` | pip | Pinned by `requirements.txt`; first line with native **sm_120** SASS |
| numpy | `1.26.4` | pip | Last 1.x. Code was written against 1.24; keeps numpy-2 breakage off the table |
| numba | `0.61.2` | pip | Required by `models/patch_collate.py`; supports py3.11 and numpy 1.26 |
| lightning | `2.5.6` | pip | Needs torch ≥ 2.1; no longer touches `pkg_resources` |
| pointnet2_ops | built from source | pip, `--no-build-isolation` | Custom CUDA kernels; no wheel exists |

> **Migrated 2026-08-22** from `torch 2.1.0+cu121` / CUDA 12.1 / Python 3.10 / numpy 1.24.4 /
> numba 0.58.1. The trigger was deploying to 4× RTX PRO 6000 Blackwell (sm_120); see
> [GPUs newer than the pinned CUDA](#gpus-newer-than-the-pinned-cuda). The same migration also
> dropped 16 never-imported dependencies (torchvision/torchaudio/torchtext/torchdata, pandas,
> seaborn, mlxtend, nltk, jupyterlab, ipywidgets, streamlit, scikit-learn, scikit-image,
> opencv-python, Pillow, imgaug) — see the comment block in `requirements.txt`. The
> `setuptools<81` pin is gone with them, since it only existed for `lightning 2.1.0`.

### Why the CUDA toolkit is installed *inside* the conda environment

`pointnet2_ops_lib` is a `CUDAExtension` compiled from source, so `nvcc` must match the CUDA
version torch was built against (12.8) or the resulting `.so` is ABI-incompatible with torch.

Matching that exactly from system sources is unreliable: whatever CUDA the box happens to ship is
whatever the admins installed, and it is usually *newer* rather than equal. The Blackwell server
this was deployed to has CUDA **13.2** system-wide, a full major version ahead of torch's 12.8 —
which is precisely the case `--system-cuda` would get wrong.

The `nvidia` conda channel publishes exactly `12.8.1`, so the toolkit goes in the environment.
This is also strictly better for a shared server: no system-wide CUDA to conflict with other
users' projects, no `sudo` needed, and multiple CUDA versions can coexist in different envs.

The same reasoning applies to gcc-13: it is installed as a conda package
(`gcc_linux-64=13` / `gxx_linux-64=13`) rather than system-wide. Conda's activation scripts set
`CC` and `CXX` automatically, so `torch.utils.cpp_extension` picks it up with no extra config.

---

## Server-specific concerns

> The first two subsections apply **only if you pass `--with-driver`**, i.e. you are provisioning a
> bare machine yourself. On a server that already has a working driver, the script leaves it
> completely alone and neither applies — skip to
> [Do not let the training GPU drive a display](#do-not-let-the-training-gpu-drive-a-display).

### Secure Boot will silently break the driver

DKMS-built NVIDIA kernel modules are **unsigned**. With Secure Boot enabled the module refuses to
load after reboot, and the only symptom is `nvidia-smi` still failing — the apt install itself
reports success. Fixing it requires enrolling a Machine Owner Key at the **physical console**
(a password typed into a blue UEFI screen at boot), which **cannot be done over SSH**.

`setup_server.sh` detects this and refuses to continue rather than leaving you with a mystery.
Check it yourself with:

```bash
mokutil --sb-state
```

On a server, disabling Secure Boot in the BIOS/UEFI is usually the pragmatic answer. If policy
forbids that, enrol a MOK with `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` and
complete the enrolment at the console on the next boot.

### Open vs. proprietary kernel modules

The script prefers the `-server-open` driver branch, falling back to `-open`, then to whatever
`ubuntu-drivers` recommends:

- **`open`** — the open-source kernel modules. On recent kernels the proprietary module frequently
  fails to build; the open one is the reliable choice, and NVIDIA's default from r560 onward. It
  supports **Turing (RTX 20-series) and newer**. On Pascal/Maxwell you must use the proprietary
  module instead — pass the package name manually.
- **`server`** — the datacenter branch, which omits the desktop X/Wayland userspace you do not
  want on a headless box.

Verify DKMS actually *built* the module before rebooting — a failed build does not fail the apt
install:

```bash
sudo dkms status          # expect: nvidia/<version>, <your kernel>, x86_64: installed
```

If it is missing, you are probably lacking kernel headers:

```bash
sudo apt-get install linux-headers-$(uname -r)
```

### Do not let the training GPU drive a display

On a workstation where the GPU also runs the desktop, long CUDA kernels can be killed by the
driver's watchdog, surfacing as:

```
torch.AcceleratorError: CUDA error: the launch timed out and was terminated
```

A headless server does not have this problem, which is one good reason to train on one. Check
with `nvidia-smi --query-gpu=display_active --format=csv` — you want `Disabled`.

### GPUs newer than the pinned CUDA

CUDA 12.8 emits native code for **sm_50 through sm_120**, which covers Blackwell (RTX 50-series
and RTX PRO 6000 Blackwell are `sm_120`; B200 is `sm_100`). Anything *newer* than sm_120 would run
only through PTX JIT compilation at first launch — slow to start and not guaranteed to work.

`setup_server.sh` detects the compute capability, appends `+PTX` to `TORCH_CUDA_ARCH_LIST` for
forward compatibility, and warns loudly if the GPU is newer than `MAX_NATIVE_CAP` (12.0).

**What this failure actually looks like**, if you ever run a too-old CUDA pin on newer hardware:
the install succeeds, `import torch` succeeds, and `torch.cuda.is_available()` returns `True`.
It breaks at the first real kernel launch, because cuBLAS/cuDNN ship precompiled SASS with no
usable PTX fallback — so you get `CUBLAS_STATUS_NOT_SUPPORTED` or `no kernel image is available
for execution on the device` from an innocuous-looking matmul. That is why Phase 7 of
`setup_server.sh` runs an actual matmul and cross-checks `torch.cuda.get_arch_list()` against the
device capability, rather than trusting `is_available()`.

This bit the project for real on 2026-08-22: the cu121 pins were carried onto a 4× RTX PRO 6000
Blackwell server and had to be migrated to `torch 2.8.0+cu128` before anything could train. If you
hit the same wall on future hardware, the four things that must move together are
`requirements.txt`'s torch/numpy/numba pins and `setup_server.sh`'s `PY_VERSION`, `CUDA_VERSION`,
`GCC_VERSION`, `MIN_DRIVER` and `MAX_NATIVE_CAP` constants — plus the Phase 7 assertions, which
hard-code the expected versions.

---

## Verification

`setup_server.sh` ends with a self-check. To re-run it by hand:

```bash
conda activate dtsegnet
python -c "
import torch
from pointnet2_ops import pointnet2_utils as pu
print('torch      ', torch.__version__)
print('cuda avail ', torch.cuda.is_available())
print('device     ', torch.cuda.get_device_name(0))
xyz = torch.randn(2, 4096, 3, device='cuda').contiguous()
idx = pu.furthest_point_sample(xyz, 512)
torch.cuda.synchronize()
print('pointnet2 CUDA kernel OK:', tuple(idx.shape))
"
```

That last call matters more than `torch.cuda.is_available()`: it exercises the **custom CUDA
kernels compiled on this machine**. `is_available()` can return `True` on a box where the
extension was built for the wrong architecture and every real kernel launch fails.

Confirm the compiled architecture matches your GPU:

```bash
cuobjdump --list-elf $(python -c "import pointnet2_ops,glob,os;print(glob.glob(os.path.dirname(pointnet2_ops.__file__)+'/_ext*.so')[0])")
# expect e.g. sm_86 cubins for an RTX 3090
```

---

## Troubleshooting

Every error below was hit for real while building this environment.

**`ModuleNotFoundError: No module named 'numba'`**
`models/patch_collate.py` imports numba at module level. It was missing from `requirements.txt`
until 2026-08-20 — update your checkout, or `pip install numba==0.61.2`. Do **not** install an
unpinned numba: current releases resolve numpy past the pinned 1.26.4.

**`ModuleNotFoundError: No module named 'pkg_resources'` on `import lightning`**
Only affects the pre-2026-08-22 stack, where `lightning 2.1.0` called
`pkg_resources.declare_namespace()` at import and setuptools ≥ 82 had removed it. `lightning
2.5.6` does not, and the `setuptools<81` pin has been dropped. If you still see this, you are on
an old environment — rebuild it.

**`ModuleNotFoundError: No module named 'dataset.patch_dataset'`**
The `dataset/` package is missing from your working tree, not from the repo. This happened on the
Blackwell server on 2026-08-22 — an incomplete transfer left all 17 `dataset/*.py` files and
`train_patch_network_color.py` (the entry point every `run_experiment_matrix_gpu*.sh` calls)
deleted locally while still tracked in git. Check with `git status` and fix with:
```bash
git restore dataset/ train_patch_network_color.py
```

**`ModuleNotFoundError: No module named 'torch'` while building pointnet2_ops**
PEP 517 builds in an isolated environment, but `pointnet2_ops_lib/setup.py` imports torch at module
level. Build with isolation disabled:
```bash
pip install --no-build-isolation ./pointnet2_ops_lib
```

**`unsupported GNU version! gcc versions later than 14 are not supported`**
A too-new system gcc is being used instead of the env's gcc-13. Confirm `echo $CC` points into the conda
env; if empty, re-activate the environment (`conda deactivate && conda activate dtsegnet`).

**`nvidia-smi: command not found` after installing the driver**
You have not rebooted, or the DKMS build failed, or Secure Boot is blocking the module. Check in
that order: `uptime`, `sudo dkms status`, `mokutil --sb-state`.

**`torch.cuda.is_available()` is False, but `nvidia-smi` works fine**
Most often the machine's driver is too old for the CUDA 12.8 runtime. torch 2.8.0+cu128 needs
driver **>= 570.26** (the floor for sm_120 support at all); newer is always fine, older cannot initialise CUDA at all. Check with
`nvidia-smi --query-gpu=driver_version --format=csv,noheader` — `setup_server.sh` warns about this
during Phase 1. Updating a driver is a system change, so on a server you do not administer this is
a conversation with whoever does, not something to force. The other common cause is that the
process cannot see the GPU at all (container without `--gpus all`, or `CUDA_VISIBLE_DEVICES` set
to an empty or wrong value).

**torch reports an unexpected CUDA version, or CUDA libraries fail to load**
A system-wide `LD_LIBRARY_PATH` pointing at another CUDA can shadow the environment's own 12.8
libraries. `setup_server.sh` warns when it sees this but deliberately does not change it — it is
usually set by a site-wide profile or an environment module. Unset it for the training shell:
```bash
env -u LD_LIBRARY_PATH python train_patch_network_color.py ...
```

**conda asks you to accept Anaconda Terms of Service**
The `defaults` channel carries commercial licensing conditions. Every conda command in
`setup_server.sh` passes `--override-channels -c conda-forge` to avoid it entirely. An automated
installer must not accept a licence on your organisation's behalf — if you *want* `defaults`,
accept it deliberately with `conda tos accept`.

**`pip check` reports `ninja 1.11.1.1 is not supported on this platform`**
Cosmetic defect in ninja's own wheel metadata; the shipped binary works. Do not gate automation on
a clean `pip check` because of it.

---

## The Docker path

The repo's [Dockerfile](../Dockerfile) is an alternative to `setup_server.sh` and skips the conda
layer entirely — it starts `FROM nvidia/cuda:12.8.1-devel-ubuntu22.04`, so Python 3.11, CUDA 12.8
and a compatible gcc all come from the base image. (That base image was bumped from
`12.1.0-devel-ubuntu20.04` on 2026-08-22 with the rest of the Blackwell migration: it consumes the
same `requirements.txt`, so a stale cu121 base under a cu128 torch would fail at the first kernel
launch. It also installs `jupyterlab` itself now, since the dependency prune removed it from
`requirements.txt` and this image's default `CMD` is `jupyter lab`.) The **host still needs an NVIDIA driver** (and
the NVIDIA Container Toolkit, for `--gpus all`); only the toolkit and userspace move into the
image. Phase 1 of `setup_server.sh` is therefore still relevant on a Docker host — the rest is not.

Two things the Dockerfile does not do:

- It does **not** build `pointnet2_ops_lib`. The upstream README has you do that by hand inside the
  running container. That is not an oversight to "fix" casually: a `docker build` has no GPU
  visible, so the extension cannot detect an architecture and you must set
  `TORCH_CUDA_ARCH_LIST` explicitly at build time (e.g. `8.6+PTX`) — which bakes a
  GPU-specific assumption into the image. Building it at container start instead keeps the image
  portable at the cost of a slow first run.
- Its separate `pip install ninja` (line 38) is now redundant, since `ninja` is pinned in
  `requirements.txt`. Harmless, just no longer necessary.

---

## Performance notes

Measured on the 12-core / RTX 3090 box, training `train_patch_network_color.py`:

- **The pipeline is CPU-bound, not GPU-bound.** GPU utilisation averaged **~21%** while load
  average sat at **12–21 on 12 cores**. The bottleneck is the KD-tree gather + FPS in
  `models/patch_collate.py`, which runs in the dataloader workers on CPU.
- **Throughput scales with core count, not GPU class.** The same config runs ~4× faster on a
  24-core machine. When specifying a server for this workload, **buy cores**.
- **VRAM is not a constraint**: peak 7.0 GB of 24 GB at `--max_faces 20000`, batch size 1.
- `num_workers` 5 vs 8 made no measurable difference on 12 cores (1.40 vs 1.33 s/batch, within
  run-to-run noise — patch sizes are random, so short benchmarks vary a lot).
- Note `persistent_workers=True`: `num_workers=8` keeps **16** worker processes alive, because the
  train and val dataloaders each hold their own pool.
- `train_patch_network_color.py` caps each worker to one internal thread via env vars set before
  the numpy/torch import — including `NUMBA_NUM_THREADS`, which is a separate pool that
  `torch.set_num_threads()` does not touch. See the comment block at the top of that file; do not
  remove it, it is worth a large factor on many-core machines.

Long runs should go in `tmux`/`screen` — a dropped SSH session otherwise kills training with no
error in the log, which is exactly how one previous run died mid-epoch.
