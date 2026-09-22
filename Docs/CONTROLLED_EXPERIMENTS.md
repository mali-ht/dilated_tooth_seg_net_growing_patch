# Controlled tooth-training experiments

Status: implementation, launchers and the frozen v1 manifest are prepared; no experiment training has been started.

The pre-experiment work was preserved on `main` at `c9d7152`. New work lives on
`codex/controlled-tooth-training-experiments`. The existing training entry points,
models, live guidance and historical data split remain unchanged. This is a separate
versioned experiment path under `experiments/controlled/`.

## Run count and decision gates

Start with **four training runs**, seed 42. Do not launch architectural or midline
sweeps until these results are reviewed.

| Run | Focal-loss alpha | Patch sampling |
|---|---|---|
| A_inverse_sweep | Fixed inverse frequency | Ordinary scan sweep |
| B_sqrt_sweep | Fixed square-root inverse frequency | Ordinary scan sweep |
| C_inverse_targeted | Same vector as A | 60% sweep, 20% adjacent-pair footprints, 20% terminal-region starts |
| D_sqrt_targeted | Same vector as B | Same mixture as C |

Manifest ID: `c5b92ea292ac29728f309e8282ea78663d62d7cbe4b5cc0cd30eb1665d471233`.

The proposed 60/20/20 sampling mixture is an experimental starting point, not a
previously validated optimum. Pair candidates are 4/5, 12/13, 1/2, 2/3, 14/15,
15/16. Pair crops use geometric footprints; they are not cut out at ground-truth
tooth boundaries. When eligible teeth are missing, the sample falls back to sweep
and records its actual mode. Terminal-region starts include true terminal teeth
and their neighbors; absence/visibility is determined from actual labels, not
assumed from the requested mode. Per-sample metadata exposes these distinctions.

Shared recipe: 17 classes, early-global transformer, HSV-synthesized RGB,
20/90/180 mm² area thresholds, gating on, power 1.5, gamma 2, 20,000-face cap,
32-bit precision, and no dedicated midline or whole-tooth training branch.
Dedicated midline examples ARE included in fixed validation.

Each run uses **200 epochs × 1,200 sampled examples = 240,000 optimizer updates**,
batch size 1, Adam at 0.001. The learning rate halves at steps 72,000, 144,000 and
216,000. These are virtual sampling epochs, not one pass over every training arch.
Validation runs every five epochs, including the final epoch. Changing budget
arguments is supported for debugging, but it does not change LR milestones;
use the documented defaults for the actual comparison.

After selecting two finalists, repeat each with seeds 43 and 44: **four additional
runs, eight total**. Reuse the same manifest and alpha vectors. Later midline and
architecture experiments are separate decisions; no launchers automatically start
them. Boundary/instance heads, positional attention and live guidance changes are
not bundled into this first loss/sampling comparison.

## Exactly seven new shell scripts

All are in `training_launch/controlled_experiments/`; existing launchers remain available.

1. `run_controlled_A_inverse_sweep.sh`
2. `run_controlled_B_sqrt_sweep.sh`
3. `run_controlled_C_inverse_targeted.sh`
4. `run_controlled_D_sqrt_targeted.sh`
5. `run_controlled.sh` — shared implementation, terminal progress bar and log capture
6. `run_controlled_local.sh` — all four sequentially, foreground
7. `run_controlled_server.sh` — one serial queue per selected GPU; parent stays attached

The server launcher parallelizes across independent runs, not across GPUs within
one model. With four GPUs it runs one recipe on each; with two it runs two waves.
Use GPU ordinals visible to the process, accounting for `CUDA_VISIBLE_DEVICES`.
Do not run the same recipe/seed concurrently from two terminals.

## Local versus server

The last scratch run took approximately **22.8 hours between its first and final
validation events**, about 412 seconds per epoch, on the local RTX 5090. A simple
four-run reference is **about 92 GPU-hours**. This is historical extrapolation,
not a benchmark of the new pipeline. Smaller patient-separated validation cases
are offset by multiple evaluation crops, and targeted sampling changes patch sizes.

If server GPUs perform similarly: one GPU means roughly four run durations, two
GPUs roughly two, and four GPUs roughly one. Server hardware, VRAM and CPU throughput
must be checked before treating this as a schedule. The existing local GPU has
32 GB VRAM; smaller cards may require profiling. Do not change face caps/precision
for only some recipes, because that changes the experiment.

Each concurrent run defaults to eight workers (training plus validation worker
pools can coexist), with internal numerical-library threads capped at one. Budget
CPU/RAM per active GPU and pass `--workers` if needed. The current processed cache
contains 1,800 files totaling approximately **7.25 GiB**. Copy it without reprocessing.
The frozen v1 manifest is checked into `experiments/controlled/manifests/v1/`.
Checkpoints and metrics go under git-ignored `experiment_artifacts/`;
console logs go under `logs/experiments/controlled/`.

## Shared preparation and server verification

The v1 manifest has already been generated locally from all 1,800 cached arches
and 2,048 training samples, and is included in this branch. On the server, copy
the unchanged processed cache, then run from the repository root:

```bash
conda activate MAA
python3 -m experiments.controlled.prepare --verify
```

To create a *different* version for future experiments (not needed for v1):

```bash
python3 -m experiments.controlled.prepare --out experiment_artifacts/controlled_v2
```

The preparation command is CPU-only. It hashes the cache, groups matching upper/lower filename case
IDs, creates deterministic 70/15/15 train/validation/test splits, and estimates
frequency counts from 2,048 reproducible TRAIN samples. It uses the ordinary sweep
distribution to produce one common inverse vector and one common softened vector.
No model is trained. Preparation fails if a class is entirely unobserved; increase
the frequency sample count instead of inventing an extreme weight.

For the current 900 paired IDs, the expected split is 630/135/135 IDs (1,260/270/270
arches). Confirm that filename IDs correspond to patient identity in the dataset
provenance. If patient metadata groups multiple IDs together, rebuild using those
groups before claiming a patient-independent test.

The manifest records actual filenames, hashes, sample seeds, alpha counts and
vectors, and preprocessing-code fingerprint. It cannot overwrite an existing
directory. The server must use the SAME checked-in
`experiments/controlled/manifests/v1/manifest.json`, not an independently generated
file, plus the exact processed cache. Verification can also be rerun locally:

```bash
python3 -m experiments.controlled.prepare --verify
```

Training also validates every cache hash. Different cache availability cannot
silently shrink either split. Missing files or changed preprocessing fail clearly.

## Fixed evaluation

Every validation/test arch has five geometric modes: ordinary sweep, first-stage
early patch, midline, adjacent posterior pair, terminal-region start. Each crop is
evaluated with full synthetic color and 60% synthetic color dropout: ten records
per arch, expected 2,700 validation records. The two color conditions share exact
geometry, selected faces and labels. They do not depend on a training recipe.

All modes are reproducible from immutable record seeds, code fingerprint and
hashed cached meshes. Inputs are reconstructed rather than saved as duplicate
mesh tensors. Dataset-library versions and numerical backends should be matched
between machines; bitwise equivalence across arbitrary GPU/software stacks is
not promised. Keep the test suite reserved until recipes/thresholds are selected.

To perform stage-0 comparison of existing checkpoints, supply explicit paths:

```bash
python3 -m experiments.controlled.evaluate \
  --checkpoints /absolute/path/to/model_A.ckpt /absolute/path/to/model_B.ckpt \
  --output experiment_artifacts/evaluation/legacy_comparison
```

This performs inference only, sequentially on GPU 0. `--device`, `--workers`,
`--split val|test`, and debug-only `--limit` are available. Results from `--limit`
are explicitly marked partial. Start with the early transformer, midline
continuation, recent scratch model, gating-off model and whole-tooth model.
The evaluator detects the existing dilated/early/deep architectures and neighbor
sizes from their saved hyperparameters. Color models all see the same chosen
synthetic RGB condition; geometry-only models see the same crops without RGB.

**New random splits are not automatically unseen by historical models.** These
models may already have trained on cases now assigned to validation/test. A
retrospective score must not be presented as a clean held-out generalization score.
For truly unseen historical comparisons, verify old training manifests or collect
new independent cases. The new four runs use the new split consistently.

Each training epoch also writes actual class-exposure and requested/actual sampling-mode
counts, so fallback frequency and posterior underexposure can be inspected.

Outputs include per-class precision/recall/IoU and confusion counts; upper/lower,
area, mode, color and terminal absent/not-visible/visible strata; false identities per patch; false terminal-label
physical area when absent; and an adjacent-pair merge proxy. Absent-terminal
denominators and pair-trial counts are preserved. Physical area after random face
capping is an unbiased sampling estimate, not exact full-mesh area.

The pair-merge proxy asks whether two visible GT teeth (at least 20 retained faces
each) share the same dominant predicted non-gum label. It is NOT full connected
instance matching, boundary F1, or a robot-completion metric. Real scan annotations,
registered-surface flicker, confidence calibration, and robot detection-delay/
completion evaluation remain necessary. This synthetic suite does not claim to
measure them or substitute for real scanner color.

## Inspect commands before launching

```bash
bash training_launch/controlled_experiments/run_controlled_local.sh --dry-run
bash training_launch/controlled_experiments/run_controlled_server.sh 0,1,2,3 --dry-run
```

These print all four recipes, devices, manifest path and update budgets. They do
not prepare data or start training. One individual run, when ready:

```bash
bash training_launch/controlled_experiments/run_controlled_A_inverse_sweep.sh
```

All four locally, or all four distributed across selected server GPUs:

```bash
bash training_launch/controlled_experiments/run_controlled_local.sh
bash training_launch/controlled_experiments/run_controlled_server.sh 0,1,2,3 --workers 8
```

The user launches these commands in their terminal. There is no automatic tmux,
nohup, scheduled task or background training service.

## Checkpoints and reproducibility

Patches depend explicitly on run seed, epoch and sample index, not worker timing.
The collator uses a geometry-derived neighbor seed. The final growth stage is
reachable. Model RNG state is saved/restored along with optimizer/scheduler state.
Sampler epoch keys handle Lightning's initial prefetch during checkpoint resume.

Two independent checkpoint callbacks save the best validation model and current
epoch `last.ckpt`. The unmonitored last writer saves current model state every
epoch even if validation did not run or improve. The final epoch and optimizer
step count are verified after successful completion.

An interrupted epoch restarts from the last COMPLETED saved epoch. To resume A:

```bash
bash training_launch/controlled_experiments/run_controlled_A_inverse_sweep.sh --resume
```

Resume requires matching manifest, seed, full budget, validation interval, alpha
and code fingerprint. Existing output directories are rejected without explicit
resume. Do not use `--resume` on the whole matrix if some runs are already complete;
resume only the incomplete individual recipes. Best checkpoints are for evaluation;
training resume uses the verified `last.ckpt`.

Before freezing a full experiment, run the included small CPU verification:

```bash
python3 -m pytest testing/test_controlled_experiments.py -q
```

These tests include a tiny scalar model to verify save/resume mechanics, not a
tooth-training job. They also check paired-jaw splits, final-stage reachability,
weight softening, failure metrics, and identical real-patch geometry across color
conditions and worker counts. Full production-scale GPU throughput and the proposed mixture's
accuracy still require execution on the selected machine.
