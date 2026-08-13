# Claude Code task — Adapt DilatedToothSegNet for patch-based robotic IOS segmentation (training pipeline)

## Context
This repo (`dilated_tooth_seg_net`) trains DilatedToothSegNet on Teeth3DS to segment **full** dental
arches, one shot, per triangle face. I'm repurposing it as the perception model for a robotic intraoral
scanner: at inference it will run repeatedly on a **partial, growing** arch mesh that accumulates as the
robot sweeps. This task covers the **offline training pipeline only** — not live inference/robot integration.

Key measured facts about the target (live) data, so the training data matches it:
- Live meshes are indexed triangle meshes in **millimeters**, with per-vertex unit normals. Same kind of
  data as Teeth3DS (both from intraoral scanners), ~74 faces/mm² native.
- At inference the live arch is downsampled to a **working density W = 5 faces/mm²** (~16k faces per full
  arch). Training data MUST be at this same density.
- The model will see contiguous partial arches that grow from a front-tooth seed backward, occlusal
  surfaces first, then buccal/lingual — starting from ~one tooth up to the full arch.

Please work on a new git branch, keep changes modular and config-driven, and DON'T delete the existing
full-arch path — add the new path alongside it (a config flag selects patch-mode). The local edge-conv
backbone is good and should be preserved; we're changing the data pipeline, normalization, norm layers,
the STN, the dilation blocks, and making N variable.

Proceed one component at a time; after each, show me the full changed file(s) so I can paste them, and
tell me what to run and what output you expect. Don't hand me a giant diff all at once.

## Component 1 — Teeth3DS downsample to working density
- Add a preprocessing step that downsamples each full Teeth3DS arch to **density W = 5 faces/mm²**
  (target face count = round(surface_area_mm² × 5)), using the repo's existing quadric simplification
  (pyfqmr), transferring per-face labels by nearest-face as the repo already does.
- Replace the hard-coded fixed-16,000-face target with this **density-based** target. Cache the
  downsampled labeled arches.

## Component 2 — Patch generator (the core new piece)
Generate partial-arch patches from the downsampled arches. Make it a `Dataset` that samples patches
on-the-fly per epoch (with a fixed, seeded val set for reproducibility).

Two axes of "partialness":
1. **Extent (position):** pick a seed face on a FRONT tooth (central/lateral incisor or canine region).
   Grow a **contiguous** region by breadth-first flood-fill over face adjacency, biased mesio-distally
   (along the arch, both directions from the seed). Boundaries may cut ACROSS teeth — do NOT snap to whole
   teeth. Sample target extent uniformly-ish in area from ~one tooth (~150–250 mm² ≈ ~1k faces) up to the
   full arch.
2. **Coverage (occlusal-first):** estimate the arch's occlusal axis via PCA of vertices (smallest-variance
   axis ≈ occlusal-plane normal; sign it so it points toward the occlusal/scanner side). Give each face an
   "occlusality" = dot(face_normal, occlusal_axis). A coverage threshold `c` keeps only faces with
   occlusality ≥ threshold (low c = occlusal caps only; raise c to admit buccal/lingual walls).
- Combine: patch = (extent region) ∩ (faces passing coverage), then keep the **largest connected
  component** so it stays contiguous.
- **Sampling:** broad over (extent, coverage), **weighted toward occlusal-heavy/early** states (small
  extent + low coverage should be common, since that's what the model sees most).
- **Labels:** carry per-face labels; support BOTH label sets via a config flag:
  - `num_classes=17` — the existing FDI mapping.
  - `num_classes=5` — coarse: {incisor, canine, premolar, molar, gum}. Add this remap in `teeth_numbering.py`.
- Include a small visualization script that dumps a few generated patches (colored by label) to PLY/PNG so
  I can eyeball them before training.

## Component 3 — Patch-invariant normalization (replaces the whole-mesh z-score)
In the feature builder (currently `PreTransform`):
- Center each patch by its **own centroid** (subtract mean position).
- Divide coordinates (vertices + face center) by a **FIXED constant** `SCALE_MM` (config, default ~15 mm) —
  NOT the per-patch std/min/max. This makes the same tooth land at the same scale regardless of patch size.
- Leave **normals raw** (already unit vectors); do NOT z-score them.
- Apply this identical scheme at train and (later) inference. Keep the mm-based scale so it's consistent
  with the live pipeline.

## Component 4 — Model surgery (`models/layer.py`, `models/dilated_tooth_seg_network.py`)
- **Norm layers:** replace all `BatchNorm` with `GroupNorm` (or `LayerNorm`) so normalization is per-point and independent of batch/patch composition.
- **STN:** remove `STNkd` (the Spatial Transformer / T-Net) from the forward path.
- **Variable N:** remove the pad/resample-to-16,000 assumption; support variable face count per sample. Batch via `batch_size=1`, or masked padding / size-bucketing for larger batches — masked points must beexcluded from kNN, FPS, pooling, and loss.
- **Area-gated dilation:** make the three dilated blocks engage based on the patch's accumulated **area in mm²** (thresholds ~40 / ~180 / ~360 mm² for blocks 1/2/3). Below a block's threshold, bypass it while keeping the concatenation into `global_hidden_layer` valid (e.g. pass the identity/zero-filled contribution so channel counts stay fixed). Make thresholds and block count config-driven.
- Parameterize `num_classes` everywhere (network, Lightning module, torchmetrics) — currently hard-coded 17.

## Component 5 — Loss, metrics, experiment matrix
- **Loss:** class-weighted cross-entropy (inverse-frequency) or focal loss (config flag), to handle the varying gum/tooth ratio across patches.
- **Metrics:** patch-aware — ignore classes absent from a patch when computing mIoU; log per-class where possible; add a boundary/edge accuracy metric if easy.
- **Experiment flags** (config/CLI): `num_classes ∈ {17,5}`, `dilation_gating ∈ {on, off}`, norm scheme. I want to run and compare these.

## Deliverables
- New branch, modular config-driven changes, existing full-arch path preserved behind a flag.
- The patch-visualization script.
- A short README section on the new config flags and how to run training for each experiment.
- Sanity checks / minimal tests: patch generator produces contiguous patches at ~5 faces/mm² with valid
  labels; model forward pass runs on variable N at batch_size=1; area-gating switches blocks correctly.

## Working style
Go step by step, one component at a time. After each, give me the FULL updated file(s) to copy-paste (I
won't apply diffs myself), tell me exactly what command to run, and ask me for the output before moving on.
