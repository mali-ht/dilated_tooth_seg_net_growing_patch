# Engineering concerns before training (living doc)

Raised during a design-review checkpoint after Components 1-3 (density-based downsampling,
occlusal/facial/lingual patch generator, patch-invariant normalization), before starting
Component 4 (model surgery) and Component 5 (loss/metrics). Companion to `MAA_ToDo.md` - that
doc is the high-level brainstorm/plan; this one tracks specific risks, the reasoning behind them,
and what we decided to do about each. Cross-referenced from `MAA_ToDo.md`'s TODO list (items 17,
18) and "Open design questions" (coverage-axis trajectory question).

## 1. O(N²) `cdist`, and a hard N>=33 floor - RESOLVED in Component 4

Refined after reading the DilatedToothSegNet paper itself alongside `models/layer.py` line by
line (paper: `9_DilatedToothSegNet_...pdf`, in this directory).

**There are 4 separate O(N^2) pairwise-distance computations per forward pass, not 1.** The paper
is explicit that the 3 local edge-conv layers each use a DYNAMIC kNN graph "recomputed after each
layer based on the learned feature space" - confirmed in code: `edge_graph_conv_block1` gets the
true metric `pos` explicitly, but `edge_graph_conv_block2`/`3` are called with only `x1`/`x2` (no
`pos` arg), so inside `EdgeGraphConvBlock.forward`, `pos = x` falls back to that block's own
INPUT FEATURES - each one calls `knn()` -> its own fresh `torch.cdist` in feature space. On top
of those 3, `DilatedToothSegmentationNetwork.forward()` computes a 4th, `cd = torch.cdist(pos,
pos)` on the true metric position, once upfront, shared by the 3 dilated blocks. Area-gating
(`MAA_ToDo.md` TODO #8) decides whether a block *uses* a matrix - it does not skip *building* any
of the 4. Our patches range from a few hundred faces up to 40,000+ (full-arch + facial + lingual
walls); at 40k faces the metric-space one alone is a 40,000 x 40,000 float32 matrix (~6.4GB), and
the 3 feature-space ones are the same N^2 shape at 24-48 channels of context each.

**The paper's own mitigation doesn't transfer to us.** Table 5 shows precomputing the dilated
FPS/kNN indices as a PREPROCESSING step (since their pipeline decimates every mesh to a fixed
16,000 faces once and reuses it every epoch) cuts a 3-layer forward pass from 0.1291s to 0.026s -
about 5x. Our patches are regenerated fresh (different faces, different N, different geometry)
on essentially every `__getitem__` call via the scan-sweep + augmentation randomness - there is no
static mesh to precompute indices against, so this specific optimization is not available to us;
whatever we do needs to be cheap to compute ad hoc, every draw.

**New, empirically confirmed finding: the unmodified model hard-crashes for any patch with N<=32
faces.** `EdgeGraphConvBlock`'s `knn()` calls `pairwise_distance.topk(k=k+1, ...)` with the fixed
`k=32` - this requires at least 33 points to exist at all (self + 32 neighbors) and PyTorch's
`topk` raises `RuntimeError: selected index k out of range` below that, regardless of device.
Verified directly on the actual GPU (RTX 5090, CUDA): a random (1, N, 24)/(1, N, 3) forward pass
succeeds for N=33 and above, fails for N=32, 24, 10. This is not a tail-case risk - our own
`test_patch_dataset.py` runs have drawn faces counts like 24, 54, 124, 184 (the whole point of
`early_bias_power` is to bias toward exactly these small/early patches), so the unmodified model
would crash on a meaningful fraction of real training draws. `DilatedEdgeGraphConvBlock` already
guards its OWN dilation neighborhood with `dilation_k = min(self.dilation_k, N)`, so the fixed
`dilation_k` values (200/900/1800 - the paper's own values, from its fixed N=16,000 setting)
degrade gracefully on small N; `EdgeGraphConvBlock`'s plain `k=32` has no equivalent guard.

**Correction (later session, see REALTIME.md):** this line previously read "200/600/1800 -
tuned for the paper's fixed N=16,000 setting" - factually wrong. The actual paper/reference file
(`models/dilated_tooth_seg_network.py:31-39`, untouched) uses 900 for the middle dilated block,
not 600. 600 had silently become the default in `models/patch_dilated_tooth_seg_network.py` (and
propagated into `patch_lightning_module.py`/`patch_collate.py`'s defaults) with no design
rationale anywhere in this project's docs - `claude_code_prompt.md`'s Component 4 spec never asked
for a dilation_k change, only norm/STN/variable-N/area-gating - and this very line then
misdescribed that drift as intentional. Restored to 900 everywhere; this line corrected rather
than left showing the wrong value it was itself part of propagating.

**Status: RESOLVED as part of Component 4.** Both problems fixed:

- **Small-N floor**: `models/patch_layer.py`'s `knn()` clamps `k = min(k, N-1)`; verified on the
  real GPU that the new network runs without crashing all the way down to N=2 (old network
  crashed at N<=32, confirmed reproduced first, then fixed).
- **O(N^2) cost**: `models/patch_collate.py` precomputes 2 of the 4 cdist sites (block 1's local
  kNN + all 3 dilated blocks' candidate-gather-and-FPS, all a pure function of metric position) on
  CPU via a single shared `scipy.spatial.cKDTree` query (a k-nearest query sorted by distance
  already contains every smaller neighborhood as a prefix, so one query serves all 4 of these
  needs) plus a numba-JIT'd batched farthest-point-sampling kernel. Verified the precomputed
  indices are EXACTLY identical (not just "close") to the on-the-fly cdist+topk/FPS result, at
  every N tested. The remaining 2 sites (local blocks 2/3's dynamic feature-space kNN) are left
  as on-the-fly dense `cdist` - they can't be precomputed (depend on the network's own mid-forward
  activations) but are cheap in absolute terms (small feature dims, only 2 of the original 4).

  Performance note along the way: a first version of the CPU precompute (plain per-face Python
  loop, then a numpy-vectorized-but-still-per-block version) measured ~20-35s to collate a single
  large (~25-30k face) patch - way too slow. Root cause was FPS, not the KD-tree query: a
  numpy-vectorized FPS iteration allocates a fresh large temporary array every one of its k=32
  iterations (memory-bandwidth-bound). A numba-JIT'd, `prange`-parallelized FPS that fuses the
  whole per-point distance+argmax into one pass with no large temporaries measured ~90-100x
  faster (0.04s vs 3.6s at N=5083, m=1800, k=32). numba was already installed, is a pure CPU JIT
  compiler (no CUDA-architecture-compatibility risk, unlike the torch_cluster option that was
  considered and abandoned - see below). End-to-end result: even the worst realistic case (a
  ~34,000-face full occlusal+facial+lingual patch) collates in ~1.6-1.7s; typical patches (a few
  hundred to a few thousand faces, per `early_bias_power`'s bias toward small/early stages)
  collate in well under 0.1-0.5s.

  `torch_cluster` (proper CUDA kNN/radius-graph kernels, would also cover the 2 remaining dynamic
  sites) was considered - no prebuilt wheel exists for torch 2.12.1+cu132, and a from-source build
  attempt hit a >5-minute CUDA compile with real risk of silently lacking sm_120 (RTX 5090/
  Blackwell) support even if it finished, similar to the vanilla-PyTorch sm_120 issue in this
  project's history. Abandoned in favor of the CPU KD-tree + numba path above. Worth revisiting
  later, as a deliberate time-boxed task, only if profiling shows blocks 2/3's dense cdist is a
  real bottleneck at realistic patch sizes - not observed so far.

## 2. Sequential per-arch training batches were a bad idea - RESOLVED

We had briefly built `PatchTeeth3DSDataset.__getitem__` to return one arch's ENTIRE ~40-stage
sweep (occlusal -> facial -> lingual) as "one batch," meant to be fed as ~40 consecutive
forward/backward passes before moving to the next arch. Problem: consecutive stages differ by one
tooth's worth of faces - about as far from i.i.d. as SGD steps get. Optimizer momentum/Adam state
drags along a slow-moving trajectory within an arch, then jumps to a new arch; risk of getting
stuck in local minima from the correlated updates, per feedback from Mohamed.

**Decision: reverted to randomized sampling.** `__getitem__(index)` again returns ONE random patch
(a single (pos, x, labels) triple, weighted toward small/early stages) - `index` selects the arch,
but the specific stage/size drawn is re-randomized every call in train mode. With
`DataLoader(..., shuffle=True)`, consecutive training steps now pull different patch sizes from
different arches, interleaved - much closer to standard SGD assumptions. The full-sequence
capability wasn't thrown away - it's preserved as an explicit `get_full_sequence(index)` method
for visualization/debugging, just no longer the default training path.

## 3. The occlusal->facial->lingual curriculum may not match the real robot policy

The three-phase curriculum (sweep occlusal fully, then facial in one continuous pass, then
lingual) is a hypothesis about what the robot will actually do - not a confirmed trajectory.
`MAA_ToDo.md`'s own "Open design questions" already flagged this as TBD (independent (E,c)
sampling vs. ordered occlusal-sweep-then-sides trajectories), and the robot's actual motion
policy hasn't been designed yet - per Mohamed, it'll be a vision-guided motion routine developed
*based on* this model's performance, not a fixed target to train against, and finalizing it isn't
blocking.

**Decision: accepted as an open risk, mitigated (not resolved) by decision #2.** A model trained
on a broad, randomized distribution of partial-arch states is inherently more robust to whatever
trajectory the eventual policy takes than one trained on a single rigid hand-designed curriculum -
so the randomization fix above is doing double duty here. Plan is to iterate empirically: train,
observe what scanning strategy actually works well given the model's real performance, adjust.
Not going to gate progress on nailing the policy down first.

## 4. Facial/lingual phases contribute relatively little coverage

Measured on one test arch: occlusal-complete = 10,491 faces; facial phase added only 1,012 more;
lingual added only 502 more - together under 15% on top of occlusal alone. Worth confirming this
isn't just adding complexity for marginal benefit.

**Status: measured.** Swept `min_occlusality` on one test arch (facial/lingual shown as % of
occlusal's own face count, added on top):

| min_occlusality | occlusal faces | +facial | +lingual |
|---|---|---|---|
| 0.2 (old default) | 10,491 | +9.6% | +4.8% |
| 0.4 | 6,374 | +32.2% | +20.6% |
| 0.6 | 3,513 | +76.0% | +55.7% |
| 0.8 | 1,780 | +71.9% | +55.7% |

Confirms the hypothesis: `min_occlusality` was restricting BOTH the occlusal phase's own capture
and how much wall was left for facial/lingual to claim (occlusal's generous 0.2 threshold already
grabbed most of the upper wall before facial/lingual ever ran). 0.6 looks like a good candidate -
facial/lingual become genuinely substantial contributions rather than an afterthought - without
occlusal itself getting too sparse to be meaningful (0.8's 1,780 faces across a whole arch's
occlusal-complete state seems too thin).

**Decision: 0.6 is now the fixed default everywhere** (`rectangular_footprint`,
`grow_patch_full_sweep`, the visualization/GUI script `--min_occlusality` CLI defaults, and
`PatchTeeth3DSDataset.min_occlusality`). On top of that, `PatchTeeth3DSDataset.__getitem__` now
also RANDOMIZES `min_occlusality` per draw in train mode, uniformly over
`min_occlusality_range=(0.2, 0.8)` (per Mohamed - domain randomization over how "top-down vs.
tilted" the simulated scan angle is, on the theory that training across the whole measured range
rather than one fixed value makes the model more robust, similar in spirit to #3's argument for
randomized patch sampling generally). `get_full_sequence` deliberately does NOT randomize - it
always uses the fixed 0.6 value, so a single call still returns one consistent, inspectable
curriculum for visualization/debugging rather than a different threshold per stage. In eval mode
the draw is seeded per-index (like everything else - see `_rng_for`), so validation patches stay
reproducible across epochs despite the randomization being enabled by default.

Side note from re-testing this change: bumping the default to 0.6 exposed a stale test
assumption in `testing/test_patch_generator.py` - it had asserted the WHOLE accumulated
tooth-sweep mask stays one connected component at every stage, which only happened to hold at
the old, permissive 0.2 default (patches were large enough that neighboring teeth's occlusal
footprints touched). At 0.6, adjacent teeth's occlusal-only captures can legitimately stop a
couple mm short of the interproximal contact area between them - confirmed by direct inspection
(two real, valid single-tooth footprints, 1.3mm apart, not touching) - not a regression of the
isolated-island fix. Test updated: dropped the whole-mask connectivity assertion for tooth-sweep
stages (not a real invariant of that code path) and added a direct, isolation-tested check that
`rectangular_footprint` still returns one connected component per tooth, independent of any
accumulated mask.

## 5. Confirmed must-do: class-weighted/focal loss, sim-to-real augmentation - BUILT

Both already on `MAA_ToDo.md`'s TODO list (#14, #18) and confirmed by Mohamed as required before
real training, not optional polish. Both now built, tested (`testing/test_patch_losses.py`,
`testing/test_patch_augmentation.py`) and visually verified (`testing/visualize_patch_augmentation.py`).

- **Loss weighting**: `dataset/patch_losses.py`'s `FocalLoss` - gamma-focusing focal loss
  (Lin et al. 2017), same (logits (B,C,N), targets (B,N) long) calling convention as
  `nn.CrossEntropyLoss` so it drops into `models/dilated_tooth_seg_network.py`'s
  `self.loss = nn.CrossEntropyLoss()` pattern unchanged whenever a patch-training Lightning
  module gets built on top (that file itself stays untouched). `gamma=0, alpha=None` reduces
  exactly to plain CE (tested). `compute_class_alpha(dataset, num_classes, n_samples=...)`
  measures real inverse-class-frequency weights from the ACTUAL patch-sampling distribution (not
  whole-arch frequency, which would understate how gum-heavy a typical small/early patch is) -
  on a 30-patch sample, gum (class 0) got the smallest alpha as expected, confirming the
  60-70%-gum concern is real and the weighting responds to it correctly.
- **Sim-to-real augmentation**: `dataset/patch_augmentation.py`'s `SimToRealAugment`, composing
  5 independent perturbations, each randomized per draw: `punch_holes` (0-2 small connected
  face-dropout blobs, BFS-grown, capped at 15% of the patch), `erode_boundary` (randomly drops up
  to 30% of boundary-only faces for a ragged edge instead of our generator's clean rectangular
  cut), `density_subsample` (uniform random thinning to 70-100% keep-fraction), `jitter_positions`
  (per-vertex Gaussian noise up to 0.15mm, shared vertices get identical jitter so the patch stays
  watertight instead of tearing at seams), `jitter_normals` (small per-vertex/per-face normal
  perturbation, renormalized). Measured on 17 real samples: mean 23.8% face removal (range
  8.1-51.6%) from the three dropout mechanisms combined. Wired into `PatchTeeth3DSDataset` as
  `augment=True` by default, train-mode only (never on the fixed validation set, never in
  `get_full_sequence`) - see its class docstring.

## 6. Priority: real-world scanning/segmentation performance over validation-metric rigor
 
Explicit steer from Mohamed: optimize for real-world IOS performance (correct segmentation +
tooth identification on live scans), not for polished validation methodology. Noted so we don't
over-invest engineering effort in validation-set sophistication (e.g. multi-trajectory-per-arch
eval, which was raised as a minor concern) at the expense of things that actually move the needle
on deployed performance.

## MAA from Claude Code
New files (originals — dataset/mesh_dataset.py, dataset/preprocessing.py, models/layer.py, models/dilated_tooth_seg_network.py — remain untouched, confirmed via git diff --stat):
- models/patch_layer.py — GroupNorm everywhere BatchNorm was, plus the k = min(k, N-1) clamp that fixes the small-N crash.
- models/patch_dilated_tooth_seg_network.py — STN removed, area-gated dilation (thresholds 40/180/360mm², config-driven, dilation_gating flag for the ungated baseline), variable N throughout.
- models/patch_collate.py — CPU-side precompute for the 2 of 4 O(N²) cdist sites that are pure functions of metric position, via one shared KD-tree query + a numba-JIT'd batched FPS.
- dataset/patch_dataset.py extended to also return area_mm2 (patch's real mm² area, needed for gating).

Verified, not just built (testing/test_patch_model.py):
- Forward pass runs crash-free from N=2 up to 20,000+ (old model crashed at N≤32).
- Gated-off dilated blocks receive exactly zero gradient; engaged ones don't.
- GroupNorm's output is batch-composition-independent (float noise only, ~1e-3, not the large swings BatchNorm-with-running-stats would show).
- Precomputed KD-tree neighbor sets are exactly identical to brute-force cdist+topk — not approximate.
- End-to-end timing: worst case (~34k-face full-arch patch) collates in ~1.6s; typical patches in well under 0.5s.

One inherited, out-of-scope finding, not fixed: DilatedEdgeGraphConvBlock's on-the-fly fallback has a pre-existing B>1 indexing bug (flattens pos across batch without offsetting). Doesn't matter for us — batch_size=1 was the deliberate design choice, and real training always uses PatchCollator, which only ever handles one patch at a time and never touches that path. Documented in the test rather than silently patched.

One dependency added: numba (already present in the MAA env, not a new install) — 90-100x speedup on the FPS precompute. torch_cluster was tried per your offer, hit a >5min from-source CUDA compile with real sm_120-compatibility risk, and was abandoned in favor of the CPU KD-tree approach. requirements.txt is already out of sync with the live env (still lists torch 2.1.0+cu121, missing pyvista, etc.) so I didn't touch it — worth a real sync pass separately if you want one.

## Component 5 — loss/metrics/training loop, done and validated end to end

New files (same untouched-originals guarantee as always):
- models/patch_lightning_module.py — wraps PatchDilatedToothSegmentationNetwork + FocalLoss + torchmetrics, Adam/StepLR matching the original paper's hyperparameters.
- train_patch_network.py — the actual CLI training entry point, adapted from train_network.py. batch_size hard-coded to 1 (PatchCollator's requirement), --accumulate_grad_batches for a larger effective batch, --num_workers defaults to 8 (this session's measured optimum), --num_classes/--dilation_gating are the two real experiment-matrix flags, class_alpha (FocalLoss weighting) measured automatically from the real training distribution before each run.
- models/patch_collate.py: added PatchBatch.to(device) so Lightning's Trainer can move it automatically (it's a plain dataclass, not a type Lightning recognizes by default).

Metrics finding worth knowing: I built a custom patch-aware mIoU class (to "ignore classes absent from a patch" per the spec), tested it against plain torchmetrics.JaccardIndex(average="macro"), and found the installed torchmetrics version (1.9.0) already does exactly this natively - excludes classes with zero union from the macro average, while still correctly penalizing false-positive-only classes. Deleted the custom class as unnecessary duplication and used the built-in directly.

Also found and fixed a real (if minor) issue along the way: compute_class_alpha's docstring claimed the same seed reproduces the same alpha across runs - false for a train-mode PatchTeeth3DSDataset, since dataset[i] draws fresh unseeded randomness every call by design (seed tooth, sweep side, augmentation, etc.). The seed only fixes which arch indices get sampled, not what patch comes back from them. Confirmed directly: n_samples=20 gave wildly different alpha across two runs. n_samples=200 (the real default) is intended to be stable in aggregate, not bit-reproducible. Docstring corrected.

End-to-end validation, not just "it doesn't crash": ran a real (short) training loop - 8 truncated "epochs" of 100 train / 20 val patches each (not the full 1200/600 - kept small so the check itself was fast) - and read the actual logged metrics back from the TensorBoard event file (the terminal's rich progress bar overwrites in place, so tailing raw stdout only shows the last epoch). val_loss dropped cleanly from 0.32 to 0.17 over the first several epochs before settling into noise - real evidence of learning, not just successful plumbing. The noise afterward is expected given how tiny 100/20-sample "epochs" are relative to the real 1200/600, plus how much patch size/composition varies (10 to 40,000+ faces) - a real epoch averages over far more data and should read cleaner. Confirmed a checkpoint (.ckpt), hparams.yaml, and TensorBoard log all write correctly.

tensorboard wasn't actually installed in the MAA env despite being in requirements.txt (train_network.py would hit the same missing-module error) - installed it, which bumped protobuf 5.29.3 -> 7.35.1. Verified torch/lightning/torchmetrics/trimesh/pyvista all still import fine afterward.

Not done: real hyperparameter tuning (LR, focal gamma, whether the class-weighting is too aggressive - alpha ranged up to ~8.8x for rare classes in one measurement) - the infrastructure is confirmed working and learning, but nobody's tuned it yet. The actual full 100-epoch run (~4.6hrs estimated on this GPU) hasn't been kicked off - that's a real decision for you to make, not something to launch unprompted.

TRAINING_CONCERNS.md #1 is updated to RESOLVED with the full writeup. Not started yet: the actual Lightning training wrapper (loss + metrics + optimizer), which is Component 5 territory — natural next step whenever you want it.
