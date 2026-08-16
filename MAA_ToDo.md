# DilatedToothSegNet → Robotic IOS: Implementation TODO & Brainstorm

Living doc. Adapting the DilatedToothSegNet repo (local:
`C:\Users\moham\OneDrive\Documents\Computer Vision Models\dilated_tooth_seg_net`)
from full-arch one-shot face segmentation → iterative patch segmentation on a
growing scan, as the perception brain of a robotic intraoral-scanning platform.

## System architecture (CONFIRMED)
- WINDOWS box: vendor scanner (Runyes 3ds.exe / IOSAlgo.dll). The DLL's OnlineMergingManager
  ALREADY fuses every frame into ONE unified volumetric TSDF/voxel model. Frida hook on
  ThreadMP_Publish intercepts DISPOSABLE per-region snapshots re-extracted from that volume.
  Windows role = thin: tap + ship over TCP/IP.
- LINUX box: robot + full platform. CV-in-the-loop: assemble current arch → segment (this model) →
  decide next robot pose/orientation/speed → scan more.
- Frame stability CONFIRMED by watching a long scan: earlier regions stay put as new ones arrive ⇒
  vendor frame is globally stable. Robot poses are also available on Linux (independent cross-check /
  needed for motion anyway). ⇒ drift is a non-issue; no own registration/loop-closure needed.

## Loop
snapshot(s) → assemble current arch view → understand (segment + localize) → robot move command → repeat.

## Scanner output (CONFIRMED, 268 live meshes)
Indexed triangle mesh: vertices float32 N×3 (mm), per-vertex unit normals, per-vertex uint8 RGB,
uint32 M×3 triangle indices. Metadata: mesh_id, modelset, **catalog(region)**, counts, seq, source, ts.
Density ~74.5 faces/mm² (edge ~0.19 mm). Per snapshot median ~12.9k faces / ~174 mm² ≈ ONE TOOTH.
Native density ≈ Teeth3DS raw (~77/mm²).

## KEY FINDINGS
1. No stable IDs across publishes → NO cross-publish face identity from the scanner.
2. Snapshots are overlapping region re-extractions from the DLL's single volume, ONE shared frame.
3. Geometry accumulation is the VENDOR'S job (single TSDF) → we inherit a consistent frame; don't rebuild it.

## CHOSEN: Path B — run the model on the growing accumulated arch
### Accumulation design (vendor already fuses geometry)
- Get current arch via (a) BEST: a Frida hook to extract the DLL's WHOLE volume in one shot; or
  (b) FALLBACK: newest-snapshot-per-(modelset,catalog) union + light voxel weld at seams.
- Stable identity: none across extractions, but Path B re-runs seg on the whole current arch each
  cadence ⇒ coherent latest labeling. Per-voxel LABEL grid = OPTIONAL temporal stabilizer.
- Decouple cadences: assemble continuously; run seg + controller every N snapshots / per +X mm² / on demand.

## Working density — LOCKED
- **W = 5 faces/mm² (full arch ≈16k faces)** — repo-proven, compute-cheap.
- Derived: one tooth ≈1,000 faces; min patch (~200 mm²) ≈1,000 faces; triangle edge ~0.65 mm;
  voxel-weld / marching-cubes cell ~0.5–0.65 mm.
- Dilation reach in physical units: block1 ~40 mm², block2 ~180 mm², block3 ~360 mm² (≈0.2/0.9/1.8 teeth).
- INVARIANT: live-arch density == Teeth3DS density == W. Downsample BOTH to W (both are downsamples
  from ~74–77 native → faithful). Full arch at native ~74 (~185k faces) is infeasible for the N² dilated block.

## Training-data patch curriculum (DESIGN IN PROGRESS)
Goal: reproduce the assembled arch as it grows, so the model trains on what it will actually see.
Two axes of partialness + supporting pieces:
1. EXTENT along the arch — DECIDED:
   - Downsample each Teeth3DS arch to W first, THEN crop.
   - Seed at a FRONT tooth (matches manual robot start); grow posteriorly (both sides from midline).
   - Growth is CONTINUOUS by area (geodesic flood-fill over face adjacency), boundaries cut ACROSS
     teeth freely (NOT tooth-quantized). Floor ≈ one tooth's area (~150–250 mm² ≈ ~1k faces) → full arch.
   - Front-loads the hardest (anterior) teeth — good, that's where type is most ambiguous.
2. SURFACE COVERAGE per tooth (occlusal-first) — NEXT.
3. Sampling mode: independent (E,c) snapshots vs ordered growing trajectories — TBD.
4. Labels: inherit per-face; emit 17-class AND 5-class; handle ambiguous anterior — TBD.
5. Normalization consistency: same fixed-mm scheme as inference, on every patch.
6. Sim-to-real augmentation: holes, ragged boundaries, normal/coord noise, density jitter.
7. Balancing: patches per size bucket; oversample anterior-occlusal-only, premolar↔molar boundary, missing teeth.

## Decisions locked in
- Path B; vendor owns TSDF; assemble via whole-volume hook or newest-per-region+weld. Drift non-issue.
- W = 5 faces/mm² (~16k/arch); downsample both scanner arch and Teeth3DS to W.
- Windows thin: ship all snapshots. Accumulation + CV + control on Linux.
- Retrain from scratch; train BOTH 17-class (FDI) and 5-class (+gum); compare.
- Variable-N is free (net fully convolutional over points).
- Patch curriculum: downsample-then-crop; anterior seed; continuous cross-tooth growth.

## Running TODO / ideas
1.  [x] Instrument the IOS — DONE.
2.  [x] Resolve design fork — Path B.
3.  [x] Locate accumulation — DLL TSDF; consistent frame; drift non-issue.
4.  [ ] Arch assembly on Linux: whole-volume hook preferred; else newest-per-region + voxel weld.
5.  [ ] Windows→Linux transport off on_mesh() (TCP/IP).
6.  [ ] Patch generator: extent axis (anterior seed, continuous area growth) — design mostly done.
7.  [ ] Patch generator: coverage axis (occlusal-first via face-normal alignment) — NEXT.
8.  [ ] Dilation curriculum gated in mm² (40/180/360); keep global_hidden_layer concat valid when on/off.
    NOTE: gating alone doesn't fix the O(N²) cdist cost (computed upfront regardless of gating) -
    see TRAINING_CONCERNS.md #1, blocking for large (40k+ face) patches.
9.  [ ] Variable-N pipeline: drop pad/resample-to-16,000; simplify to fixed DENSITY.
10. [ ] Patch-invariant normalization: fixed mm scale; normals raw; identical train & infer.
11. [ ] Replace BatchNorm → GroupNorm/LayerNorm.
12. [ ] Remove / rethink STN for fragments.
13. [ ] Drop or localize absolute-coordinate input channels; keep normals; consider curvature.
14. [ ] Class-weighted or focal loss; balanced patch sampling.
15. [ ] Patch-aware metrics.
16. [ ] Uncertainty/abstain signal.
17. [ ] Compute: KD-tree/spatial-hash neighbors; precompute static dilation graph.
18. [ ] Sim-to-real augmentation (holes, noise, density jitter).
19. [ ] Ingestion adapter: scanner arrays → per-face feature builder; resample to W.
20. [ ] (Optional) per-voxel label grid for temporal stabilization.
21. [ ] (Later) Exploit real RGB color — blocked by Teeth3DS lacking real color.

## Open design questions
- Frida hook for the DLL's WHOLE unified mesh at once (vs reassembling tiles)?
- Coverage axis: independent (E,c) sampling vs ordered occlusal-sweep-then-sides trajectories?
  See TRAINING_CONCERNS.md #3 - built the ordered occlusal->facial->lingual trajectory, but it's
  a hypothesis (robot policy not finalized), mitigated by training on randomized/broad patches.
- Dilation gating mechanism: single tolerant net vs two heads vs always-on-trained-to-ignore.
- Seg+controller cadence: every N snapshots? per +X mm²? on demand?

See TRAINING_CONCERNS.md for a fuller design-review pass (Aug 2026) covering the cdist bottleneck,
training batch structure, curriculum-vs-real-policy risk, facial/lingual contribution, and
confirmed priorities (loss weighting, sim-to-real augmentation) before training starts.

## Toward the Claude Code prompt (later)
Collect: Path B + arch-assembly method, patch-generator spec, repo files/classes to change,
experiment matrix (17 vs 5 class; dilation on/off; norm scheme), pipeline changes → one scoped prompt.

## Whole-tooth / multi-tooth patch curriculum (PROPOSED - not started)

Motivation (Mohamed, live-testing session): the occlusal/facial/lingual scan-sweep patches
(`grow_patch_full_sweep`, confirmed already built and wired into `PatchTeeth3DSDataset.__getitem__`
- this section doesn't replace that) are all LOCAL-FOOTPRINT captures (`rectangular_footprint`,
~14x14mm boxes aligned to a scanner-simulated viewing direction) - no patch is ever bounded by
tooth IDENTITY, so the model never sees a training example that's explicitly "this is one (or a
few) COMPLETE tooth/teeth, cleanly bounded." Hypothesis: adding patches bounded by ground-truth
tooth labels (trivial to build - `load_cached_arch` already returns per-face labels) would give the
model a structural prior for whole-tooth shape/extent, complementing the local, partial-view
footprints the scan-sweep already provides. Directly motivated by a live investigation finding
(REALTIME.md, this session): a small early patch predicts neighbor teeth before enough context
accumulates - by design (area-gated dilation), not obviously fixable by this alone, but worth
testing whether whole-tooth training examples help the network's prior regardless.

Relationship to TRAINING_CONCERNS.md #3 (curriculum-vs-real-policy risk, accepted/mitigated by
randomized broad sampling): this is the same philosophy applied one level further - ADD a second,
structurally different patch family and mix it in probabilistically, rather than replace the
scan-sweep curriculum with a hand-designed alternative.

1. [ ] New generator function(s) in `dataset/patch_generator.py`, mirroring the existing
   `order_teeth_from_seed`/`grow_patch_scanner_sweep` pattern but swapping
   `rectangular_footprint`'s geometric capture for a trivial label mask:
   - `whole_tooth_mask(labels, tooth_label)`: `labels == tooth_label`.
   - `grow_patch_whole_teeth(mesh, labels, seed_label, rng)`: reuse `order_teeth_from_seed` for
     the visiting order, but stage i's mask = union of `labels == t` for the first i teeth in that
     order (instead of accumulating rectangular footprints) - gives a sequential "one whole tooth
     at a time" growth curriculum directly analogous to the existing occlusal sweep, for either
     single-tooth or N-teeth-at-once sampling (see stage-count sampling below).
2. [ ] OPEN DESIGN QUESTION, needs a decision before building: pure tooth-label boundary (clean,
   but unlike anything a real noisy scan produces - a real capture always includes some margin of
   surrounding gum/adjacent tooth sliver) vs. tooth-label + a small buffer margin (geodesic or
   spatial dilation of the mask by a few mm, matching how `rectangular_footprint` already admits
   "immediately adjacent gum"). Leaning toward the margin version for realism, but flagging rather
   than deciding unilaterally.
3. [ ] Decide the multi-tooth sampling distribution: mostly single teeth, occasionally 2-3
   contiguous teeth (reuse the adjacency already encoded in the 1-16 label order - "several teeth"
   = a contiguous label range, e.g. {6,7,8} or {8,9,10} spanning the midline), rarely more -
   probably an `early_bias_power`-style biased draw over tooth-COUNT, mirroring the existing
   biased draw over scan-sweep STAGE.
4. [ ] Wire into `PatchTeeth3DSDataset.__getitem__` as a second sampling mode, chosen
   probabilistically per draw alongside the existing scan-sweep mode (a new mix-ratio parameter,
   e.g. `whole_tooth_patch_prob`) - NOT a replacement path. Confirm `area_mm2`/labels/the
   `(pos, x, labels, area_mm2)` shape stay identical regardless of which mode produced the mask, so
   nothing downstream (transform, collate, area-gating) needs to know which one it got.
5. [ ] Confirm `SimToRealAugment` (`punch_holes`/`erode_boundary`/`density_subsample`/jitter) still
   makes sense applied to whole-tooth patches - `erode_boundary`'s "boundary = adjacent to a mask
   edge" definition should work generically regardless of what shaped the mask, but verify directly
   rather than assume (same category of stale-assumption bug already found once in
   TRAINING_CONCERNS.md #4's connectivity-assertion incident).
6. [ ] Tests (`testing/test_patch_generator.py`-style, matching existing coverage): correct masks
   for single/multi-tooth draws, correct behavior when a tooth in the requested range is absent
   from that arch, correct behavior spanning the midline (8/9) and a missing-tooth gap.
7. [ ] Visualization script (`testing/visualize_*`-style, matching `visualize_patch_growth.py`) to
   eyeball the new patch type before trusting it in training.
8. [ ] Validation plan tied to the actual hypothesis, not just "does it run": some way to measure
   whether whole-tooth training changes small-patch tooth-ID behavior - e.g. compare how quickly a
   live/synthetic small patch's prediction stabilizes to the correct label with vs. without
   whole-tooth patches mixed in, using the snapshot methodology already built
   (`testing/realtime/visualize_snapshots.py`) as the measurement tool.

## Real-hardware finding: terminal molars (labels 1 and 16) systematically over-predicted (NEW - motivates the above)

Confirmed across TWO independent real-hardware runs (`mesh_viewer_segmented_20260815_111441`,
`_104816`) via the snapshot logging - label 16 (rightmost 3rd molar) grows to 2,700-3,000 faces by
scan end while its immediate neighbor label 15 (2nd molar) actually SHRINKS to under 100 - a
23-30x imbalance, reproduced in both runs, and visually confirmed in rendered snapshots (label
16's dark-green region visibly swallows what should be label 15's territory). Label 1 vs. label 2
(the arch's other end) shows the same DIRECTION of bias but milder (roughly 1.3-1.5x in one run,
comparable magnitude in the other) - both cases are the LAST tooth in a sweep direction being
over-predicted at its immediate neighbor's expense, not random noise (an anti-correlated trend
over many cycles, not just variance).

Practical impact: this is directly why the live scan guide (REALTIME.md's motion-guidance arrow)
was seen ending prematurely - a small/spurious early bleed of label 1 or 16 face count crosses
`guidance_min_faces` before that tooth is genuinely fully covered, since the guidance system only
checks "is a label present with enough faces," not "does this look like genuine full coverage."

Two distinct, unconfirmed root-cause hypotheses worth investigating before assuming which one it
is:
- 3rd molars (wisdom teeth) are frequently missing/impacted/variable in real dental data - if
  Teeth3DS under-represents "2nd molar directly next to a PRESENT, normal 3rd molar," the model
  may have systematically weaker training signal at exactly that boundary, on both sides of the
  arch (matching where the imbalance is worst).
- The scan-sweep curriculum's per-side sweep-to-completion pattern may under-represent how much
  ACCUMULATED CONTEXT surrounds the last tooth in a real deployed scan (an arch's worth of
  already-scanned neighbors) vs. how much surrounds it during a training-time patch draw - same
  general curriculum-vs-deployment mismatch risk TRAINING_CONCERNS.md #3 already flagged as an
  open, accepted risk for a different reason.

The whole-tooth/multi-tooth patch idea above is a plausible mitigation for the first hypothesis
(explicit whole-tooth shape examples for 1/16 specifically) - worth prioritizing 3rd molars in
whatever tooth-count sampling distribution gets built, given this evidence, rather than uniform
sampling across all 16 positions.

## Per-face color as a model input (PROPOSED - blocked on Teeth3DS, needs synthetic color instead)

Motivation (Mohamed): color should help gum/tooth separation a lot (Finding above shows real
boundary-level teeth/gum confusion) - teeth are enamel-white/cream, gum is pink/red, a strong
natural visual cue the model currently can't see at all. Proposed generating gum/tooth ground
truth from pre-set color ranges.

**Verified blocker before writing anything else down**: the raw Teeth3DS `.obj` files carry NO
real color at all - checked directly (`trimesh.load` on a real Teeth3DS mesh, all 186,860 faces
return the exact same placeholder RGB(128,128,128), not real captured appearance). This isn't a
label-encoding trick either - `dataset/mesh_dataset.py`'s `process_mesh()` has a
`colors_to_label(mesh.visual.face_colors)` fallback that LOOKS like it decodes labels from color,
but the real training loader (`Teeth3DSDataset._iterate_mesh_and_labels`) never uses it - labels
always come from the separate per-mesh `.json` sidecar file. So there is no color signal to build
ground-truth-from-color-ranges FROM in the current training data - confirms/sharpens the existing
`MAA_ToDo.md` TODO #21 note ("blocked by Teeth3DS lacking real color"), now with a concrete reason
why, not just a remembered blocker.

**What DOES have real color: the live scanner** (`mesh_wire.py`'s `parse_mesh_payload`, `colors`
field - real per-vertex RGB from the vendor scanner) - but nothing currently logs it anywhere
usable for training (the live snapshot tool, `LiveSegmenter._save_snapshot`, only saves
vertices/faces/pred_labels, not the original scanner color).

**Corrected plan - synthetic color painted FROM the already-known ground truth, not the reverse**:
since we can't derive gum/tooth ground truth from color (no real color exists in Teeth3DS), do the
opposite - use the ground truth labels we already have to PAINT plausible per-face color as a new
sim-to-real augmentation, then feed that as a new model input channel:
1. [ ] Capture real color statistics to calibrate against: add color capture to
   `LiveSegmenter._save_snapshot` (or a lighter dedicated logging path) so a real scan's actual
   per-face RGB can be measured and correlated against its predicted class - gives real target
   distributions for "what does enamel/gum actually look like on THIS scanner" instead of guessing
   generic values.
2. [ ] New `SimToRealAugment` perturbation (`dataset/patch_augmentation.py`): paint each face a
   color sampled from a per-class distribution (tooth classes -> whites/creams with staining
   noise; gum -> pinks/reds with realistic noise), calibrated from step 1 once real statistics
   exist - same pattern as the existing 5 perturbations (`punch_holes`/`erode_boundary`/
   `density_subsample`/`jitter_positions`/`jitter_normals`), randomized per draw, train-mode only.
3. [ ] Widen the model's input: `dataset/patch_preprocessing.py`'s `PatchPreTransform` currently
   builds a 24-dim feature (3 triangle-vertex positions + face center + 3 vertex normals + face
   normal, verified directly) with no color channel - add RGB (per-face, or per-vertex matching
   the existing 3x-repeated position/normal pattern) and bump `feature_dim` in
   `PatchDilatedToothSegmentationNetwork` to match. This is an input-layer dimension change - not
   fine-tune-compatible, requires a full retrain from scratch, same as any other experiment-matrix
   variant already planned (17 vs 5 class, dilation on/off).
4. [ ] Validation ablation: train with vs. without the synthetic color channel, compare gum/tooth
   boundary quality specifically (not just overall mIoU) - the snapshot methodology
   (`testing/realtime/visualize_snapshots.py`) plus this session's Finding above (visible
   boundary-level speckle noise) gives a concrete before/after to check against on real hardware,
   not just held-out validation metrics.

**Also worth fixing independently of any retrain**: labels 1/16 and 2/15 are nearly indistinguishable
colors in the current palette (`utils/teeth_numbering.py`'s `_teeth_color` - label 1 is RGB(7,118,7),
label 16 is RGB(8,118,47); label 2 is RGB(144,239,144), label 15 is RGB(145,239,184) - each pair
differs only in the blue channel). Part of what reads as "these teeth are always confused" when
watching the live view is a genuine model bias (confirmed above), but part of it is simply that a
human can't visually tell left-side-last-molar from right-side-last-molar apart by color alone even
when both are correctly labeled. A GUI-only palette fix (give each side's terminal 1-2 teeth more
distinguishable hues) would remove that confound and make it easier to see whether real-hardware
predictions are actually right or wrong at those positions.