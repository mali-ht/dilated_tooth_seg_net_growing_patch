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
- Dilation gating mechanism: single tolerant net vs two heads vs always-on-trained-to-ignore.
- Seg+controller cadence: every N snapshots? per +X mm²? on demand?

## Toward the Claude Code prompt (later)
Collect: Path B + arch-assembly method, patch-generator spec, repo files/classes to change,
experiment matrix (17 vs 5 class; dilation on/off; norm scheme), pipeline changes → one scoped prompt.