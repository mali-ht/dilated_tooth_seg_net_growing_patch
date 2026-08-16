# Real-time inference on the live scanner feed (living doc)

Companion to `TRAINING_CONCERNS.md` (which covers Components 1-5, the offline training
pipeline) - this one covers the separate real-time integration built on top of a trained
checkpoint: connecting to the physical intraoral scanner (via a Windows machine running
`mesh_server_win.py`, which Frida-hooks the scanner's own DLLs) and running the model on the
mesh as it streams in, live.

Everything here lives under `testing/realtime/`. Two files in that directory are the existing,
untouched interface to the real hardware and must stay that way:
- `mesh_wire.py` - the wire protocol (framing, header/blob layout) shared by both machines.
- `mesh_viewer_linux.py` - the original raw-color Open3D viewer.

Everything else in `testing/realtime/` is new, built this session, and safe to keep changing.

## Architecture

```
Windows: mesh_server_win.py (Frida-hooked scanner DLLs)
              |  (one real TCP connection)
              v
       mesh_relay.py  <-- optional, only needed if running 2 viewers at once
         /            \
        v              v
mesh_viewer_linux.py   mesh_viewer_segmented.py
 (unmodified,           (this session's work:
  raw scan colors)       live inference + predicted-class colors)
```

For normal single-viewer use, `mesh_viewer_segmented.py` connects **directly** to the Windows
server (same default host as `mesh_viewer_linux.py`, `192.168.1.30`) - no relay needed. The relay
exists only for the case of running both viewers simultaneously, to avoid a second direct
connection to a server whose multi-client support is unverifiable (no access to
`mesh_server_win.py`'s source).

## Files

- **`mesh_relay.py`** - local TCP fan-out: one real upstream connection to the Windows server,
  N local downstream connections (for the two viewers). Caches the latest frame per `mesh_id` so
  a viewer that connects mid-scan gets caught up, not stuck waiting for chunks that already
  arrived. Routes command frames (start_scan/stop_scan) from any local client straight upstream.
- **`live_preprocessing.py`** - the actual bridge from raw scanner geometry to model-ready input:
  - `downsample_to_density()`: quadric-decimates a chunk to the model's training density
    (5 faces/mm², matching `processed_w5`/`dataset/mesh_dataset.py`'s `_donwscale_mesh`, minus
    the label-remapping that doesn't apply to unlabeled live data).
  - `merge_meshes()`: concatenates independent chunk meshes into one, offsetting face indices.
  - `claim_new_voxels()` / `voxel_size_for_density()`: cross-chunk deduplication - see Finding #2
    below for why this exists. **The single most important piece added after real-hardware
    testing** - without it the pipeline reliably OOMs within a couple minutes of real scanning.
- **`mesh_viewer_segmented.py`** - `LiveSegmenter` (pure processing core: accumulate chunks →
  per-chunk decimate → cross-chunk dedup → merge → model inference → per-face colors + Universal
  tooth-number label positions; no Open3D dependency, drivable headlessly) wrapped by
  `SegmentedViewerApp` (the actual Open3D GUI - Start/Stop buttons, color legend, floating 3D
  tooth-number labels, a background worker thread that never blocks the GUI or queues stale work).
- **`fake_mesh_server.py`** / **`test_live_segmenter.py`** - offline test harness and headless
  pipeline test, since this session had no access to the real hardware to test against directly.
  Both were originally built on an assumption about chunk behavior that turned out to be WRONG
  once tested against the real scanner - see Finding #1. `fake_mesh_server.py` still simulates
  the old (wrong) "one mesh_id grows in place" pattern; `test_live_segmenter.py` has been
  corrected to match reality (see "Not yet done" below - `fake_mesh_server.py` itself should
  probably be updated too, hasn't been yet).

## Findings from real-hardware testing (empirical, not assumed - this is the important part)

### Finding #1: mesh_ids are permanent and never reused

Before ever connecting to real hardware, the code (including this doc's author) assumed a
`mesh_id` was a persistent spatial region that gets refined/replaced over time, based on
`mesh_viewer_linux.py`'s `remove_geometry(name); add_geometry(name, ...)` pattern (which reads as
"replace this named region's geometry"). **Confirmed wrong** against the real scanner: `mesh_id`
values observed were strictly increasing (334, 335, 336, ... 548+ in one session) and never
repeated. The real system streams a continuous flow of small, NEW fragments as the scanner
sweeps - normal scanning motion re-observes already-covered surface, so these fragments overlap
each other spatially. Nothing about a `mesh_id` being reused was ever observed.

Real per-chunk sizes observed: typically ~9,000-17,000 raw faces per chunk (before our own
decimation), arriving roughly every 0.1-0.3s during active scanning; occasional much larger
bursts (60,000-109,000 raw faces in a single frame) also seen, cause unclear (possibly a
keyframe/loop-closure event in the scanner's own internal reconstruction).

### Finding #2: chunk overlap causes unbounded growth -> CUDA OOM (RESOLVED)

Direct consequence of Finding #1: per-chunk decimation alone correctly bounds each individual
chunk's own density, but does nothing about redundancy BETWEEN chunks. Naively merging every
chunk ever received made the total grow with elapsed scan TIME, not true surface area - measured
live: 44,366 -> 92,863 -> 197,452 -> 241,611+ merged faces over a few minutes of scanning a
single arch, against the ~16,000-18,000 faces a true full arch at 5 faces/mm² should need.

This directly caused CUDA OOM crashes with escalating allocation requests as the session
continued (32.13 GiB, then 145.24 GiB, then 217.47 GiB attempted allocations - all on a 32GB
GPU). It also very likely caused the "connection lost, will reconnect" cycling observed
alongside the OOMs: processing 200k+ points on the CPU side (KD-tree + FPS in
`models/patch_collate.py`) plausibly took long enough to starve the reader thread and make the
TCP connection look dead, even though the process itself never crashed.

**Fix**: `claim_new_voxels()` - a persistent, scan-wide voxel grid (`set` of integer voxel-index
tuples, never cleared). Each chunk's faces are checked against it after per-chunk decimation;
only the first face to claim a given voxel survives. Total kept faces are bounded by true
covered area, independent of how many times it's been re-observed.

Getting the voxel size right took two iterations, both worth remembering:
- **First attempt**: `voxel_size = 1/sqrt(target_density)` (the natural face-to-face spacing a
  SINGLE chunk decimated to `target_density` would have). Measured wrong: this made genuinely
  adjacent, non-redundant faces WITHIN one chunk alias into the same voxel and get discarded -
  lost ~30% of a single, non-overlapping test chunk for no real reason.
- **Fix**: scale the grid down to ~0.3x that naive spacing (`voxel_size_for_density`'s
  `fineness` parameter, default 0.3) - fine enough that real neighbors stay distinguishable,
  still coarse enough to catch true redundant re-observations. Verified both properties
  together in `test_live_segmenter.py`: a clean 6-chunk non-overlapping growth sequence lands
  the merged density at 4.68-4.74/mm² (vs. the 5.0 target) with <15% loss to grid aliasing at
  every step, AND a dedicated test re-sending an EXACT duplicate of already-covered surface
  under a brand-new mesh_id adds exactly 0 new faces, while a genuinely disjoint new chunk still
  adds real ones.

**Status: fixed and verified against synthetic data reproducing the real failure mode. Re-tested
against real hardware afterward - see Finding #5 below for what that turned up.**

### Finding #3: `downsample_to_density`'s original overshoot-trim was too slow (RESOLVED, earlier in this session)

Before the OOM issue, a separate freeze was diagnosed and fixed: `downsample_to_density`
originally used `trimesh.sample.sample_surface_even()` (rejection sampling) to trim decimation
overshoot, matching `dataset/mesh_dataset.py`'s offline preprocessing. That's fine for one-time
offline preprocessing but measured badly live - it printed "only got X/Y samples!" and could
stall for a long time as the input mesh grew. Fixed by switching to a plain uniform random face
subsample (pyfqmr's decimation output is already close to evenly distributed, so trimming the
small remaining overshoot doesn't need rejection sampling's stronger, much more expensive
guarantee).

Also fixed at the same time: `process_once()` originally re-decimated the ENTIRE accumulated raw
mesh from scratch every cycle. Since the raw mesh only ever grows, this got slower every cycle,
compounding as the scan continued. Fixed by decimating each chunk INDEPENDENTLY, cached per
`mesh_id`, only redone when that specific chunk's raw data changes (see `LiveSegmenter.
_decimated_chunks`). This is what made Finding #1/#2 visible in the first place - once per-chunk
decimation stopped being the bottleneck, the cross-chunk redundancy problem became the next one.

### Finding #4: `_teeth_codes_lower`/`_teeth_codes_upper` in `utils/teeth_numbering.py` are named backwards

Discovered while building `label_to_universal_number()` (converts the internal 1-16 tooth label
scheme to the real Universal Numbering System, 1-32). `_teeth_codes_lower` actually contains FDI
codes 11-28 (quadrants 1&2, real maxillary/UPPER per the FDI standard); `_teeth_codes_upper`
contains 31-48 (quadrants 3&4, real mandibular/LOWER). Confirmed directly against a real
Teeth3DS `*_lower.json` label file (FDI codes present: 31-47, matching `_teeth_codes_upper`'s
keys, not `_teeth_codes_lower`'s). This never mattered before: `fdi_to_label()` merges both
dicts into one combined lookup (their key sets are disjoint) and never needed to pick one
specifically. `label_to_universal_number()` selects the correct dict per arch, with the swap
called out in a comment; the original (mislabeled but functionally harmless) dict names were
left alone.

Verified two independent ways: round-tripping through the internal label scheme, and computing
straight from real FDI codes with zero shared logic - both agree exactly.

### Finding #5: voxel dedup alone doesn't bound total merged size over a LONG scan -> a second, different CUDA OOM (RESOLVED)

Re-tested against real hardware after Finding #2's fix. The dedup mechanism itself worked
correctly on a per-observation basis (verified live: an exact duplicate re-observation added
`+0` faces, same as the synthetic test predicted). But over a much longer real scan (~600 mesh
frames, mesh_id 549 through 1164), the total merged face count still climbed past the ~16-18k a
full arch needs - it reached 124,437, then 158,599, then 161,691 faces before crashing, all
still growing. Two contributing factors, both visible in the debug log:

- **Continuous small leakage**: hundreds of small, genuinely-overlapping chunks each leak a
  handful to a few hundred "new" faces past dedup (normal grid-boundary effects, not a bug by
  itself) - this alone adds up over ~600 chunks.
- **Periodic large chunks leak disproportionately more**: several mesh_ids arrived far larger
  than the typical 8k-25k-triangle chunk (e.g. mesh_id 849: 103,813 triangles; 948: 122,237;
  1046: 135,014; 1144: 148,161; 1162-1164: ~90k-149k each) - likely server-side re-batched/
  re-triangulated regional updates rather than true full-scan snapshots (their raw size is far
  below the multi-million-face accumulated total by that point). Each one still leaked 600-1800
  "new" faces past dedup despite covering mostly already-seen surface, probably because
  re-triangulation shifts face centroids just enough to land in different voxel cells than the
  original observation.

**This alone would just mean the model runs on a somewhat-larger-than-ideal patch - not a crash.
The actual crash came from where that oversized merged mesh gets fed next.** Reading
`models/patch_dilated_tooth_seg_network.py`'s `forward()` (not the protected original
`models/dilated_tooth_seg_network.py` - this is the new Component-4 patch network) shows only
`edge_graph_conv_block1` receives a precomputed `idx` (from `local_idx`); blocks 2 and 3
(`models/patch_layer.py`'s `EdgeGraphConvBlock`) are called with no `idx` at all, which falls
into `get_graph_feature()` -> `knn()` -> an **on-the-fly `torch.cdist(x_t, x_t, p=2)`** over
every point, every forward pass, with no cap. This is deliberate and documented
(`models/patch_collate.py`'s module docstring: blocks 2/3's kNN is in *feature* space, computed
from mid-forward activations, so it genuinely can't be precomputed on CPU ahead of time like the
other 2 O(N²) ops the patch model has) - it was never a problem for bounded, training-scale
patches, but nothing bounds N for a live merged mesh.

The math confirms it exactly: `torch.cdist` on N points materializes an (N, N) float32 tensor,
N²×4 bytes. At N=124,437 that's ~61.9 GB (observed OOM: tried to allocate 57.69 GiB); at
N=158,599, ~100.7 GB (observed: 93.71 GiB); at N=161,691, ~104.6 GB (observed: 97.39 GiB) - all
close to the real 32GB-GPU allocation attempts once other overhead is accounted for. The
"connection lost, will reconnect" cycling that followed each OOM, and the eventual full stall
(no more `recv mesh frame` at all), is almost certainly a downstream symptom of this same
resource exhaustion, not a separate bug.

**Fix**: `LiveSegmenter.process_once()` now hard-caps the merged mesh's face count at
`max_model_faces` (new `__init__` parameter, `--max_faces` CLI flag, default 20000 - comfortably
above the ~16-18k a full arch needs at `target_density=5.0`) by re-running
`downsample_to_density` on the merged mesh at whatever density makes `target_count` land on the
cap, whenever the post-dedup count exceeds it. This is independent of and layered on top of the
voxel-dedup fix (which still does the real work of keeping normal-length scans near their true
covered area) - it's a final backstop for whatever dedup doesn't catch, and it's also just
correct regardless of memory: the model was never trained on patches anywhere near 100k+ faces,
so capping here keeps live inputs within the distribution the checkpoint actually learned.
`stats["dedup_faces"]` (post-dedup, pre-cap) vs `stats["down_faces"]` (post-cap, what the model
actually sees) are both tracked separately now, and the GUI's stats panel shows when a cap
actually triggered.

Verified headlessly (`test_live_segmenter.py`'s new "max_model_faces cap" section): feeding the
entire 186,860-face test mesh as a single chunk with `max_model_faces=500` produces
`dedup_faces=17121` (dedup did its normal job) but `down_faces=500` (the cap held), and
inference completes normally afterward.

**Status: fixed and re-verified against real hardware.** The `max_model_faces` cap held the whole
run - no more OOM, no allocation attempts anywhere near GPU capacity. See Finding #6 for what
that same real-hardware run turned up next.

### Finding #6: the cap itself became the new "redo full work every cycle" cost (RESOLVED)

Re-tested Finding #5's fix against real hardware. The OOM was gone, but the user reported it
"worked, but was slow, not real-time" - cycle times climbed to 6-7 seconds. The cap step
(`downsample_to_density(down, target_density=self.max_model_faces/down.area)` at the time) was
calling a full quadric decimation on the ENTIRE merged mesh every cycle once it exceeded
`max_model_faces` - and once a scan covers anywhere near a full arch (~16-18k faces), the merged
mesh exceeds the 20000 default almost immediately and stays over it for the rest of the scan, so
this ran on literally every subsequent cycle. Measured live: `downsample_to_density: 54031 ->
53967 (target 20000) in 1.674s`, then `68196 -> 68058 (target 20000) in 2.141s` and climbing -
this was the dominant share of the multi-second cycle times, and it's the exact same anti-pattern
Finding #3 already fixed once (re-decimating an ever-growing whole from scratch every cycle),
just re-emerging one step later in the pipeline.

**Fix**: new `cap_face_count()` in `live_preprocessing.py` replaces the cap's decimation call
with a plain uniform random face subsample (same reasoning as `downsample_to_density`'s own
overshoot-trim: the merged mesh reaching this point is already evenly covered from per-chunk
quadric decimation, so a random subsample loses density evenly rather than needing decimation's
shape-aware simplification). Verified headlessly: the cap step dropped from >1s to `0.000s` in
`test_live_segmenter.py`'s cap test.

Real-hardware log also showed `connection lost, will reconnect` clustering right around the
heaviest decimation calls in both this run and the Finding #5 run - plausibly explained by
`pyfqmr`'s C++ decimation call holding the GIL for its full duration (it's a single blocking
call, not chunked with periodic GIL release), starving the reader thread from servicing the
socket long enough for the server to give up on the connection. Not proven (no access to
`mesh_server_win.py`'s source to know its timeout behavior), but consistent with every observed
instance, and removing the largest recurring GIL-holding stall (the cap decimation, which used to
run on literally every cycle once the scan grew past ~18-20k faces) should reduce it regardless
of whether that specific theory is right.

**Status: fixed and verified headlessly (cap step now ~0.000s instead of 1-2s+). NOT yet
re-verified against real hardware - that's the next real-hardware test to run.** One cost this
does NOT eliminate: per-chunk decimation of periodically-arriving large "keyframe-like" chunks
(seen at ~90-150k raw triangles, e.g. mesh_id 1292-1294 in the Finding #6 log) still takes
0.6-0.7s each - but that's a one-time cost per such chunk (cached afterward), not a per-cycle
cost, so it shows up as an occasional stall rather than the compounding slowdown the cap fix
addresses. If real-hardware testing still shows unacceptable stalls from these, the next place to
look is there.

### Finding #7: full per-stage profiling added, and the "slow" stage was NOT what was suspected (RESOLVED for the two bugs found; investigation ongoing)

Cycle 6 asked directly: "put a lot of timing debug statements... I want to start investigating
and finding solutions to what you are doubting" - the four suspects being the model's on-the-fly
`cdist` (blocks 2/3), periodic large chunks, the single sequential worker thread, and GIL
contention. Added full instrumentation to answer this with real numbers instead of guesses:

- **`models/patch_dilated_tooth_seg_network.py`**: `forward()` gained an optional `profile=False`
  parameter (default - zero cost, every training call is unaffected). When `True` (only
  `LiveSegmenter` ever passes this), every block is bracketed by `torch.cuda.synchronize()` +
  `time.time()` and the per-stage wall times land on `self.last_timings` (an instance attribute,
  not a second return value - keeps the training call signature untouched). Synchronizing matters
  here specifically: CUDA ops queue asynchronously, so a naive `time.time()` around a GPU call
  without `synchronize()` measures how long it took to *enqueue* the op, not run it - block1 vs.
  block2/block3 (the ones with no precomputed idx, doing the real on-the-fly feature-space cdist)
  needed to be separated cleanly to test the suspicion directly.
- **`models/patch_collate.py`**: `precompute_neighbor_indices()` gained the same `profile=False`
  opt-in (training's `PatchCollator.__call__` never passes it, so it's still a plain 2-tuple
  return there - only `LiveSegmenter`'s direct call uses the 3-tuple form). Broken into
  `kdtree_build`, `kdtree_query`, `candidate_gather`, and one `fps_dilation{i}_k{dk}` entry per
  dilation level.
- **`testing/realtime/live_preprocessing.py`**: `downsample_to_density` now tags any single
  chunk's decimation with `[SLOW]` if it takes >0.3s, so the "periodic large chunk" suspect is
  grep-able directly in a live log instead of needing to eyeball every line.
- **`testing/realtime/mesh_viewer_segmented.py`**: `LiveSegmenter.process_once()` now times every
  pipeline stage separately (`t_decimate_chunks`, `t_dedup_chunks`, `t_merge`, `t_cap`,
  `t_feature`, plus the collate/model breakdowns above) and tags every debug line with an
  incrementing `cycle=N` so a cycle's pipeline/collate/model lines can be correlated. The `reader()`
  loop in `main()` now also logs `recv_blocked` and `gap_since_last_frame` per mesh frame (tagged
  `[GAP]` above 200ms) - the direct test for the GIL-contention suspect: gaps that track this
  process's own CPU-heavy stages rather than anything server-side would be the signal to look for.

**What the numbers actually showed** (headless, but against a real cached full-arch mesh at
~17k faces post-cap - the same scale a real scan reaches): the on-the-fly `cdist` suspect was
*not* the bottleneck. `block2_onthefly_cdist` and `block3_onthefly_cdist` measured ~0.011s
*each* - the model's entire forward pass was ~0.06s total. The real cost was almost entirely in
`precompute_neighbor_indices` (`t_collate`), which was running **0.75-0.83s** - more than 10x the
model's own cost - and of that, the newly-added `candidate_gather` timing (a stage that used to be
completely unmeasured, hidden inside the old bundled `t_collate` number) accounted for **0.42-
0.43s** of it: bigger, on its own, than the entire model forward pass.

Two concrete bugs found and fixed once that stage was actually visible:
1. **A redundant full-array copy.** `cand_pos_all = pos_np[cand_idx_all].astype(np.float32)` -
   `pos_np` is already `float32` (traced through `PatchPreTransform`'s `torch.zeros(s,
   24).float()`), so this `.astype()` was silently making a second ~370MB copy of a result that
   already had the right dtype (same issue on `cand_idx_all`'s `.astype(np.int64)` - `cKDTree`
   already returns `np.intp`, which is `int64` on this platform). Fixed with `copy=False` on both
   - `astype` only copies when it actually needs to. This alone roughly halved `candidate_gather`
   (0.42s -> 0.22s).
2. **The gather itself was single-threaded.** Plain numpy fancy indexing (`pos_np[cand_idx_all]`)
   doesn't parallelize across cores on its own - this file's *own* comment on `kdtree_query`
   already documented needing `workers=-1` for exactly this reason ("~4.3s single-threaded vs
   ~0.3s parallel"). Added `_gather_candidate_positions_numba` (same `numba.njit(parallel=True)` +
   `numba.prange` pattern this file's existing `_fps_batched_numba` already uses for the FPS step)
   and routed the gather through it. `candidate_gather` dropped from ~0.22s to **~0.01-0.02s**
   (cycle 2 in a fresh run shows inflated numbers from one-time numba JIT compilation, not a
   per-cycle cost - by cycle 3 onward it's fast).

Combined effect, measured on the same 6-chunk growth sequence used throughout this file's testing:
`t_collate` for the final ~16.8k-face step went from **0.80s -> 0.35-0.38s**, and total cycle time
(`t_total`) went from **~0.93-0.96s -> ~0.45-0.47s** - roughly **2x faster**, verified with zero
change to correctness (`test_live_segmenter.py`'s dedup/cap/overlap assertions all still pass
unchanged).

**What's left, now that the gather is fixed**: at ~17k faces, `kdtree_query` (~0.13s) and
`fps_dilation2_k1800` (~0.10-0.13s) are now the two largest remaining collate costs, roughly
comparable in size to each other and together still bigger than the model's own ~0.06s. Both are
already parallelized (`workers=-1` / `numba.prange`) - further speedup there would most likely
mean reducing how much work they do (smaller `dilation_ks`, e.g.), which is a *trained* model
hyperparameter, not something safe to change without retraining. **The GIL-contention and
periodic-large-chunk suspects are instrumented but not yet re-tested against real hardware** - the
`[GAP]`/`recv_blocked` reader-thread logging and the `[SLOW]` chunk tagging both need an actual
live run to produce real data, which hasn't happened yet as of this finding.

**Status: two concrete, verified bugs fixed (candidate_gather: ~20x faster; overall cycle time
at ~17k faces: ~2x faster). NOT yet re-verified against real hardware - that's the next
real-hardware test to run, and this time the debug log itself will show exactly which stage
(pipeline/collate/model/GIL-gap) is dominant instead of needing to guess from architecture alone.**

### Finding #8: on real hardware, voxel dedup is much weaker than synthetic testing suggested, so caching-and-re-merging-every-cycle grows unboundedly with elapsed scan time (RESOLVED)

Re-tested Finding #7's fixes against real hardware (a 22-cycle run, ~9.2M raw faces observed by
the end). Two things showed up in the same log:

**The good news first: Finding #7's fixes hold up perfectly.** `model_forward` stayed at ~0.08s
and `collate` stayed at ~0.35-0.45s for the *entire* run, even as raw data climbed into the
millions of faces - confirms `candidate_gather` and the model's on-the-fly `cdist` are genuinely
fixed, not just fast in the synthetic test.

**The bad news: `decimate_chunks` - a pipeline stage, not collate or the model - climbed from
0.075s (cycle 1) to 8.524s (cycle 21), and the reader-thread `[GAP]` events (Finding #7's GIL
instrumentation) got as large as 4.467s, lining up *exactly* with these bursts. This is now
confirmed, not just plausible: heavy CPU work in the worker thread (pyfqmr's decimation calls,
which are blocking C++ calls that hold the GIL for their full duration) starves the reader thread
long enough to explain the "connection lost, will reconnect" cycling this same log ends with.

Digging into *why* `decimate_chunks` was climbing revealed the actual root cause, and it's not
what Finding #6 assumed. The previous design (`_decimated_chunks`/`_deduplicated_chunks`) cached
every chunk's decimated+deduped mesh forever and re-merged the *entire* cached set from scratch
every cycle via `merge_meshes(list(deduped.values()))` - safe as long as voxel dedup kept the
total near a full arch's true covered area (~16,000-18,000 faces at target_density=5.0), which
held in synthetic testing (repeat observations there are *exact* duplicate geometry, trivially
landing in the same voxel). Real hardware breaks that assumption: each scanning pass
independently re-triangulates the same physical surface, so a repeat observation is *close to*
but not identical to the earlier one, and a real fraction of it lands in a different voxel and
survives dedup. Measured live: "after voxel dedup" reached **221,825 faces** by cycle 21 - more
than 10x the true covered area a full arch needs. `max_model_faces` (Finding #5) still protected
the *model* from ever seeing more than 20,000 of those faces, but everything *before* the cap -
re-decimating new chunks, then re-merging an ever-larger cached set from scratch - had no such
bound and got more expensive every single cycle, for the whole scan.

**Fix: fold each chunk into a persistent accumulator exactly once, then discard it.**
`LiveSegmenter._decimated_chunks`/`_deduplicated_chunks` (which cached forever and re-merged
everything every cycle) were replaced with `_fold_new_chunks`/`_fold_in`
(`testing/realtime/mesh_viewer_segmented.py`):
- Each mesh_id is decimated + voxel-deduped **once**, the moment it's new or changed. Its
  newly-kept faces are merged onto a single running `self._accumulated` mesh
  (concatenate-and-offset, same operation as `merge_meshes` but applied incrementally). After
  that, the chunk's raw vertices/triangles are evicted from `self.chunks` - Finding #1 already
  confirmed real mesh_ids are never updated again after they first arrive, so nothing is lost by
  forgetting it; its contribution is now permanent in `self._accumulated` and `_claimed_voxels`.
  Only two tiny per-mesh_id records survive eviction (`_folded_signature`, `_raw_faces_seen` - an
  int/tuple each), not the full geometry.
- This makes each cycle's fold cost proportional to what's *new* since the last cycle, not to the
  whole scan's history - the exact same principle Finding #3 already established for per-chunk
  decimation, now applied one stage later, to the merge step itself.
- A second, independent bound: `_fold_in`'s own concatenate is O(current accumulator size) per
  call, so even the *fold* step alone could still grow unboundedly over a long scan given how
  weak real dedup turned out to be. `self.retention_ceiling` (10x `max_model_faces`, so 200,000 by
  default) bounds `self._accumulated` itself via the same cheap `cap_face_count` random subsample
  already used for the final model-input cap - engages only in the same pathological case that
  cap already exists for, not on ordinary cycles.
- All of this cycle's newly-decimated+deduped pieces are merged together *once* via
  `merge_meshes` (cheap now - it only ever sees this cycle's small new pieces, never the whole
  scan) before a single `_fold_in` call, rather than one `_fold_in` per new chunk - otherwise a
  catch-up cycle folding in many chunks at once would reintroduce an
  O(chunks_this_cycle x accumulator_size) cost.
- The external `process_once()` stats-dict contract is unchanged (`raw_faces`, `decimated_faces`,
  `dedup_faces`, `down_faces`, etc. all mean exactly what they meant before - now computed via
  incremental running counters instead of by re-scanning every known chunk each cycle) - so
  `test_live_segmenter.py` and the GUI's `_apply()` needed zero changes, only `LiveSegmenter`'s
  internals did. `test_live_segmenter.py`'s dedup/cap/overlap/disjoint assertions all still pass
  unchanged, and its debug output now shows every chunk correctly evicted from `self.chunks`
  immediately after folding (`"0 chunk(s) still resident"` after every step).

**Status: fixed and verified headlessly (zero regressions in `test_live_segmenter.py`, chunk
eviction confirmed working). NOT yet re-verified against real hardware** - that's the next
real-hardware run to check: whether `decimate_chunks` now stays flat across a long scan instead
of climbing, and whether the `[GAP]`/"connection lost" cycling this finding's log ended with is
gone as a result.

### Finding #9: pyfqmr's `simplify_mesh()` does not release the GIL - this CONFIRMS the GIL-contention hypothesis (Finding #4), not just "supports" it

Re-tested Finding #8's fold-based rewrite against real hardware again (arch=upper, 20-cycle run).
Confirms Finding #8 is working as designed: `_fold_new_chunks` debug lines show chunks folded in
exactly once and evicted (e.g. cycle 20: "4 chunk(s) folded in ... (0 chunk(s) still resident)"
even right after folding in three 100k-200k-triangle re-batched chunks back to back), and
Finding #7's collate/model fixes stayed flat (collate ~0.38-0.45s, model ~0.08s) for the entire
run regardless of total scan size - both previous fixes hold up.

**What's still slow, and why:** `decimate_chunks` climbs again in this run (0.07s -> 5.7s by
cycle 19) - but this time NOT from stale re-processing (that's fixed). Two real, additive causes:
1. Several periodic large re-batched chunks landed close together: mesh_id 703 (107k triangles),
   840 (201k), 841 (170k), and 842 (187k) all arrived within about 4 cycles, each taking
   0.7-1.3s (`[SLOW]`-tagged) to decimate - three of them (840/841/842) landed in the *same* fold
   pass (cycle 20), accounting for ~3.4s of that cycle's 3.5s `decimate_chunks` cost by
   themselves. Finding #6 assumed these were rare, isolated, non-compounding events; this run
   shows they can cluster.
2. The worker's skip-if-busy loop snapshots and processes everything queued in one go; a slow
   cycle lets more chunks queue before the next cycle starts, so the next cycle has more to do -
   not a runaway spiral in this run (cycle 20 dropped back to 3.5s and fully caught up), but a
   real feedback effect worth knowing about.

**The important correction:** partway through analyzing this run's `[GAP]`/`recv_blocked` data, it
looked like the GIL-contention hypothesis (Finding #4) might be *wrong* - every large gap in this
log has `recv_blocked_s` ≈ `gap_since_last_frame_s` (e.g. `gap=2.180s` before mesh_id=840 with
`recv_blocked=2.180s`), which looks at first like "the reader was genuinely blocked waiting on the
socket, not fighting the worker for the GIL." That reasoning is incomplete, and checking it against
`pyfqmr`'s actual source before reporting it turned up why: **`Simplify.pyx`
(`site-packages/pyfqmr/Simplify.pyx`) does not wrap its `simplify_mesh()` call in `nogil`** - only
the small helper functions that copy vertex/face arrays into the C++ mesh (`setVerticesNogil`/
`setFacesNogil`) are marked `nogil`; the actual quadric-decimation work runs as plain Cython code
holding the GIL for its entire duration. CPython's socket `recv()` *does* release the GIL while
genuinely waiting on the OS for bytes - but returning from that wait back into Python bytecode
(parsing the header, updating `self.chunks`, computing `t_recv_end`) requires re-acquiring the
GIL, and a thread stuck inside a non-`nogil` C call cannot be preempted to hand it over - CPython's
periodic switch-interval mechanism only works between bytecode instructions, not mid-C-call. So
`recv_blocked_s` measures wall-clock time until `recv()` *returns*, which silently includes any
time spent waiting to reacquire a GIL held by the worker thread mid-`downsample_to_density()` -
`recv_blocked ≈ gap` does not rule out GIL contention, it's consistent with it. This is now a
verified mechanism, not a suspicion: a long `simplify_mesh()` call (confirmed non-`nogil`) directly
explains both the reader-thread stalls and, very plausibly, the "connection lost, will reconnect"
cycling itself (whatever keeps the connection alive - TCP keepalive, an application-level
heartbeat - can miss its window if the reader thread is starved long enough).

One genuine improvement in this run: the connection dropped and successfully reconnected twice,
and the scan continued both times, unlike earlier logs where "connection lost" cycling never
recovered. Not yet clear whether that's due to Finding #8's fixes or was always possible under the
right conditions.

**Fix implemented: decimation moved to a process pool (RESOLVED).** Since `simplify_mesh()` holds
the GIL, a `ThreadPoolExecutor` would not have helped (GIL-bound calls just serialize with added
thread-switching overhead on top) - genuine isolation needs separate OS processes, each with its
own GIL. `LiveSegmenter` (`mesh_viewer_segmented.py`) now owns a `ProcessPoolExecutor` (default 6
workers, `multiprocessing.get_context("spawn")` - `spawn` deliberately, not Linux's default
`fork`, since this constructor runs after the model is already on CUDA in `main()`, and forking a
process with an initialized CUDA context is documented as unsafe). `_fold_new_chunks` submits
every new/changed chunk's decimation to the pool at once via `decimate_chunk_worker`
(`live_preprocessing.py`, new - a picklable, side-effect-free function taking/returning plain
numpy arrays, not a `trimesh.Trimesh`, to keep the pickled payload minimal) and collects results
via `as_completed` as they finish, so several chunks decimate in genuine parallel across cores
*and* the reader thread is never blocked by decimation regardless of how many chunks or how large
any one of them is. `claim_new_voxels` (dedup) still runs sequentially in the main process
afterward, since it mutates the shared `_claimed_voxels` set and first-writer-wins needs a defined
order. The pool is warmed up (a few trivial submitted tasks, waited on) inside `__init__`, before
the scan starts, so each worker's one-time import cost (numpy/trimesh/pyfqmr/torch - a few seconds
total) doesn't show up as a surprise latency spike mid-scan.

Verified in isolation (not just "should work in theory") with a controlled A/B script: a ~374k-
triangle mesh (mirroring the ~200k-triangle periodic re-batch chunks seen live) decimated once
in-thread and once via the pool, with a second thread ticking a heartbeat every 20ms throughout
both runs to measure how long the rest of the process stayed frozen:
- **In-thread (old behavior):** decimation took 1.068s; the heartbeat thread's worst gap during
  that window was **0.897s** (84% of the total time) - direct, measured confirmation that
  `simplify_mesh()` really does freeze every other thread in the process for nearly its entire
  duration, exactly as `Simplify.pyx` predicts.
- **Pooled (fixed):** decimation took 1.066s (same wall-clock cost - no compute is saved by
  moving process, as expected) but the heartbeat thread's worst gap was **0.020s** - its normal,
  undisturbed cadence. The main process was never blocked at all.

Also re-verified end-to-end via `test_live_segmenter.py` with zero regressions (dedup/cap/overlap/
disjoint assertions, per-cycle density, everything unchanged) - the process-pool plumbing
(spawn-based cross-process import of `testing.realtime.live_preprocessing.decimate_chunk_worker`)
works correctly.

**Status: implemented and verified (both in isolation and via the existing headless test suite).
NOT yet re-verified against real hardware** - that's the next real-hardware run to check: whether
`decimate_chunks` wall-clock time actually drops (it should, especially on cycles with many queued
chunks, since several now run in parallel), and whether the `[GAP]`/"connection lost" cycling is
reduced or gone as a direct result.

Deliberately not pursued for now (per discussion): capping/splitting per-cycle decimation work -
a smaller, complementary fix that bounds how long any single `simplify_mesh()` call can run
uninterrupted without requiring a process pool, at the cost of not reducing total work. Worth
revisiting only if the process pool alone doesn't fully resolve the real-hardware symptoms.

### Finding #10: process pool confirmed on real hardware; residual lag traced to single mega-chunks that can't be parallelized within one pyfqmr call (RESOLVED via spatial splitting)

Re-tested Finding #9's process-pool fix against real hardware (arch=upper, 57-cycle run, reaching
10.7M raw faces in ~53 seconds - a much faster-arriving scan than earlier test runs). Confirms the
fix works as intended:
- `t_total` stayed under ~2s for the *entire* run, vs. 5-9s at similar/smaller scales before -
  e.g. cycle 57 reached 10.7M raw faces at `t_total=1.712s`, where the Finding #9 run was already
  at 5-6s by ~5.8M raw faces.
- Reader-thread `[GAP]` events dropped from ~15-20 per run (several 1-4s) to 8 total, mostly under
  0.5s.
- The run's eventual disconnect (repeating "connection lost" with no successful reconnect) was
  confirmed by the user to be the scanner's physical Stop button, not a bug - it happened ~3
  seconds after the last `process_once()` cycle had already finished and gone idle, with no
  correlation to any of our own CPU-heavy work, unlike every previous disconnect this session.

The user still noticed real lag late in the run, and the log explains exactly why: the pool
parallelizes decimation *across* chunks arriving in the same cycle, but a *single* chunk's own
`pyfqmr.simplify_mesh()` call can't itself be split across cores - one huge periodic re-batched
chunk arriving alone still pins one cycle's wall time to however long that one call takes on one
core. Directly visible in the log: cycle 56 folded in exactly one chunk, `decimate wall=1.448s
(1.427s worker-seconds across 1 chunk(s))` - no parallelism possible, since there was only one
thing to parallelize across. Cycle 57 (2 chunks) show the pool working as designed: `decimate
wall=1.179s (2.241s worker-seconds across 2 chunk(s))` - true concurrent speedup, just not
available when only one big chunk shows up.

**Fix: spatially split any chunk above `split_threshold_faces` (40,000 raw faces) into up to
`pool_workers` pieces before submitting to the pool.** New `spatial_split()`
(`live_preprocessing.py`) reuses the same PCA-axis-ordering approach `test_live_segmenter.py`
already used to build realistic test chunks from a full mesh (spatially-coherent pieces, not a
random face subset - this project's own earlier testing found random subsets fragment badly
under pyfqmr's decimator). `_fold_new_chunks` now submits each piece as its own future, and only
proceeds to dedup/fold a given mesh_id once *all* of its pieces have come back, merging them
first (`merge_meshes` - cheap, bounded by the already-decimated piece sizes) into one chunk exactly
as before.

Verified in isolation on the same real full-arch test mesh used throughout this project (186,860
raw faces, comparable to the largest periodic re-batched chunks seen live): single-call
decimation took 0.370s; spatially split and decimated in parallel, wall time dropped to 0.205s
(2 pieces, 1.80x), 0.097s (4 pieces, 3.80x), and 0.064s (6 pieces, 5.80x) - near-linear speedup
with piece count. Output quality/density was preserved almost exactly across all of them (17,906
faces at 2/4 pieces, 17,904 at 6 - target density 5.00 faces/mm2 either way, matching the
single-call result to within rounding). Also re-verified via the full `test_live_segmenter.py`
suite (its `max_model_faces` cap test feeds the entire 186,860-face mesh as one chunk, exercising
the split path directly) - zero regressions.

**Status: RESOLVED - verified on real hardware.** Re-tested (arch=upper, 79-cycle run, reaching
14.5M raw faces - the largest scan tested this session, in ~82 seconds). `t_total` stayed in the
0.6-1.0s range for the *entire* run, never exceeding 1.05s even at the highest raw face counts -
no single-mega-chunk spike anywhere close to the ~1.4-2.0s worst cases seen before this fix.
Direct confirmation the split is doing its job: mesh_id=2079 arrived as a single 322,070-face
chunk (the largest of any run this session) and was spatially split into 6 pieces, decimating in
`wall=0.429s` despite `1.390s worker-seconds` - roughly a 3.2x effective speedup for that one
chunk, consistent with the isolation benchmark's scaling.

Two transient ~14s and ~8s gaps occurred mid-scan (each followed by "connection lost, will
reconnect" then a successful reconnect with no data loss) - but both happened while this process
was completely idle (finished its previous cycle, no decimation or any other work running), which
rules out GIL contention as the cause this time. This looks like a genuine, transient network or
server-side hiccup, not a client-side bug - and the existing reconnect logic already recovers from
it cleanly on its own, so there's nothing to fix here unless it starts happening often enough to
matter. The run's *final* disconnect (repeating "connection lost" with no further recovery) was
confirmed by the user to be the scanner's physical Stop button, same as the Finding #10
investigation run.

### Finding #11: guidance target can jump 40+mm and reverse direction when the left sweep completes (RESOLVED)

Re-tested the guidance arrow against real hardware after the on-mesh-arrow/split-view/snapshot
fixes (97-cycle run, arch=upper, full arch successfully identified by the end - `guidance -
full arch regimen complete` from cycle 77 onward). Performance held up well throughout (0.5-0.9s
per cycle even at 13.5M raw faces, 14 `[GAP]` events of similar magnitude to previous runs, none
correlated with anything on our side - consistent with Finding #10's fixes still holding).

But the guidance target itself has a real bug, both confirmed in the logged data and visually in
the snapshot renders. At cycle 41, target was label1 (8.0mm away, compass NE) - close, expected.
At cycle 42, target instantly became label11 (47.7mm away, compass SW) - the opposite side of the
arch. The histogram data pinpoints it exactly: label 1's face count was 25 at cycle 41 (just under
`guidance_min_faces=30`) and jumped to 396 at cycle 42 - crossing the confidence threshold in one
cycle. The moment that happens, `_compute_guidance`'s `first_missing(_GUIDANCE_LEFT_SEQUENCE)`
returns `None` (all of 8..1 now "identified"), so it immediately checks
`_GUIDANCE_RIGHT_SEQUENCE` instead - but labels 9 and 10 had *already* been confidently identified
for many cycles by then (real coverage, not noise - the physical scan/model naturally covered some
right-side surface while the user was still finishing the left side), just never surfaced as an
active target because the left sweep always takes strict priority. So the very first cycle the
right sweep is ever checked, it's already several teeth in, and the target snaps straight to
label11 with no transition through the center - a large, disorienting jump, confirmed visually in
`testing/logs/snapshots/mesh_viewer_segmented_20260815_111441/` (rendered via
`visualize_snapshots.py`): cycles 39-41 show a short arrow at the growing edge; cycle 42 shows the
identical current position with the arrow now spanning almost the entire arch to the opposite end.

This doesn't match the user's stated regimen either, which explicitly includes an actual "come
back to 7/8 (center)" transition as a real physical step - the current implementation has no
concept of that transition at all, since `_GUIDANCE_LEFT_SEQUENCE`/`_GUIDANCE_RIGHT_SEQUENCE` are
tracked as fully independent sequences with no coordination between them beyond "check left fully
before ever checking right."

**Fix implemented: gate the sweep transition on the scanner's own recent position, not just on
prediction confidence** (the third candidate direction, chosen over the other two - rate-limiting
target movement would smooth the jump but still skip the "return to center" step the user's actual
regimen includes; the explicit-waypoint idea is effectively what this does). `_compute_guidance`
now checks, once every left-sweep tooth is confidently identified: is `_last_chunk_centroid` (the
scanner's own recent position) actually within `guidance_center_return_threshold_mm` (20mm
default) of the 8/9 midline position? If not, the target stays `9` - the right sweep's own start,
and the regimen's literal "come back to 8/9" instruction - pointing at its already-known position
rather than jumping ahead. Only once the scanner has genuinely returned near center does the
transition to `first_missing(_GUIDANCE_RIGHT_SEQUENCE)` proceed (which may still land past 9/10 if
they were already covered - that's correct once the scanner is actually back there, just not
before).

Verified with a direct reproduction of the real bug scenario (synthetic mesh, scanner position
held at the left sweep's last tooth while 9/10 are already-identified far away): before the fix
this setup would have jumped straight to label 11 (the real 47.7mm case); with the fix, target
correctly holds at label 9 (pointing at the 8/9 midline, ~70mm from the scanner's current
position in this test) instead. Also verified: the transition proceeds correctly to the true
first-missing right-side tooth once the scanner position is moved to within the threshold: the
20mm boundary itself gates correctly on both sides (19.9mm proceeds, 20.1mm holds); and all
previously-verified cases (nothing identified, single-anchor extrapolation, regimen-complete
detection) still pass unchanged. Re-verified via the full `test_live_segmenter.py` suite - zero
regressions. **NOT yet re-tested against a real live scan** - the next run should show the target
holding at label 9 with a "come back to center" arrow instead of jumping, whenever the left sweep
finishes ahead of where the scanner physically is.

- **Prediction-debugging snapshot methodology**, for a new live investigation question: does the
  model correctly identify a specific tooth (e.g. #8/#9) right away from a small occlusal patch,
  or does it confuse it with a neighbor at first and "space out" as the accumulated scan grows?
  `LiveSegmenter._save_snapshot` (`mesh_viewer_segmented.py`) dumps (mesh vertices/faces, per-face
  predicted labels, `area_mm2`, dilation gate states, raw face count) to
  `testing/logs/snapshots/<run_id>/cycle_NNNN.npz` every cycle by default (`--snapshot_every`/
  `--no_snapshots` to change or disable) - `<run_id>` matches the same timestamp as that run's
  debug log, so the two can be correlated. `models/patch_dilated_tooth_seg_network.py`'s
  `forward()` now also exposes `self.last_gates` (which of the 3 area-gated dilated blocks were
  on) unconditionally, mirroring the existing `last_timings` pattern - a concrete, checkable
  mechanism for the "small patch" symptom: a patch under the smallest area threshold (40mm2 by
  default) runs with ALL dilated blocks off, i.e. local features only, exactly where visually
  similar neighbor teeth would be hardest to disambiguate.
  `testing/realtime/visualize_snapshots.py` is the offline half - loads a run's snapshot sequence
  and renders a grid of PNG panels (per-face flat coloring, same palette as the live viewer) using
  ONE fixed PCA-based 2D projection fit on the largest snapshot and reused for every panel, so the
  same physical region lines up across cycles instead of each frame reprojecting independently.
  Also prints a plain-text per-cycle table (area/gates/raw-faces/predicted-class-set) for a
  quantitative read without opening any image. Deliberately does not import
  `mesh_viewer_segmented.py` (would drag in open3d/torch/the model stack) - only needs
  `utils/teeth_numbering.py`'s color tables, so it stays a fast, display-independent CLI tool.
  Verified end-to-end on a synthetic 15-step growth sequence first (real checkpoint, real mesh),
  which confirmed the render pipeline works but - as expected - didn't reproduce the gates-off
  regime (its chunks were already past all three thresholds from step 1).

  **A real live run (92 cycles, arch=upper) answered the actual question.** Cycle 1 (area=58mm2,
  gates=`100` - only the first dilated block on, since 58mm2 clears the 40mm2 threshold but not
  180/360) rendered as a chaotic speckle of 5 different tooth classes scattered across the tiny
  starting patch - visually exactly the "neighboring predictions" symptom described. Cycle 2
  (area=1134mm2, gates flip to `111` - all three dilated blocks now on) resolves almost instantly
  into two clean, well-separated blobs with a sharp boundary between them, and stays that clean
  for the rest of the 92-cycle, 15.6M-raw-face scan - each newly-added tooth at the growing edge
  shows brief initial fuzziness, but already-established teeth in the interior never destabilize
  again. Rendered at `testing/logs/snapshots/mesh_viewer_segmented_20260815_102157/`.

  **Conclusion: this is not a model-quality issue, and does not by itself motivate a retrain.**
  It's the by-design behavior of Component 4's area-gated dilation (`claude_code_prompt.md`):
  below the gating thresholds, the network deliberately runs on local-only features (to avoid
  wasting compute on tiny patches), which is inherently ambiguous between visually similar
  neighbor teeth - and the moment enough area accumulates, it resolves within a single cycle.
  Practically: expect brief per-tooth fuzziness right at the scanning frontier as a normal,
  self-correcting part of how this architecture works, not a bug to chase.

- **Real-time motion-guidance arrow**, for the user's fixed scanning regimen: start at the two
  central incisors (internal labels 8 and 9 - position-in-arch is what these internal labels
  encode, identical regardless of upper/lower arch), sweep one side down to that side's 3rd molar
  (label 1), then the other side up to its 3rd molar (label 16). `LiveSegmenter._compute_guidance`
  (`mesh_viewer_segmented.py`) makes the "what's the next tooth" question fully deterministic
  given that fixed regimen - no coverage-gap/frontier-detection heuristics needed (considered and
  explicitly rejected: it doesn't use the model's actual purpose - tooth ID - at all, and the
  voxel dedup grid is too coarse/artifact-prone to trust for "is this really unscanned" on its
  own). The target tooth's real-world position isn't known until it's actually scanned, so it's
  estimated by extrapolating from the 1-2 nearest already-identified teeth in the same sweep
  direction (2 anchors: continue their local spacing/direction one more step; 1 anchor: point
  toward that landmark directly). "Current position" is the most recently folded chunk's own
  centroid (`_last_chunk_centroid`, tracked in `_fold_new_chunks`) - the freshest real-world
  position data available, since `mesh_wire.py`'s protocol carries no actual scanner pose
  (checked directly against its `parse_mesh_payload` - only geometry, no pose/orientation field
  exists to use instead). The 3D guidance vector is projected onto a (3,2) PCA basis fit ONCE
  (first cycle with enough vertices) and reused for the rest of the scan, so the arrow's frame
  stays fixed instead of drifting cycle to cycle.

  **First version showed this only as GUI panel text - not visible enough to actually use while
  moving the scanner, per direct feedback after a real run.** Fixed: `guidance_arrow_mesh()`
  builds a real `o3d.geometry.TriangleMesh` arrow (Open3D's `create_arrow`, oriented via
  Rodrigues' rotation formula onto the true 3D guidance vector - not the 2D-projected one used
  for the compass label) drawn directly in the live scene as a `"guidance_arrow"` geometry,
  added/replaced every cycle right on top of the mesh being actively scanned. Length is clamped
  to a fixed visual range (8-25mm) regardless of the true target distance, so it stays legible
  whether the target is a few mm or several cm away. The text readout (`guidance_label`) is now a
  secondary reference alongside it, not the only signal.

  **This also motivated a bigger GUI change: split view.** The single shared viewport (raw and
  predicted layers overlaid, toggled via checkboxes) is gone - `SegmentedViewerApp` now has two
  independent `SceneWidget`s side by side (`scene_widget` left / `pred_scene_widget` right, each
  with its own `Open3DScene`, own auto-fit camera and bounds tracking - `_fit_camera(which, ...)`
  takes `'raw'`/`'pred'` rather than being shared). Left pane: the live raw scan (unchanged
  behavior) plus the guidance arrow. Right pane: LiveSegmenter's predicted-class mesh for the
  same covered voxels, replaced wholesale each cycle - always exactly what the model currently
  believes, viewable side by side against the raw scan instead of z-fighting with it in one
  viewport. `show_predictions` (no longer meaningful once it's its own always-visible pane) was
  removed; `show_raw` still toggles the raw sub-meshes within the left pane.

  Verified three ways: (1) `guidance_arrow_mesh` in isolation - correct orientation confirmed for
  both a +X and a -Z target case via Rodrigues' formula, correct `None` on a degenerate
  zero-distance case; (2) the split-view frame arithmetic (panel + two panes) checked standalone -
  panes are adjacent, non-overlapping, and exactly fill the remaining window width; (3) a GUI
  smoke test that actually constructs `SegmentedViewerApp` (real Open3D window/scenes, this
  environment does have GUI support) and drives `_apply_raw`/`_apply` with synthetic data -
  confirmed bounds tracking in both panes independently, and confirmed the `"guidance_arrow"`
  geometry is actually present in the raw scene after `_apply()`. Also re-verified via the full
  `test_live_segmenter.py` suite (unaffected - guidance/GUI changes are additive to the
  GUI-independent `LiveSegmenter` core) - zero regressions. The underlying sequencing logic (5
  synthetic cases: nothing-identified, single-anchor pointing, two-anchor extrapolation with
  exact distance math, left-to-right sweep transition, regimen-complete detection) was verified
  earlier and is unchanged by this GUI rework. **NOT yet tested against a real live scan** -
  that's the natural next check, especially since the guidance's accuracy is coupled to how
  quickly/reliably the model identifies each tooth (see the snapshot-methodology finding above -
  resolves within one cycle in practice, but a target extrapolated from a still-settling tooth
  could briefly point slightly wrong).

  **Follow-up after real use (same session):** two GUI bugs and one investigation feature, all
  fixed/added:
  1. The legend's leading "■" bullet (and originally the compass text's Unicode arrows) rendered
     as "?" - Open3D's GUI font doesn't cover those codepoints. Fixed by dropping the bullet
     entirely (the whole legend line is already colored via `text_color`, so it was purely
     decorative) and switching `_COMPASS` to plain ASCII compass abbreviations (`"NE"`, `"SW"`,
     etc.) - the real directional signal is the 3D arrow geometry, not this text.
  2. The arrow often ended up embedded in/under the mesh, hard to see - direct feedback after a
     real run. Fixed by offsetting both arrow endpoints 4mm along the mesh's own average face
     normal (not a fixed world axis - robust regardless of scan orientation) before building the
     arrow geometry, so it floats just above the occlusal surface instead of sitting at surface
     level.
  3. **Guidance is now logged alongside every snapshot** (`_save_snapshot` gained a `guidance`
     parameter; new `guidance_state`/`guidance_target_label`/`guidance_target_universal`/
     `guidance_distance_mm`/`guidance_compass`/`guidance_current_pos`/`guidance_target_pos`
     fields, `"none"`/`"done"`/`"active"` states, NaN/-1 filled when not applicable - old
     snapshots without these fields still load fine via `visualize_snapshots.py`, which defaults
     them to `"none"`). `visualize_snapshots.py` now draws the actual arrow (`ax.annotate` with
     `arrowprops`, same fixed projection as the mesh) directly on each panel, and the per-cycle
     text table gained `guidance_target`/`distance_mm` columns - the combined "scan + prediction
     + arrow, all in one place" methodology the user asked for.

  **Already found something with it**: a synthetic growth-sequence smoke test showed the arrow's
  extrapolated distance jump from 6-13mm (cycles 4-7) to 22mm at cycle 8, visibly overshooting
  past the actual mesh boundary into empty space in the rendered panel. Not yet root-caused - the
  two-anchor extrapolation (`p_last + (p_last - p_prev)`) has no distance cap, so if the two
  anchor teeth it's extrapolating from happen to be unevenly spaced (e.g. one of them is a
  still-growing/partial patch near the edge, not yet at its true center), the projected step can
  overshoot substantially. Worth investigating against a real run's guidance-logged snapshots
  next, now that the tooling to do so exists.

- **Independent raw-scan layer, decoupled from prediction** (`SegmentedViewerApp.push_raw`/
  `_apply_raw` in `mesh_viewer_segmented.py`). Requested explicitly: scan accumulation should
  stay fast and immediate even while predictions lag behind. Every incoming mesh frame is now
  displayed as its own Open3D geometry (`raw_<mesh_id>`) the moment it arrives, with the
  scanner's own vertex colors, straight from the reader thread via `post_to_main_thread` -
  completely bypassing `LiveSegmenter`/the worker thread/the model. This mirrors
  `mesh_viewer_linux.py`'s `MeshViewerApp._apply()` pattern exactly (same per-mesh_id
  replace-in-place, same unlit-material/vertex-color approach) but reimplemented locally so this
  viewer doesn't need that file running alongside it just to see the scan accumulate live.
  The predicted layer (`"segmented"`) is unchanged in how it's built, just now optional: two new
  checkboxes, "Show live scan (raw colors)" (default ON) and "Show predictions (may lag)"
  (default OFF - both layers occupy the same real-world mm coordinate space, so showing both at
  once is valid but will z-fight somewhat where their two different triangulations of the same
  surface nearly coincide). Camera auto-fit was refactored to track running bounds across BOTH
  layers together (`_region_bounds`, same incremental min/max-over-all-regions approach
  `mesh_viewer_linux.py` already uses), so it reflects the true accumulated scan extent
  regardless of which layer(s) are currently visible.
  Verified with a direct (no-network) smoke test driving `push_raw`/`_apply_raw`/the toggle
  callbacks/a real `LiveSegmenter.process_once()` cycle against a live Open3D GUI window
  (`gui.Application.instance.run_one_tick()` pumped manually since `.run()` was never called) -
  all assertions passed with zero exceptions (raw geometries added/toggled correctly, predicted
  layer populated and toggled correctly, shared camera fit didn't error).

- **Universal tooth numbers**: `utils/teeth_numbering.label_to_universal_number(labels, arch)`.
  Needs an explicit `arch` ('upper'/'lower') - there's no way to recover which arch a label came
  from the label alone. `mesh_viewer_segmented.py --arch {upper,lower}` (optional - omit it and
  you get tooth type names like "central_incisor" with no number instead).
- **Color legend + floating 3D labels**: side-panel legend (`gui.Label.text_color` matched to
  the mesh's own palette), and per-predicted-tooth floating 3D text (Open3D's
  `SceneWidget.add_3d_label`/`remove_3d_label`) at each class's face centroid, rebuilt every
  cycle. Toggleable via a "Show tooth-number labels" checkbox.
- **Debug logging**: `[debug:segmented]`, `[debug:live_preprocessing]`, `[debug:relay]` prefixes
  throughout - chunk arrivals, cache hit/miss with before/after counts at every stage
  (decimation AND dedup), full per-cycle timing breakdown, prediction class histograms, GUI apply
  timing, relay throughput/fan-out stats. This is what made Findings #1-2 possible to diagnose
  from a single pasted terminal log without needing to reproduce the issue interactively.

### Finding #12: middle dilated block's `dilation_k` had silently drifted from the paper's 900 to 600 (RESOLVED)

Raised by the user during an architecture walkthrough (layer-by-layer comparison against the
original reference model, requested to inform the upcoming from-scratch retrain). Verified
directly against `models/dilated_tooth_seg_network.py:31-39` (the untouched original/paper
reference): the three dilated blocks' `dilation_k` values are **200 / 900 / 1800**.
`models/patch_dilated_tooth_seg_network.py`'s default had them as **200 / 600 / 1800** - the
middle block's candidate-neighborhood size (before FPS samples 32 of them) was 600, not 900.

Traced as far as possible without git history (this repo has no commit-level record of when the
value changed): `claude_code_prompt.md` (the spec that drove Component 4's build) explicitly says
"the local edge-conv backbone is good and should be preserved," and lists exactly what to change -
norm layers, STN removal, variable N, area-gated dilation, `num_classes` - dilation_k is never
mentioned, implying it should have carried over unchanged from the original. Instead,
`TRAINING_CONCERNS.md` itself had a line asserting "200/600/1800 - tuned for the paper's fixed
N=16,000 setting" - a factually incorrect claim (re-verified directly against the actual original
file), which had reinforced the drift by documenting it as if it were the paper's own value rather
than catching it. Checked and ruled out both of the user's hypotheses for why it might have
changed: (a) the 40/180/360mm² area-gating thresholds are a completely separate mechanism
(controls whether a block runs at all, not its candidate-neighborhood size once it does) with no
numerical relationship to 600 anywhere; (b) `TRAINING_CONCERNS.md` Finding #1's entire O(N²)-cost
discussion never once proposes reducing `dilation_k` as a mitigation - its actual fix was CPU
KD-tree + numba precomputation - and a `dilation_k = min(dilation_k, N)` runtime clamp already
existed in both the original and patch versions to handle small patches, so there was no
compute-driven need to lower the permanent default either. Best available conclusion: an
unintentional value substitution during Component 4's original implementation (most likely a
transcription slip porting the three blocks' constructor calls), never caught because it doesn't
crash or otherwise look obviously wrong - just silently trains a modestly different architecture
than the paper's validated one.

**Fix: restored `dilation_k=900` for the middle dilated block everywhere it was hardcoded** -
`models/patch_dilated_tooth_seg_network.py`'s `dilation_ks` default, `models/patch_lightning_module.py`'s
own separate default (the actual training entry point - `train_patch_network.py` has no CLI
override, so this was the value any new training run would actually have used), and
`models/patch_collate.py`'s `PatchCollator.__init__` default (only relevant when a collator isn't
built via the normal `PatchCollator.for_network(model)` path, which reads the value straight off a
model instance instead). `testing/test_patch_model.py`'s generic collate-correctness test fixtures
were also updated from 600 to 900 for consistency, though their assertions never depended on
matching the model's real config either way. `TRAINING_CONCERNS.md`'s incorrect line corrected in
place rather than left standing.

**Does NOT affect the checkpoint used throughout this entire live-testing session** - Lightning's
`load_from_checkpoint` restores a checkpoint's own saved `dilation_ks` (200/600/1800, via
`save_hyperparameters()`) regardless of the class's current default, so `mesh_viewer_segmented.py`
(which reads `dilation_ks` off the loaded model via `PatchCollator.for_network`) is unaffected -
this fix only changes what the *next* from-scratch training run will default to.

Verified via the full `testing/test_patch_model.py` suite (forward pass across N=2..20,000, area-
gating gradient isolation, GroupNorm batch-independence, precomputed-vs-on-the-fly exact-match
checks now running against the real 900 value, full pipeline timing) - zero regressions, and the
dilated-block candidate-pool match check explicitly confirms 200/900/1800 all still verify exactly
against brute-force.

## How to run it

Against real hardware, single viewer (the common case):
```bash
python3 testing/realtime/mesh_viewer_segmented.py \
  --host <WINDOWS_SERVER_IP> \
  --ckpt <path to a .ckpt> \
  --arch <upper|lower>
```
`--max_faces` defaults to 20000 (see Finding #5) - only pass it explicitly to change that cap.

Against real hardware, both viewers at once:
```bash
python3 testing/realtime/mesh_relay.py --host <WINDOWS_SERVER_IP>
python3 testing/realtime/mesh_viewer_linux.py --host 127.0.0.1 --port 8780
python3 testing/realtime/mesh_viewer_segmented.py --host 127.0.0.1 --port 8780 --ckpt <path> --arch <upper|lower>
```

Offline, no real hardware needed:
```bash
python3 testing/realtime/fake_mesh_server.py --pt_path <a processed_w5 cache file> --port 18770
python3 testing/realtime/mesh_relay.py --host 127.0.0.1 --port 18770 --local_port 18780
# then either viewer, pointed at 127.0.0.1:18780
```

Headless pipeline test (no sockets, no GUI, no real hardware):
```bash
python3 testing/realtime/test_live_segmenter.py --ckpt <path to a .ckpt>
```

## Not yet done / open items for whoever picks this up next

0. **Finding #11's guidance-jump fix has NOT been re-tested against real hardware yet.** Fixed and
   verified via a direct reproduction of the real bug scenario plus the full test suite (see
   Finding #11), but the next real scan is what actually confirms it - watch for the target
   holding at label 9 with a "return to center" arrow (instead of jumping to a far-side tooth)
   whenever the left sweep finishes before the scanner's physically back near the midline.
1. **Findings #8-10 are all now verified against real hardware** - chunk eviction (no compounding
   re-merge cost), the process pool (no GIL-blocking freezes), and spatial splitting (no
   single-mega-chunk spikes) together kept `t_total` in the 0.6-1.0s range for a full 79-cycle,
   14.5M-raw-face run, the largest and best-behaved run of this whole investigation. What's left
   open: (a) whether `split_threshold_faces=40000` and `pool_workers=6` need tuning against
   different traffic patterns (e.g. a much longer scan, or a server that sends even larger
   re-batched chunks than the ~322k-face one seen so far); (b) the occasional transient
   network/server-side connection gap (two ~8-14s gaps seen in the Finding #10 verification run,
   both while this process was idle, both recovered automatically) - not yet investigated further
   since it's infrequent and already handled gracefully, and there's no visibility into the
   server's own implementation to dig deeper. Also still open: (c) whether the raw-scan layer
   (`push_raw`/`_apply_raw`, see "Other things built this session") holds up over a long real scan
   - only verified with synthetic data and a direct (no-network) smoke test so far.
2. **`fake_mesh_server.py` still simulates the OLD (now-known-wrong) "one mesh_id grows in
   place" pattern** (see `ScanSession._run()` - it re-sends a growing snapshot of the SAME
   `mesh_id` per region). `test_live_segmenter.py` was corrected to use the real pattern (many
   small, permanent, non-updating mesh_ids); `fake_mesh_server.py` itself would be worth
   updating the same way for more realistic offline/relay testing.
3. **`mesh_relay.py`'s fan-out/catch-up logic has only been tested against `fake_mesh_server.py`,
   never against the real Windows server.** All real-hardware testing so far has used a direct
   connection (single viewer, no relay).
4. **`DilatedEdgeGraphConvBlock`'s on-the-fly fallback has a pre-existing B>1 indexing bug**
   (inherited from `models/layer.py`, documented in `TRAINING_CONCERNS.md` #1 and
   `testing/test_patch_model.py`) - not relevant here since `LiveSegmenter` always supplies
   precomputed neighbor indices for a single patch (batch size 1), but worth remembering if this
   code ever changes to batch multiple patches together.
5. **`EdgeGraphConvBlock` 2/3's on-the-fly feature-space `torch.cdist` (Finding #5) is capped,
   not eliminated.** `max_model_faces` bounds N before the model ever sees it, so the cdist stays
   affordable - but the O(N²) op itself is still there, still deliberate (feature-space kNN
   genuinely can't be precomputed on CPU - see `models/patch_collate.py`'s docstring), and still
   the practical ceiling on how large a single live patch can be. If a future need arose to raise
   `max_model_faces` well past ~20-25k, this is the wall that would be hit again.
6. **Voxel-grid dedup is a coarse, non-manifold-aware approximation** (see
   `claim_new_voxels`'s docstring) - fine for a roughly-flat dental surface at the resolutions
   involved, but worth knowing if a genuinely 3D/high-curvature scanning target were ever in
   scope.
7. **No access to `mesh_server_win.py`'s source in this environment** - understanding of the
   wire protocol comes from `mesh_wire.py` plus observed real traffic, not the server
   implementation itself. Any server-side behavior not yet observed (e.g. genuine mesh_id reuse
   under some condition not yet triggered) could still surprise this code.
