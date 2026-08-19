#!/usr/bin/env python3
"""mesh_viewer_segmented.py -- live tooth/gum segmentation on the streamed mesh.

Same Open3D GUI shell as mesh_viewer_linux.py (Start/Stop buttons, auto-fit
camera - that file is untouched, its Conn class is reused here directly), but
instead of showing raw scanner colors, it accumulates the incoming mesh_id
chunks into one mesh, downsamples it to the model's training density, runs
the trained patch model, and colors each face by predicted class.

One command, connects directly to the Windows server by default - same as
mesh_viewer_linux.py already does, nothing else needs to be running:

    python3 mesh_viewer_segmented.py --host <WINDOWS_SERVER_IP> --ckpt <path to a .ckpt>

Only use mesh_relay.py (--host 127.0.0.1 --port 8780 here instead) if you
specifically want mesh_viewer_linux.py's raw-color view running AT THE SAME
TIME as this one - two simultaneous direct connections to the real server
would risk one interfering with the other if it turns out not to support
multiple clients (unverifiable without its source); a single connection,
either viewer alone, carries none of that risk.

The actual data pipeline (accumulate -> merge -> downsample -> infer -> color)
is the LiveSegmenter class below, which has NO Open3D/GUI dependency and can
be driven headlessly (see realtime/test_live_segmenter.py) - the GUI
class just wraps it and pushes its output to the screen.
"""

import argparse
import math
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import open3d as o3d
import torch
import trimesh

import mesh_wire
from debug_log import logger, setup_logging
from mesh_viewer_linux import Conn, _have_gui

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -
# this file moved from testing/realtime/ to realtime/ (one level shallower), so this now only
# needs to strip 2 path components (file -> realtime -> root), not 3
from dataset.patch_preprocessing import PatchPreTransform  # noqa: E402
from dataset.patch_preprocessing_real_color import PatchPreTransformWithRealColor  # noqa: E402
from models.patch_collate import PatchCollator, precompute_neighbor_indices  # noqa: E402
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork  # noqa: E402
from dataset.patch_color_augmentation import GUM_RGB_HIGH, GUM_RGB_LOW, TOOTH_RGB_HIGH, TOOTH_RGB_LOW  # noqa: E402
# live_preprocessing.py is a sibling in this same realtime/ directory - bare import, same
# convention as mesh_wire/debug_log/mesh_viewer_linux above, resolves whether this file is run
# directly (its own dir is auto-added to sys.path) or imported as realtime.mesh_viewer_segmented
# (replay_guidance.py/test_live_segmenter.py both explicitly add realtime/ to sys.path themselves).
from live_preprocessing import (NO_COLOR_SENTINEL, cap_face_count, claim_new_voxels,  # noqa: E402
                                 decimate_chunk_worker, downsample_to_density, merge_meshes,
                                 mesh_to_model_inputs, mesh_to_model_inputs_with_color,
                                 spatial_split, voxel_size_for_density)
from utils.teeth_numbering import (_coarse_class_color, _coarse_class_names, _teeth_codes_upper,  # noqa: E402
                                    _teeth_color, coarse_label_to_colors, label_to_colors,
                                    label_to_coarse_label, label_to_universal_number)

# internal 1-16 label -> tooth type name (e.g. 'central_incisor') - shared across upper/lower
# (only the FDI codes differ per arch, not the position-in-arch names - see
# _teeth_codes_upper/_teeth_codes_lower), so either dict gives the same names; arbitrarily using
# _teeth_codes_upper here.
_LABEL_NAMES = {internal_label: name for _fdi, (internal_label, name) in _teeth_codes_upper.items()}

GUM_COLOR_255 = np.array([255, 192, 203], dtype=np.uint8)  # light pink - same convention as the
                                                             # offline visualizations (testing/
                                                             # visualize_patch_growth.py's GUM_COLOR)

# Motion-guidance regimen (LiveSegmenter._compute_guidance): start at the two central incisors
# (internal labels 8 and 9 - position-in-arch is what these internal labels encode, identical
# regardless of upper/lower arch, see _LABEL_NAMES above), sweep one side down to the 3rd molar
# (label 1), then the other side up to the 3rd molar on the far side (label 16).
_GUIDANCE_LEFT_SEQUENCE = (8, 7, 6, 5, 4, 3, 2, 1)
_GUIDANCE_RIGHT_SEQUENCE = (9, 10, 11, 12, 13, 14, 15, 16)
_COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]  # plain ASCII - Open3D's GUI font doesn't
                                                          # cover Unicode arrow glyphs (renders as
                                                          # "?" - see _build_legend's "■" for the
                                                          # same issue); the real directional cue
                                                          # is the 3D arrow geometry, not this text


def update_guidance_confirmation(counts, streaks: dict, confirmed: set, min_faces: int, confirm_cycles: int):
    """Finding #13 (REALTIME.md) - mutates `streaks` and `confirmed` IN PLACE. A label needs
    `confirm_cycles` CONSECUTIVE calls with counts[label] >= min_faces before being added to
    `confirmed` - once added it stays there permanently (a later dip below min_faces does NOT
    remove it; by the time a label has genuinely sustained min_faces for confirm_cycles cycles in a
    row, a later single low-count cycle is far more likely to be its own transient noise than
    evidence the earlier detection was wrong). A label that hasn't reached the streak requirement
    yet has its streak reset to 0 the instant it dips below min_faces - this is what filters out a
    momentary spike: real hardware data (2026-08-19) showed label 1 (left terminal molar) hit 31
    faces for exactly one cycle then fell back under 30 for ~5 more before genuinely climbing much
    later, and label 16 (right terminal molar) hit 34 faces for a few cycles then dropped to 0 for
    ~40 cycles before a real, sustained, 500+ face detection began - neither would reach a 3-cycle
    streak from those early spikes alone.

    Kept as a standalone function (not inlined into _compute_guidance) specifically so
    replay_guidance.py can drive it directly against saved snapshot data without needing a live
    LiveSegmenter/model/GPU - that's how this fix was validated before ever touching live
    behavior."""
    for label in range(1, 17):
        count = int(counts[label]) if label < len(counts) else 0
        if count >= min_faces:
            streaks[label] = streaks.get(label, 0) + 1
            if streaks[label] >= confirm_cycles:
                confirmed.add(label)
        else:
            streaks[label] = 0


def _debug(msg):
    logger.debug(f"[segmented] {msg}")


def colors_for_predictions(pred_labels, num_classes):
    """(N,) predicted class ids -> (N,3) uint8 RGB, same palette as every offline visualization
    in this project (light-pink gum override for the 17-class scheme; the coarse 5-class palette
    already has its own gum entry)."""
    if num_classes == 5:
        return coarse_label_to_colors(pred_labels).astype(np.uint8)
    colors = label_to_colors(pred_labels).astype(np.uint8)
    colors[pred_labels == 0] = GUM_COLOR_255
    return colors


def flat_shaded_mesh(mesh, face_colors):
    """Open3D's TriangleMesh only natively supports PER-VERTEX color, which would blend/smear
    colors across a shared vertex between two differently-predicted faces - duplicating each
    face's 3 vertices (so no two faces share a vertex) gives genuine flat per-face coloring with
    sharp class boundaries, matching what a segmentation result should look like. Returns
    (vertices, triangles, vertex_colors) ready for an o3d.geometry.TriangleMesh."""
    faces = mesh.faces
    new_vertices = mesh.vertices[faces.reshape(-1)]
    new_triangles = np.arange(len(new_vertices)).reshape(-1, 3)
    new_colors = np.repeat(face_colors, 3, axis=0).astype(np.float64) / 255.0
    return new_vertices, new_triangles, new_colors


def guidance_arrow_mesh(start, target, color=(0.9, 0.1, 0.1)):
    """A 3D arrow (Open3D TriangleMesh) from start toward target, in the same real-world (mm)
    coordinate space as the scan itself - drawn directly into the live/raw scene as an overlay ON
    the mesh being actively scanned (not a side-panel text readout - that alone wasn't visible
    enough to actually use while moving the scanner). Length is clamped to a fixed visual range
    regardless of the true distance, so the arrow stays a consistent, legible size whether the
    target is a few mm or several cm away. Returns None if start==target (nothing to point at)."""
    vec = np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    distance = np.linalg.norm(vec)
    if distance < 1e-6:
        return None
    direction = vec / distance
    length = float(np.clip(distance, 8.0, 25.0))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.06, cone_radius=length * 0.12,
        cylinder_height=length * 0.75, cone_height=length * 0.25)
    arrow.paint_uniform_color(color)
    # create_arrow() points along +Z by default - rotate that onto `direction` (Rodrigues'
    # rotation formula) before translating to `start`.
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, direction))
    v = np.cross(z, direction)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        rotation = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (v_norm ** 2))
    arrow.rotate(rotation, center=(0, 0, 0))
    arrow.translate(np.asarray(start, dtype=np.float64))
    arrow.compute_vertex_normals()
    return arrow


class LiveSegmenter:
    """Pure processing core - no GUI dependency, safe to drive headlessly. Accumulates named
    mesh_id chunks (mirroring mesh_viewer_linux.py's accumulation model exactly - see that file's
    _apply(): each mesh_id's snapshot REPLACES the previous one, chunks are not deltas).

    process_once() decimates each chunk to the model's training density INDEPENDENTLY (only when
    that chunk's own raw data has actually changed), voxel-dedups it against a persistent,
    scan-wide grid (claim_new_voxels, live_preprocessing.py - only the first face to claim a given
    voxel survives, so overlapping re-observations of the same surface don't pile up), and folds
    its newly-kept faces into a single running accumulator ONCE - after that, the chunk's raw
    vertices/triangles are discarded (see _fold_new_chunks/update_chunk): its contribution is
    already permanent in the accumulator and _claimed_voxels, and real hardware confirmed mesh_ids
    are never updated again after they first arrive, so there is nothing left to keep it for.
    Deliberately NOT "merge everything raw, then decimate the whole thing," which was the first
    version of this and froze the GUI in real testing (re-decimating the ENTIRE raw mesh from
    scratch every cycle gets slower every cycle, compounding as the scan continues).

    A SECOND problem surfaced testing against the real scanner and is why chunks are folded in
    and discarded rather than cached forever (an earlier version of this class did cache every
    chunk's decimated+deduped mesh and re-merge the whole cached set from scratch each cycle):
    real repeat observations of the same surface are independently re-triangulated each pass, not
    exact duplicates, so a meaningful fraction lands in a different voxel and survives dedup -
    measured live, "after voxel dedup" reached 221,825 faces after a few minutes of scanning (next
    to the ~16,000-18,000 a full arch at 5 faces/mm^2 should need), and the cost of re-merging that
    ever-growing cached set every cycle climbed to 8.5s by itself, with reader-thread stalls up to
    4.5s lining up exactly with those bursts (GIL contention - see mesh_wire reader() in main()).
    Folding each chunk in exactly once keeps per-cycle cost proportional to what's NEW since the
    last cycle; _fold_in additionally bounds the accumulator itself (retention_ceiling) so even a
    long scan where real-world dedup lets far more through than a full arch's true area needs
    can't make that per-cycle fold cost unbounded either.
    """

    def __init__(self, model, device, num_classes=17, target_density=5.0, arch=None, max_model_faces=20000,
                 pool_workers=6, snapshot_dir=None, snapshot_every=1, color_mode='varying',
                 guidance_confirm_cycles=8):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.target_density = target_density
        self.arch = arch  # 'upper'/'lower' - only needed for 17-class Universal tooth numbers
        # --color_mode 'flat': real captured color varies continuously across the mesh, but
        # training only ever saw ONE flat value per class-group per patch (dataset/
        # patch_color_augmentation.py's SyntheticColorPaint) - confirmed empirically (2026-08-19)
        # that variance alone costs real accuracy. Interim mitigation (not a retrain) requested by
        # the user: classify real captured color into tooth/gum (live_preprocessing.py's
        # classify_and_flatten_colors) and repaint with ONE flat value each, drawn ONCE here and
        # reused for the WHOLE scan (not redrawn per cycle) - matching training's one-draw-per-
        # patch structure; redrawing every cycle would introduce a NEW instability (the fed color
        # flickering between cycles for the same underlying geometry) that training never had
        # either. 'varying' (default) preserves the original real-color-as-is behavior, unchanged.
        self.color_mode = color_mode
        self._flat_colors = None
        if color_mode == 'flat':
            rng = np.random.default_rng()
            tooth_color = rng.uniform(TOOTH_RGB_LOW, TOOTH_RGB_HIGH).astype(np.uint8)
            gum_color = rng.uniform(GUM_RGB_LOW, GUM_RGB_HIGH).astype(np.uint8)
            self._flat_colors = (tooth_color, gum_color)
            _debug(f"LiveSegmenter init: color_mode=flat - drew tooth_color={tooth_color.tolist()} "
                   f"gum_color={gum_color.tolist()} for the whole scan")
        # Hard cap on the merged mesh actually fed to the model, independent of voxel dedup - see
        # process_once(). Confirmed via real-hardware testing (REALTIME.md): models/patch_layer.py's
        # EdgeGraphConvBlock 2/3 always fall back to an on-the-fly torch.cdist(x_t, x_t) over every
        # point (only block 1 gets a CPU-precomputed idx - see patch_dilated_tooth_seg_network.py's
        # forward() and patch_collate.py's docstring for why blocks 2/3 can't be precomputed: they
        # depend on mid-forward features, not fixed metric position). That's an O(N^2) GPU tensor
        # with no cap of its own - fine for training-scale patches, but a long live scan's merged
        # mesh can grow past what voxel dedup alone catches (large periodic server-side re-batched
        # chunks + continuous small-chunk overlap both leak some "new" faces past dedup even when
        # they mostly cover already-seen surface), and N=160k+ tries to allocate 100+ GiB for that
        # cdist. Capping here also matches the model's actual training distribution - it was never
        # trained on patches anywhere near this size regardless of memory.
        self.max_model_faces = max_model_faces
        # Bounds _fold_in's own concatenate cost (see that method) - set well above
        # max_model_faces (10x) so it only engages in the same pathological case the final
        # model-input cap already exists for (real-world voxel dedup letting far more through
        # than a full arch's true covered area needs - Finding #8, REALTIME.md), not on every
        # ordinary cycle.
        self.retention_ceiling = max(max_model_faces * 10, max_model_faces + 1)
        # Auto-detected from the loaded checkpoint's own saved hparams (feature_dim=27 for a
        # color-trained model, dataset/patch_preprocessing_color.py's PatchPreTransformWithColor)
        # rather than a separate CLI flag - avoids the checkpoint and the transform silently
        # disagreeing with each other (feeding a 24-dim-trained model 27 dims, or vice versa).
        self.use_color = getattr(model.hparams, 'feature_dim', 24) == 27
        self.transform = PatchPreTransformWithRealColor() if self.use_color else PatchPreTransform()
        # k/dilation_ks read straight off the loaded checkpoint's own network (PatchCollator.
        # for_network), not passed separately - a live viewer using a different k than the
        # checkpoint was trained with would silently produce garbage neighbor graphs
        self.collator = PatchCollator.for_network(model.model, seed=None)
        self.voxel_size = voxel_size_for_density(target_density)
        _debug(f"LiveSegmenter init: num_classes={num_classes} target_density={target_density} "
               f"voxel_size={self.voxel_size:.3f}mm arch={arch} k={self.collator.k} "
               f"dilation_ks={self.collator.dilation_ks} device={device} max_model_faces={max_model_faces} "
               f"retention_ceiling={self.retention_ceiling} use_color={self.use_color} "
               f"color_mode={self.color_mode}")
        self.chunks = {}
        # mesh_id -> last known triangle count. Kept forever (tiny - one int per mesh_id) even
        # after a chunk's full vertex/triangle data is evicted from self.chunks, so raw_faces
        # (a whole-scan cumulative stat) stays correct without needing the raw data around.
        self._raw_faces_seen = {}
        self._raw_faces_total = 0
        self._decimated_faces_total = 0  # cumulative per-chunk decimated output, whole scan
        self._raw_area_total_mm2 = 0.0  # cumulative decimated-chunk area, whole scan
        # mesh_id -> (vertex_count, face_count) already folded into self._accumulated. Kept
        # forever (tiny) so a chunk is never reprocessed once its contribution is permanent -
        # see _fold_new_chunks.
        self._folded_signature = {}
        self._accumulated = None  # trimesh.Trimesh - permanently kept faces, bounded by retention_ceiling
        self._claimed_voxels = set()  # persists for the whole scan - never cleared, never shrinks
        self._lock = threading.Lock()
        self._cycle = 0  # incrementing per-process_once() id, tags every TIMING debug line

        # Motion-guidance state (_compute_guidance) - see that method's docstring for the fixed
        # scanning regimen this implements (start at the two central incisors, sweep one side to
        # the 3rd molar, back to center, sweep the other side).
        self._last_chunk_centroid = None  # updated in _fold_new_chunks - "where is the scanner
                                           # right now" proxy, since the wire protocol carries no
                                           # actual pose data
        self._guidance_projection = None  # fixed (3,2) PCA projection, fit once and reused for
                                           # the whole scan so the arrow's frame doesn't drift
        self.guidance_min_faces = 30  # a label needs at least this many predicted faces to count
                                       # as "confidently identified" for guidance purposes
        # Finding #11 (REALTIME.md): how close (mm) the scanner's own recent position must be to
        # the 8/9 midline before the guidance target is allowed to move from the left sweep to
        # the right one - prevents a large, disorienting jump straight to whatever the right
        # sweep already silently progressed to while the scanner was still physically on the left.
        self.guidance_center_return_threshold_mm = 20.0

        # Finding #13 (REALTIME.md): _compute_guidance used to rebuild its "confidently identified"
        # set fresh from ONLY the current cycle's prediction counts - no memory across cycles. Real
        # hardware showed this made guidance flicker on brief, noisy misclassification spikes: label
        # 1 (left 3rd/terminal molar) hit 31 faces for ONE cycle while the scanner was still on the
        # 1st molar, immediately marking the whole left sweep "done" and jumping the target to the
        # right side - then flickering back once that label's count dropped below threshold again a
        # few cycles later. guidance_confirm_cycles requires a label to stay >= guidance_min_faces
        # for this many CONSECUTIVE cycles before counting as found - once genuinely confirmed it
        # stays confirmed (a later dip doesn't un-confirm it - by that point it's overwhelmingly
        # likely real, not noise). See update_guidance_confirmation() below for the actual logic and
        # replay_guidance.py for how this was validated against saved real-hardware snapshots before
        # ever touching live behavior. Default (8, not a smaller guess like 3) is itself
        # empirically chosen - replaying real data showed 3 still let one out-of-order confirmation
        # through (label 16 confirmed from a 3-cycle noise streak while labels 13/14/15, earlier in
        # the sequence, weren't confirmed yet); 8 was the smallest tested value that produced a
        # fully clean, in-order sweep on both sides against that same real run.
        self.guidance_confirm_cycles = guidance_confirm_cycles
        self._guidance_streaks = {}       # label -> consecutive cycles at/above threshold so far
        self._guidance_confirmed = set()  # label -> permanently confirmed once the streak requirement is met
        self._guidance_confirmed_pos = {}  # label -> last known centroid while confirmed (kept fresh
                                            # when the label has enough faces THIS cycle, else reused)
        _debug(f"LiveSegmenter init: guidance_confirm_cycles={self.guidance_confirm_cycles}")

        # Prediction-debugging methodology (per the live investigation request): every
        # snapshot_every'th cycle, process_once() dumps (mesh, per-face predicted labels,
        # area_mm2, dilation gate states) to snapshot_dir as a .npz - see
        # realtime/visualize_snapshots.py for the offline tool that turns a run's worth
        # of these into inspectable renders, so "how do predictions evolve as the accumulated
        # patch grows" can be looked at directly instead of only watched live in the GUI.
        self.snapshot_dir = snapshot_dir
        self.snapshot_every = snapshot_every
        if self.snapshot_dir is not None:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            _debug(f"LiveSegmenter init: snapshotting every {snapshot_every} cycle(s) to {self.snapshot_dir}")

        # Finding #9 (REALTIME.md): pyfqmr's simplify_mesh() holds the GIL for its full duration
        # (verified directly in pyfqmr/Simplify.pyx - only its small helper functions are marked
        # nogil), so running decimation in-process/in-thread can freeze every other thread in the
        # program - including the network reader - for as long as one call takes. 'spawn' (not the
        # Linux default 'fork') is used deliberately: this constructor runs after the model has
        # already been moved to CUDA in main(), and forking a process with an initialized CUDA
        # context is documented as unsafe; 'spawn' always starts workers as fresh interpreters
        # regardless of what's already initialized in this process. The warm-up pass below pays
        # each worker's one-time import cost (numpy/trimesh/pyfqmr/torch, a few seconds total)
        # right now, before the scan starts, instead of as a surprise latency spike on cycle 1.
        self._pool_workers = pool_workers
        # Finding #9 follow-up (REALTIME.md): the pool parallelizes decimation ACROSS chunks, but
        # one single call to pyfqmr's simplify_mesh() can't itself be split across cores - a chunk
        # large enough on its own to dominate a cycle's wall time (the periodic large re-batched
        # chunks, 100k+ raw triangles) still runs pinned to one core for its full duration even
        # with the pool in place. split_threshold_faces makes _fold_new_chunks spatially split any
        # chunk above this raw face count (spatial_split, live_preprocessing.py) into up to
        # pool_workers pieces BEFORE submitting, so even one huge chunk benefits from parallelism.
        self.split_threshold_faces = 40000
        self._pool = ProcessPoolExecutor(max_workers=pool_workers,
                                          mp_context=multiprocessing.get_context("spawn"))
        _debug(f"LiveSegmenter init: warming up {pool_workers} decimation worker process(es)...")
        t0 = time.time()
        for f in [self._pool.submit(int, 1) for _ in range(pool_workers)]:
            f.result()
        _debug(f"LiveSegmenter init: worker pool ready in {time.time() - t0:.1f}s")

    def update_chunk(self, mesh_id, vertices, triangles, colors=None):
        """colors: optional (V, 3) uint8 real per-vertex RGB from the scanner (mesh_wire.
        parse_mesh_payload's "colors"), matching vertices - only meaningful/kept when
        self.use_color is set (a color-trained checkpoint is loaded); ignored otherwise so a
        geometry-only run pays no extra cost for color data it'll never use."""
        n = len(triangles) if triangles is not None else 0
        with self._lock:
            self.chunks[mesh_id] = {"vertices": vertices, "triangles": triangles,
                                     "colors": colors if self.use_color else None}
            prev = self._raw_faces_seen.get(mesh_id, 0)
            self._raw_faces_seen[mesh_id] = n
            self._raw_faces_total += (n - prev)
        _debug(f"update_chunk: mesh_id={mesh_id} vertices={len(vertices)} triangles={n}")

    def _fold_new_chunks(self, chunks_snapshot):
        """Decimates + voxel-dedups any mesh_id that's new or changed since it was last folded
        in, then permanently merges its newly-kept faces into self._accumulated exactly once (see
        class docstring, Finding #8) and evicts it from self.chunks - its contribution is now
        baked into self._accumulated and self._claimed_voxels for good, and real hardware
        confirmed mesh_ids are never updated again after they first arrive, so there's nothing
        left worth keeping it for. A chunk that DOES reappear with a changed signature (not
        observed live, but handled correctly if the real server ever does this) is simply
        reprocessed - claim_new_voxels naturally keeps only whatever's genuinely new about it.

        This cycle's newly-decimated+deduped pieces are merged together ONCE (merge_meshes, cheap
        - bounded by what's new THIS cycle only) before the single _fold_in call onto the
        accumulator, rather than one _fold_in call per chunk - _fold_in's own cost is O(current
        accumulator size), so calling it once per new chunk instead of once per cycle would
        reintroduce an O(chunks_folded_this_cycle x accumulator_size) cost on cycles that catch up
        on many chunks at once (e.g. right after a slow cycle).

        Decimation itself (the pyfqmr call inside decimate_chunk_worker) runs in self._pool
        (Finding #9, REALTIME.md: pyfqmr holds the GIL for the whole call, so doing this in-thread
        can freeze the reader thread for as long as one call takes). Every new/changed chunk this
        cycle is submitted to the pool AT ONCE - both so several chunks decimate in true parallel
        across CPU cores, and so the reader thread is never blocked by decimation regardless of
        how many chunks or how large. A single chunk larger than split_threshold_faces is first
        spatially split (spatial_split) into multiple pieces, each submitted as its own future -
        the pool parallelizes ACROSS chunks, but one pyfqmr call can't itself be split across
        cores, so without this a single huge periodic re-batched chunk would still pin one cycle's
        wall time to however long that one call takes, unparallelized (see REALTIME.md's Finding
        #9 follow-up). claim_new_voxels still runs sequentially here in the main process
        afterward, since it mutates the shared self._claimed_voxels set and first-writer-wins
        semantics need a defined order."""
        t_decimate, t_dedup = 0.0, 0.0
        n_folded, n_skipped = 0, 0
        to_evict = []
        new_meshes = []

        futures = {}
        mesh_meta = {}  # mesh_id -> (signature, raw_face_count, n_pieces)
        pieces_pending = {}  # mesh_id -> list of decimated (vertices, faces) pieces collected so far
        raw_centroids = {}  # mesh_id -> centroid of its own raw vertices, for motion-guidance (below)
        for mesh_id, chunk in chunks_snapshot.items():
            v, t = chunk["vertices"], chunk["triangles"]
            c = chunk.get("colors") if self.use_color else None
            if v is None or t is None or len(v) == 0 or len(t) == 0:
                continue
            signature = (len(v), len(t))
            if self._folded_signature.get(mesh_id) == signature:
                n_skipped += 1
                continue
            raw_centroids[mesh_id] = v.mean(axis=0)
            if len(t) > self.split_threshold_faces:
                n_splits = min(self._pool_workers, math.ceil(len(t) / self.split_threshold_faces))
                pieces = spatial_split(v, t, n_splits, colors=c)
            else:
                pieces = [(v, t, c)]
            mesh_meta[mesh_id] = (signature, len(t), len(pieces))
            pieces_pending[mesh_id] = []
            for pv, pf, pc in pieces:
                futures[self._pool.submit(decimate_chunk_worker, pv, pf, self.target_density, pc)] = mesh_id

        t0 = time.time()
        for future in as_completed(futures):
            mesh_id = futures[future]
            down_vertices, down_faces, down_colors, t_worker = future.result()
            t_decimate += t_worker
            pieces_pending[mesh_id].append((down_vertices, down_faces, down_colors, t_worker))
            if len(pieces_pending[mesh_id]) < mesh_meta[mesh_id][2]:
                continue  # still waiting on this mesh_id's other spatial-split pieces

            signature, raw_face_count, n_pieces = mesh_meta[mesh_id]
            piece_times = [pt for _, _, _, pt in pieces_pending[mesh_id]]

            def _piece_mesh(dv, df, dc):
                m = trimesh.Trimesh(vertices=dv, faces=df, process=False)
                if dc is not None:
                    m.visual.vertex_colors = dc
                return m

            if n_pieces == 1:
                dv, df, dc, _ = pieces_pending[mesh_id][0]
                down_chunk = _piece_mesh(dv, df, dc)
            else:
                down_chunk = merge_meshes([_piece_mesh(dv, df, dc) for dv, df, dc, _ in pieces_pending[mesh_id]])
            slowest_piece = max(piece_times)
            slow_tag = " [SLOW]" if slowest_piece > 0.3 else ""
            split_note = f" ({n_pieces} spatial pieces, slowest={slowest_piece:.3f}s)" if n_pieces > 1 else ""
            _debug(f"decimate_chunk_worker: mesh_id={mesh_id} {raw_face_count} -> {len(down_chunk.faces)} "
                   f"faces (area={down_chunk.area:.1f}mm2) in {sum(piece_times):.3f}s{slow_tag}{split_note}")

            t1 = time.time()
            keep_mask = claim_new_voxels(self._claimed_voxels, down_chunk, self.voxel_size)
            t_dedup += time.time() - t1
            has_colors = down_chunk.visual.kind == 'vertex'
            kept = trimesh.Trimesh(vertices=down_chunk.vertices, faces=down_chunk.faces[keep_mask], process=False)
            if has_colors:
                kept.visual.vertex_colors = down_chunk.visual.vertex_colors
            if len(kept.faces) > 0:
                new_meshes.append(kept)

            self._decimated_faces_total += len(down_chunk.faces)
            self._raw_area_total_mm2 += float(down_chunk.area)
            self._folded_signature[mesh_id] = signature
            n_folded += 1
            to_evict.append(mesh_id)
            _debug(f"_fold_new_chunks: mesh_id={mesh_id} CHANGED (sig={signature}) - decimated to "
                   f"{len(down_chunk.faces)} faces, kept {int(keep_mask.sum())}/{len(keep_mask)} "
                   f"after voxel dedup ({len(self._claimed_voxels)} voxels claimed scan-wide)")
        t_wall_decimate = time.time() - t0

        if to_evict:
            # mesh_ids are monotonically increasing = temporally ordered (Finding #1, REALTIME.md)
            # - the highest one folded this cycle is the freshest real-world position data
            # available, used as a "where is the scanner right now" proxy for motion guidance
            # (_compute_guidance) since the wire protocol carries no actual scanner pose.
            self._last_chunk_centroid = raw_centroids[max(to_evict)]
            with self._lock:
                for mesh_id in to_evict:
                    self.chunks.pop(mesh_id, None)

        t0 = time.time()
        if new_meshes:
            self._fold_in(merge_meshes(new_meshes))
        t_merge = time.time() - t0

        _debug(f"_fold_new_chunks: {n_folded} chunk(s) folded in, {n_skipped} already up to date, "
               f"accumulated={0 if self._accumulated is None else len(self._accumulated.faces)} "
               f"faces scan-wide ({len(self.chunks)} chunk(s) still resident) - decimate wall="
               f"{t_wall_decimate:.3f}s ({t_decimate:.3f}s worker-seconds across {n_folded} chunk(s))")
        return n_folded, t_wall_decimate, t_dedup, t_merge

    def _fold_in(self, new_mesh):
        """Permanently appends new_mesh's faces onto self._accumulated (concatenate-and-offset,
        same approach as live_preprocessing.merge_meshes but incremental). If that pushes the
        accumulator past retention_ceiling, cheaply random-subsamples it back down (cap_face_count
        - same cheap op as the final model-input cap, NOT decimation - see that function's
        docstring for why a real decimation call here would reintroduce the same "redo full
        expensive work every cycle" cost this whole fold-based design exists to avoid)."""
        if new_mesh is None or len(new_mesh.faces) == 0:
            return
        if self._accumulated is None:
            self._accumulated = new_mesh
        else:
            # any_colors (not all/and) - see live_preprocessing.merge_meshes's docstring for the
            # 2026-08-19 bug this mirrors the fix for: an all-or-nothing check here meant a single
            # color-less fold would silently wipe out every previously-accumulated real color,
            # permanently, since self._accumulated is fully rebuilt below, not incrementally
            # patched. A mesh lacking color contributes NO_COLOR_SENTINEL for its own vertices
            # instead.
            any_colors = self._accumulated.visual.kind == 'vertex' or new_mesh.visual.kind == 'vertex'
            offset = len(self._accumulated.vertices)
            vertices = np.concatenate([self._accumulated.vertices, new_mesh.vertices], axis=0)
            faces = np.concatenate([self._accumulated.faces, new_mesh.faces + offset], axis=0)
            colors = None
            if any_colors:
                acc_colors = (np.asarray(self._accumulated.visual.vertex_colors)[:, :3]
                              if self._accumulated.visual.kind == 'vertex'
                              else np.tile(NO_COLOR_SENTINEL, (len(self._accumulated.vertices), 1)))
                new_colors = (np.asarray(new_mesh.visual.vertex_colors)[:, :3]
                              if new_mesh.visual.kind == 'vertex'
                              else np.tile(NO_COLOR_SENTINEL, (len(new_mesh.vertices), 1)))
                colors = np.concatenate([acc_colors, new_colors], axis=0)
            self._accumulated = trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                                                 vertex_colors=colors)
        if len(self._accumulated.faces) > self.retention_ceiling:
            self._accumulated = cap_face_count(self._accumulated, self.retention_ceiling)

    def process_once(self):
        """Returns (down_mesh, face_colors_uint8, stats_dict), or None if there's nothing to
        process yet (scan hasn't started / no chunks received).

        Fully instrumented per REALTIME.md's live-investigation request: every stage that was
        previously bundled into one t_downsample/t_collate/t_infer number is now timed
        individually (decimate/dedup/merge/cap split out; collate broken into kdtree vs. each
        FPS pass via precompute_neighbor_indices(profile=True); inference broken into each
        network block, in particular isolating block2/block3's on-the-fly feature-space cdist -
        the model's own O(N^2) op that can't be precomputed - from everything else, via
        model.model(profile=True)). Every cycle is tagged with an incrementing number so its
        pipeline/collate/model timing lines can be correlated by eye or by grep."""
        with self._lock:
            chunks_snapshot = dict(self.chunks)

        self._cycle += 1
        cycle = self._cycle
        t_cycle_start = time.time()

        n_folded, t_decimate_chunks, t_dedup_chunks, t_merge = self._fold_new_chunks(chunks_snapshot)

        if self._accumulated is None or len(self._accumulated.faces) == 0:
            return None
        raw_faces = self._raw_faces_total
        decimated_faces = self._decimated_faces_total
        raw_area = self._raw_area_total_mm2
        dedup_faces = len(self._accumulated.faces)
        down = self._accumulated

        t0 = time.time()
        if dedup_faces > self.max_model_faces:
            # voxel dedup alone doesn't bound total accumulated size over a long scan (see
            # __init__'s comment and Finding #8, REALTIME.md) - this is the actual guard against
            # feeding the model (and its O(N^2) on-the-fly cdist in blocks 2/3) an unbounded point
            # count. cap_face_count is a cheap random subsample, NOT another decimation pass - see
            # its docstring for why a real decimation call here reintroduced the exact "redo full
            # work every cycle" cost Finding #3 already eliminated once, this time at the cap step.
            down = cap_face_count(down, self.max_model_faces)
            _debug(f"process_once[cycle={cycle}]: accumulated mesh {dedup_faces} faces exceeds "
                   f"max_model_faces={self.max_model_faces} - capped to {len(down.faces)} faces")
        t_cap = time.time() - t0
        t_downsample = t_decimate_chunks + t_dedup_chunks + t_merge + t_cap  # kept for stats-dict compat

        _debug(f"process_once[cycle={cycle}]: {decimated_faces} faces after per-chunk decimation -> "
               f"{dedup_faces} after voxel dedup -> {len(down.faces)} after cap "
               f"({decimated_faces - dedup_faces} redundant faces removed)")

        t1 = time.time()
        real_face_colors = None  # (F, 3) uint8 - what actually got fed to the model this cycle,
        # for snapshotting (see _save_snapshot) - None when not a color-trained checkpoint.
        # Named distinctly from `colors` below (colors_for_predictions' render colors for the
        # predicted-class overlay) - unrelated to this, don't conflate the two.
        if self.use_color:
            pos, x, area_mm2, real_face_colors = mesh_to_model_inputs_with_color(
                down, self.transform, flat_colors=self._flat_colors)
        else:
            pos, x, area_mm2 = mesh_to_model_inputs(down, self.transform)
        t_feature = time.time() - t1

        t2 = time.time()
        rng = np.random.default_rng()
        local_idx, dilated_idx, collate_timings = precompute_neighbor_indices(
            pos.numpy(), self.collator.k, self.collator.dilation_ks, rng, profile=True)
        t_collate = time.time() - t2

        t3 = time.time()
        pos_b = pos.unsqueeze(0).to(self.device)
        x_b = x.unsqueeze(0).to(self.device)
        local_idx_b = torch.from_numpy(local_idx).unsqueeze(0).to(self.device)
        dilated_idx_b = tuple(torch.from_numpy(d).unsqueeze(0).to(self.device) for d in dilated_idx)
        with torch.no_grad():
            pred = self.model.model(x_b, pos_b, area_mm2=area_mm2, local_idx=local_idx_b,
                                     dilated_idx=dilated_idx_b, profile=True)
            pred_labels = pred.argmax(dim=-1)[0].cpu().numpy()
        t_infer = time.time() - t3
        model_timings = dict(getattr(self.model.model, "last_timings", {}) or {})

        colors = colors_for_predictions(pred_labels, self.num_classes)
        label_positions = self._label_positions(down, pred_labels)
        guidance = self._compute_guidance(down, pred_labels)
        t_total = time.time() - t_cycle_start

        if self.snapshot_dir is not None and cycle % self.snapshot_every == 0:
            self._save_snapshot(cycle, down, pred_labels, area_mm2, raw_faces, guidance, real_face_colors)

        counts = np.bincount(pred_labels, minlength=self.num_classes)
        hist = ", ".join(f"{c}:{n}" for c, n in enumerate(counts) if n > 0)

        _debug(f"process_once[cycle={cycle}] TIMING: pipeline decimate_chunks={t_decimate_chunks:.3f}s "
               f"dedup_chunks={t_dedup_chunks:.3f}s merge={t_merge:.3f}s cap={t_cap:.3f}s "
               f"feature_extract={t_feature:.3f}s")
        collate_str = " ".join(f"{k}={v:.3f}s" for k, v in collate_timings.items())
        _debug(f"process_once[cycle={cycle}] TIMING: collate total={t_collate:.3f}s ({collate_str})")
        model_str = " ".join(f"{k}={v:.3f}s" for k, v in model_timings.items())
        _debug(f"process_once[cycle={cycle}] TIMING: model_forward total={t_infer:.3f}s ({model_str})")
        _debug(f"process_once[cycle={cycle}]: raw={raw_faces} down={len(down.faces)} faces "
               f"area={area_mm2:.1f}mm2 | downsample={t_downsample:.3f}s feature={t_feature:.3f}s "
               f"collate={t_collate:.3f}s infer={t_infer:.3f}s total={t_total:.3f}s")
        _debug(f"process_once[cycle={cycle}]: prediction histogram (class:face_count) = {{{hist}}}")
        if guidance is not None:
            if guidance.get("done"):
                _debug(f"process_once[cycle={cycle}]: guidance - full arch regimen complete")
            else:
                _debug(f"process_once[cycle={cycle}]: guidance - target=label{guidance['target_label']}"
                       f"{'/#' + str(guidance['target_universal']) if guidance['target_universal'] else ''} "
                       f"compass={guidance['compass']} distance={guidance['distance_mm']:.1f}mm")

        stats = {
            "cycle": cycle, "raw_faces": raw_faces, "raw_area_mm2": raw_area,
            "decimated_faces": decimated_faces, "dedup_faces": dedup_faces, "down_faces": len(down.faces),
            "area_mm2": area_mm2, "t_downsample": t_downsample, "t_decimate_chunks": t_decimate_chunks,
            "t_dedup_chunks": t_dedup_chunks, "t_merge": t_merge, "t_cap": t_cap, "t_feature": t_feature,
            "t_collate": t_collate, "collate_timings": collate_timings, "t_infer": t_infer,
            "model_timings": model_timings, "t_total": t_total, "guidance": guidance,
        }
        return down, colors, label_positions, stats

    def _save_snapshot(self, cycle, down, pred_labels, area_mm2, raw_faces, guidance, face_colors=None):
        """One .npz per snapshotted cycle - vertices/faces (float32/int32, keeps file size down),
        per-face predicted labels, and enough metadata (area_mm2, gate states, raw_faces,
        guidance) to correlate a prediction against how much real-world context the model had at
        that point AND where the guidance arrow was pointing at that same moment - the point of
        logging guidance alongside the scan/prediction is to be able to investigate the arrow's
        own targeting strategy after the fact (is a bad target caused by a still-unstable nearby
        prediction, or by the extrapolation itself), not just eyeball it live. Failures here are
        logged, not raised - a snapshot write should never take down a live scan.

        guidance_state is one of "none" (nothing to point at yet), "done" (regimen complete), or
        "active" (target_label/universal/distance_mm/compass/current_pos/target_pos all valid) -
        the other guidance_* fields are NaN/-1/"" filled when not applicable for that state.

        face_colors: (F, 3) uint8 real per-face RGB actually fed to the model this cycle (see
        mesh_to_model_inputs_with_color) - None for a geometry-only checkpoint (self.use_color is
        False), saved as has_face_colors=False + an empty (0,3) array in that case so downstream
        loaders (realtime/visualize_snapshots.py) always get a consistently-shaped field
        rather than needing to branch on a missing key. The point of saving this at all: added
        2026-08-19 after a real hardware run with the color-trained checkpoint scored noticeably
        worse than the geometry-only one - the leading hypothesis is a training/inference color
        distribution mismatch (training used flat, zero-variance synthetic color per class-group -
        dataset/patch_color_augmentation.py's SyntheticColorPaint draws exactly ONE RGB value
        shared by every tooth face and one shared by every gum face in a patch - real scanner
        color naturally varies continuously across the mesh, a signal the model never saw). Saving
        the actual fed-in color per cycle lets that be checked directly against the training
        distribution (dataset/patch_color_augmentation.py's TOOTH_RGB_LOW/HIGH, GUM_RGB_LOW/HIGH)
        instead of only being inferable indirectly from degraded predictions."""
        gates = getattr(self.model.model, "last_gates", (None, None, None))
        if guidance is None:
            guidance_state = "none"
        elif guidance.get("done"):
            guidance_state = "done"
        else:
            guidance_state = "active"
        g = guidance if guidance_state == "active" else {}
        path = os.path.join(self.snapshot_dir, f"cycle_{cycle:04d}.npz")
        try:
            np.savez_compressed(
                path,
                vertices=down.vertices.astype(np.float32),
                faces=down.faces.astype(np.int32),
                pred_labels=pred_labels.astype(np.int32),
                area_mm2=np.float32(area_mm2),
                raw_faces=np.int64(raw_faces),
                cycle=np.int32(cycle),
                gates=np.array([bool(gate) for gate in gates], dtype=bool),
                timestamp=np.float64(time.time()),
                guidance_state=np.str_(guidance_state),
                guidance_target_label=np.int32(g.get("target_label", -1)),
                guidance_target_universal=np.int32(g.get("target_universal") or -1),
                guidance_distance_mm=np.float32(g.get("distance_mm", np.nan)),
                guidance_compass=np.str_(g.get("compass", "")),
                guidance_current_pos=np.asarray(g.get("current_pos", [np.nan] * 3), dtype=np.float32),
                guidance_target_pos=np.asarray(g.get("target_pos", [np.nan] * 3), dtype=np.float32),
                # Finding #13 (REALTIME.md) - the sustained-confirmation set at THIS cycle, so a
                # future investigation can directly see which labels were confirmed and when,
                # rather than needing to reverse-engineer it from pred_labels + replay_guidance.py.
                guidance_confirmed_labels=np.array(sorted(self._guidance_confirmed), dtype=np.int32),
                has_face_colors=np.bool_(face_colors is not None),
                face_colors=(face_colors.astype(np.uint8) if face_colors is not None
                             else np.zeros((0, 3), dtype=np.uint8)),
            )
        except Exception as e:  # noqa: BLE001
            _debug(f"_save_snapshot: failed to write {path}: {e}")

    def _compute_guidance(self, down, pred_labels):
        """Motion-guidance for the fixed scanning regimen (see _GUIDANCE_LEFT_SEQUENCE/
        _GUIDANCE_RIGHT_SEQUENCE above): given the CURRENT set of confidently-identified labels
        (>= self.guidance_min_faces predicted faces), the next target tooth is fully
        deterministic - no coverage-gap or PCA-based frontier heuristics needed, unlike a
        free-form scanning pattern would require (only 17-class label semantics apply here; the
        regimen is meaningless for the 5-class coarse scheme).

        The target's real-world position isn't known until it's actually scanned, so it's
        estimated by extrapolating from the 1-2 nearest ALREADY-identified teeth in the same
        sweep direction: with 2 neighbors, continue their local spacing/direction one more step;
        with just 1, point toward that landmark directly (no spacing estimate possible yet).
        "Current position" is self._last_chunk_centroid (the most recently folded chunk's own
        centroid - the freshest real-world position data available, since the wire protocol
        carries no actual scanner pose - see _fold_new_chunks). The 3D guidance vector is
        projected onto a (3,2) PCA basis fit ONCE (first cycle with enough vertices) and reused
        for the rest of the scan, so the arrow's frame stays fixed instead of drifting cycle to
        cycle as the accumulated mesh's own principal axes shift.

        Returns None if nothing can be estimated yet (no chunks folded yet, or the target is 8/9
        and neither has been found), or a dict with target_label/universal/distance_mm/compass/
        angle_deg, or {"done": True} once the whole regimen (both sweeps) is complete.

        Finding #11 fix (REALTIME.md): the left->right sweep transition is gated on the
        SCANNER'S OWN recent position, not just on prediction confidence. Real hardware showed
        labels 9/10 getting confidently identified "for free" in the background - wide model
        context or overlapping coverage - well before the scanner physically finished the left
        side, but never surfacing as a target since the left sweep always takes priority. The
        instant the left sweep's last tooth crossed guidance_min_faces, the target jumped
        straight to whatever the right sweep had already silently progressed to (one real case:
        8mm away -> 47.7mm away, opposite side of the arch, no transition) - not just a big
        jump, but skipping the "return to center" step the user's actual regimen includes as a
        real physical movement. Now: once every left-sweep tooth is confidently identified, the
        transition to the right sweep only happens once self._last_chunk_centroid is actually
        back within guidance_center_return_threshold_mm of the 8/9 midline position - until then,
        target stays label 9 (the right sweep's own start / the regimen's literal "come back to
        8/9" instruction), pointing at its already-known position rather than extrapolating."""
        if self.num_classes != 17 or self._last_chunk_centroid is None:
            return None

        counts = np.bincount(pred_labels, minlength=self.num_classes)
        # Finding #13 (REALTIME.md): "confidently identified" now requires SUSTAINED evidence
        # (guidance_confirm_cycles consecutive cycles), not a single cycle's threshold crossing -
        # see update_guidance_confirmation's own docstring for why. centroids only ever contains
        # confirmed labels; its position is refreshed whenever that label has enough faces THIS
        # cycle, and otherwise reuses the last known position (a confirmed label can still have a
        # noisy near-zero count on an individual cycle without losing its guidance target).
        update_guidance_confirmation(counts, self._guidance_streaks, self._guidance_confirmed,
                                      self.guidance_min_faces, self.guidance_confirm_cycles)
        centroids = {}
        for label in self._guidance_confirmed:
            if label < len(counts) and counts[label] >= self.guidance_min_faces:
                self._guidance_confirmed_pos[label] = down.triangles_center[pred_labels == label].mean(axis=0)
            if label in self._guidance_confirmed_pos:
                centroids[label] = self._guidance_confirmed_pos[label]

        def first_missing(seq):
            for lbl in seq:
                if lbl not in centroids:
                    return lbl
            return None

        seq = _GUIDANCE_LEFT_SEQUENCE
        target = first_missing(seq)
        target_pos = None

        if target is None:
            center_anchors = [centroids[lbl] for lbl in (8, 9) if lbl in centroids]
            center_pos = np.mean(center_anchors, axis=0) if center_anchors else None
            if (center_pos is not None
                    and np.linalg.norm(self._last_chunk_centroid - center_pos)
                    > self.guidance_center_return_threshold_mm):
                target, target_pos = 9, center_pos
            else:
                seq = _GUIDANCE_RIGHT_SEQUENCE
                target = first_missing(seq)

        if target is None:
            return {"done": True}

        if target_pos is None:
            idx = seq.index(target)
            anchors = [lbl for lbl in seq[:idx] if lbl in centroids]
            if len(anchors) >= 2:
                p_prev, p_last = centroids[anchors[-2]], centroids[anchors[-1]]
                target_pos = p_last + (p_last - p_prev)
            elif len(anchors) == 1:
                target_pos = centroids[anchors[-1]]
            else:
                return None  # target is 8 or 9 and neither found yet - nothing to extrapolate from

        if self._guidance_projection is None:
            if len(down.vertices) < 10:
                return None
            centered = down.vertices - down.vertices.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            self._guidance_projection = vt[:2].T

        vec3 = target_pos - self._last_chunk_centroid
        vec2 = vec3 @ self._guidance_projection
        distance_mm = float(np.linalg.norm(vec3))
        angle_deg = float(np.degrees(np.arctan2(vec2[1], vec2[0])))
        compass = _COMPASS[int(((angle_deg + 22.5) % 360) // 45)]
        universal = int(label_to_universal_number(np.array([target]), self.arch)[0]) if self.arch else None

        return {"done": False, "target_label": int(target), "target_universal": universal,
                "distance_mm": distance_mm, "compass": compass, "angle_deg": angle_deg,
                "current_pos": np.array(self._last_chunk_centroid, dtype=np.float64),
                "target_pos": np.array(target_pos, dtype=np.float64)}

    def _label_positions(self, down, pred_labels):
        """For each non-gum class actually predicted this cycle, the centroid of its faces and
        the text to float there: the real Universal tooth number for the 17-class scheme (needs
        self.arch - see utils/teeth_numbering.label_to_universal_number), or the coarse class
        name for the 5-class scheme (no per-tooth granularity to number there)."""
        positions = []
        present = [c for c in np.unique(pred_labels) if c != 0]
        if self.num_classes == 17 and self.arch is None:
            _debug("_label_positions: no --arch given, skipping Universal-number labels "
                   "(class colors/legend still shown)")
            return positions
        for c in present:
            face_mask = pred_labels == c
            centroid = down.triangles_center[face_mask].mean(axis=0).astype(np.float32)
            if self.num_classes == 17:
                universal = int(label_to_universal_number(np.array([c]), self.arch)[0])
                text = f"#{universal}"
            else:
                text = str(_coarse_class_names.get(int(c), c))
            positions.append((centroid, text))
        return positions


class SegmentedViewerApp:
    """Open3D GUI window - structurally the same as mesh_viewer_linux.py's MeshViewerApp (Start/
    Stop buttons, auto-fit camera), but displays LiveSegmenter's flat-shaded predicted-class mesh
    alongside the raw scan instead of only raw scanner colors. A background worker thread runs
    LiveSegmenter.process_once() in a skip-if-busy loop (never blocks the GUI thread, never falls
    behind by queueing stale work - only the LATEST accumulated state is ever processed).

    Split view, two independent SceneWidgets side by side (not overlaid layers in one viewport,
    which is what an earlier version of this did - z-fighting between the two different
    triangulations of the same surface made it hard to actually compare them):
    - LEFT pane (scene_widget, push_raw/_apply_raw): every incoming mesh frame is displayed
      immediately, as its own Open3D geometry per mesh_id with the scanner's own vertex colors -
      no decimation, no dedup, no model, nothing that could ever lag. This is exactly
      mesh_viewer_linux.py's MeshViewerApp._apply() pattern, reimplemented here (not imported -
      that class's push()/_apply() are bound to ITS OWN scene_widget) so this viewer doesn't need
      mesh_viewer_linux.py running side by side just to see the scan accumulate in real time. The
      motion-guidance arrow (LiveSegmenter._compute_guidance, guidance_arrow_mesh) is drawn here
      too, directly on the live mesh - this is the pane meant to be actively watched while moving
      the scanner.
    - RIGHT pane (pred_scene_widget, _apply): LiveSegmenter's output for the SAME voxels, however
      far behind it currently is - replaced wholesale each cycle (not merged with the raw view),
      so it always shows exactly what the model currently believes about the covered surface.
    Both panes occupy the same real-world (mm) coordinate space (down's vertices are never
    touched by PatchPreTransform's normalization - only the pos/x fed to the model are) and each
    gets its own independent auto-fit camera, since they now show different content rather than
    being togglable layers of the same view."""

    def __init__(self, send_command, segmenter, num_classes=17, arch=None,
                 title="Live segmentation (predicted classes)", width=1280, height=860):
        self.send_command = send_command
        self.segmenter = segmenter
        self.num_classes = num_classes
        self.arch = arch
        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering

        app = self.gui.Application.instance
        app.initialize()
        self.window = app.create_window(title, width, height)

        self.scene_widget = self.gui.SceneWidget()  # LEFT: live raw scan + guidance arrow
        self.scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.9, 0.9, 0.9, 1.0])
        self.window.add_child(self.scene_widget)

        self.pred_scene_widget = self.gui.SceneWidget()  # RIGHT: predicted-class mesh
        self.pred_scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.pred_scene_widget.scene.set_background([0.9, 0.9, 0.9, 1.0])
        self.window.add_child(self.pred_scene_widget)

        em = self.window.theme.font_size
        self.panel = self.gui.Vert(0.3 * em, self.gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.start_btn = self.gui.Button("Start Scan")
        self.start_btn.set_on_clicked(lambda: self.send_command("start_scan"))
        self.stop_btn = self.gui.Button("Stop Scan")
        self.stop_btn.set_on_clicked(lambda: self.send_command("stop_scan"))
        self.autofit = self.gui.Checkbox("Auto-fit view")
        self.autofit.checked = True
        self.show_raw = self.gui.Checkbox("Show live scan (raw colors)")
        self.show_raw.checked = True
        self.show_raw.set_on_checked(self._on_toggle_raw)
        self.show_3d_labels = self.gui.Checkbox("Show tooth-number labels")
        self.show_3d_labels.checked = True
        self.status = self.gui.Label("connecting…")
        # Motion guidance (LiveSegmenter._compute_guidance) - the text readout is a secondary
        # reference; the actual arrow is drawn on the live mesh itself (left pane) since that's
        # what's meant to be glanced at while actively moving the scanner.
        self.guidance_label = self.gui.Label("guidance: waiting for scan data...")
        self.guidance_label.text_color = self.gui.Color(0.85, 0.1, 0.1, 1.0)
        self.stats_label = self.gui.Label("")
        for w in (self.start_btn, self.stop_btn, self.autofit, self.show_raw,
                  self.show_3d_labels, self.status, self.guidance_label, self.stats_label):
            self.panel.add_child(w)
        self.panel.add_child(self.gui.Label(""))  # spacer
        self._build_legend()
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self.material = self.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.raw_material = self.rendering.MaterialRecord()
        self.raw_material.shader = "defaultUnlit"
        self.raw_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.arrow_material = self.rendering.MaterialRecord()
        self.arrow_material.shader = "defaultUnlit"
        self.arrow_material.base_color = [1.0, 1.0, 1.0, 1.0]

        # Independent auto-fit state per pane (raw/left, pred/right) - see _fit_camera.
        self._raw_fit_bounds = None
        self._raw_camera_ready = False
        self._pred_fit_bounds = None
        self._pred_camera_ready = False
        self._label_handles = []  # active Label3D handles on pred_scene_widget - cleared/rebuilt every _apply()
        self._region_bounds = {}  # raw layer only now ("raw_<mesh_id>" -> (min, max))
        self._pred_region_bounds = {}  # predicted mesh only ("segmented" -> (min, max))
        self._raw_names = set()  # which _region_bounds keys are raw layer geometries
        _b0 = o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        self.scene_widget.setup_camera(60.0, _b0, _b0.get_center())
        self.pred_scene_widget.setup_camera(60.0, _b0, _b0.get_center())

        self._stop_worker = threading.Event()
        self._dirty = threading.Event()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _build_legend(self):
        """Color-coded legend in the side panel: which color means which tooth. text_color on
        each gui.Label is set to that class's own color, matching what's painted on the mesh -
        this is the 'explain it with a color map' half of what was asked for; the floating 3D
        labels in _apply() are the 'number next to the prediction' half. No leading bullet glyph
        (there used to be a "■" here) - Open3D's GUI font doesn't cover that Unicode codepoint and
        rendered it as "?"; the whole line's text_color already conveys the swatch, so the bullet
        was purely decorative and safe to drop rather than hunt for a font-supported substitute."""
        self.panel.add_child(self.gui.Label("Legend:"))
        if self.num_classes == 5:
            for c in range(5):
                r, g, b = _coarse_class_color[c]
                lbl = self.gui.Label(_coarse_class_names[c])
                lbl.text_color = self.gui.Color(r / 255, g / 255, b / 255, 1.0)
                self.panel.add_child(lbl)
            return
        gum_lbl = self.gui.Label("gum")
        gr, gg, gb = GUM_COLOR_255.tolist()
        gum_lbl.text_color = self.gui.Color(gr / 255, gg / 255, gb / 255, 1.0)
        self.panel.add_child(gum_lbl)
        for label in range(1, 17):
            r, g, b = _teeth_color[label]
            name = _LABEL_NAMES.get(label, f"label {label}")
            if self.arch is not None:
                universal = int(label_to_universal_number(np.array([label]), self.arch)[0])
                text = f"#{universal} {name}"
            else:
                text = name
            lbl = self.gui.Label(text)
            lbl.text_color = self.gui.Color(r / 255, g / 255, b / 255, 1.0)
            self.panel.add_child(lbl)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        pref = self.panel.calc_preferred_size(ctx, self.gui.Widget.Constraints())
        em = self.window.theme.font_size
        panel_width = max(pref.width, int(18 * em))
        self.panel.frame = self.gui.Rect(r.x + int(0.5 * em), r.y + int(0.5 * em),
                                         panel_width, min(pref.height, r.height - int(em)))
        # split whatever's left of the panel evenly between the two scene panes (left=raw,
        # right=predicted - see class docstring)
        remaining_x = r.x + panel_width + int(em)
        remaining_width = max(r.width - panel_width - int(em), 0)
        half_width = remaining_width // 2
        self.scene_widget.frame = self.gui.Rect(remaining_x, r.y, half_width, r.height)
        self.pred_scene_widget.frame = self.gui.Rect(remaining_x + half_width, r.y,
                                                       remaining_width - half_width, r.height)

    def set_status(self, text):
        self.gui.Application.instance.post_to_main_thread(self.window, lambda: setattr(self.status, "text", text))

    def _on_toggle_raw(self, checked):
        scene = self.scene_widget.scene
        for name in self._raw_names:
            try:
                scene.show_geometry(name, checked)
            except Exception as e:  # noqa: BLE001
                print(f"[!] show_geometry failed for {name}: {e}", file=sys.stderr)

    def push_raw(self, parsed):
        """Thread-safe entry point for the reader thread - every mesh frame goes straight to the
        GUI, completely bypassing LiveSegmenter/the worker thread. This is the 'accumulation stays
        fast and independent' half of the live view; see the class docstring."""
        self.gui.Application.instance.post_to_main_thread(self.window, lambda: self._apply_raw(parsed))

    def _apply_raw(self, parsed):
        name = f"raw_{parsed['mesh_id']}"
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(parsed["vertices"].astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(
            parsed["triangles"].astype(np.int32) if parsed["triangles"] is not None
            else np.empty((0, 3), dtype=np.int32))
        if parsed["normals"] is not None:
            m.vertex_normals = o3d.utility.Vector3dVector(parsed["normals"].astype(np.float64))
        else:
            m.compute_vertex_normals()
        if parsed["colors"] is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(parsed["colors"].astype(np.float64))

        scene = self.scene_widget.scene
        try:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
            scene.add_geometry(name, m, self.raw_material)
            scene.show_geometry(name, self.show_raw.checked)
        except Exception as e:  # noqa: BLE001
            print(f"[!] raw add_geometry failed: {e}", file=sys.stderr)
            return

        self._raw_names.add(name)
        v = parsed["vertices"]
        if v.size:
            self._region_bounds[name] = (v.min(axis=0), v.max(axis=0))
        self._fit_camera("raw", force=False)

    def notify_new_data(self):
        """Call whenever a chunk update arrives - wakes the worker if it's idle. If the worker is
        still busy on a previous cycle, this just marks 'there's newer data', which it'll pick up
        on its next pass - no queueing, no falling behind."""
        self._dirty.set()

    def _worker_loop(self):
        _debug("worker thread started")
        while not self._stop_worker.is_set():
            if not self._dirty.wait(timeout=1.0):
                continue
            self._dirty.clear()
            _debug("worker: new data flagged, starting a processing cycle")
            try:
                result = self.segmenter.process_once()
            except Exception as e:  # noqa: BLE001
                print(f"[!] inference cycle failed: {e}", file=sys.stderr)
                continue
            if result is not None:
                mesh, colors, label_positions, stats = result
                self.gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda m=mesh, c=colors, lp=label_positions, s=stats: self._apply(m, c, lp, s))
            else:
                _debug("worker: process_once() returned None (no chunks yet)")

    def _apply(self, mesh, colors, label_positions, stats):
        t0 = time.time()
        vertices, triangles, vcolors = flat_shaded_mesh(mesh, colors)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(vertices)
        m.triangles = o3d.utility.Vector3iVector(triangles)
        m.vertex_colors = o3d.utility.Vector3dVector(vcolors)

        pred_scene = self.pred_scene_widget.scene
        try:
            if pred_scene.has_geometry("segmented"):
                pred_scene.remove_geometry("segmented")
            pred_scene.add_geometry("segmented", m, self.material)
        except Exception as e:  # noqa: BLE001
            print(f"[!] add_geometry failed: {e}", file=sys.stderr)
            return

        for handle in self._label_handles:
            self.pred_scene_widget.remove_3d_label(handle)
        self._label_handles = []
        if self.show_3d_labels.checked:
            for centroid, text in label_positions:
                self._label_handles.append(self.pred_scene_widget.add_3d_label(centroid, text))

        guidance = stats.get("guidance")
        raw_scene = self.scene_widget.scene
        if guidance is None:
            self.guidance_label.text = "guidance: waiting for scan data..."
            if raw_scene.has_geometry("guidance_arrow"):
                raw_scene.remove_geometry("guidance_arrow")
        elif guidance.get("done"):
            self.guidance_label.text = "guidance: full arch regimen complete"
            if raw_scene.has_geometry("guidance_arrow"):
                raw_scene.remove_geometry("guidance_arrow")
        else:
            target_text = f"#{guidance['target_universal']}" if guidance["target_universal"] else \
                f"label {guidance['target_label']}"
            self.guidance_label.text = (f"guidance: {guidance['compass']}  move toward tooth "
                                         f"{target_text}  (~{guidance['distance_mm']:.0f}mm)")
            # Float the arrow above the occlusal surface rather than drawing it AT surface level -
            # direct feedback after a real run: it often ended up embedded in/under the mesh,
            # hard to see. Offsets both endpoints along the mesh's own average face normal (not a
            # fixed world axis - robust to whatever orientation this coordinate frame happens to
            # be in) by a fixed 4mm, comfortably clear of the surface without floating away from it.
            lift = np.zeros(3)
            if len(mesh.face_normals):
                avg_normal = mesh.face_normals.mean(axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 1e-6:
                    lift = avg_normal / norm * 4.0
            arrow = guidance_arrow_mesh(guidance["current_pos"] + lift, guidance["target_pos"] + lift)
            if arrow is not None:
                try:
                    if raw_scene.has_geometry("guidance_arrow"):
                        raw_scene.remove_geometry("guidance_arrow")
                    raw_scene.add_geometry("guidance_arrow", arrow, self.arrow_material)
                except Exception as e:  # noqa: BLE001
                    print(f"[!] guidance arrow add_geometry failed: {e}", file=sys.stderr)

        capped_note = f", capped from {stats['dedup_faces']}" if stats['dedup_faces'] != stats['down_faces'] else ""
        self.stats_label.text = (
            f"{stats['down_faces']} faces (raw {stats['raw_faces']}, {stats['decimated_faces']} "
            f"after per-chunk decimation{capped_note})  area={stats['area_mm2']:.0f}mm2\n"
            f"downsample={stats['t_downsample']:.2f}s collate={stats['t_collate']:.2f}s "
            f"infer={stats['t_infer']:.2f}s total={stats['t_total']:.2f}s")

        if vertices.size:
            self._pred_region_bounds["segmented"] = (vertices.min(axis=0), vertices.max(axis=0))
        self._fit_camera("pred", force=False)

        _debug(f"_apply: {len(triangles)} triangles ({len(vertices)} vertices), "
               f"{len(label_positions)} 3D label(s), GUI update took {time.time() - t0:.3f}s")

    def _fit_camera(self, which, force):
        """which: 'raw' or 'pred' - each pane now shows different content (split view, not
        overlaid layers - see class docstring), so each gets its own independent camera fit based
        on only its own geometry's bounds, tracked via the running min/max over every region ever
        seen in that pane, same incremental approach as mesh_viewer_linux.py's
        MeshViewerApp._fit_camera."""
        scene_widget = self.scene_widget if which == "raw" else self.pred_scene_widget
        region_bounds = self._region_bounds if which == "raw" else self._pred_region_bounds
        fit_bounds_attr = f"_{which}_fit_bounds"
        camera_ready_attr = f"_{which}_camera_ready"

        if not region_bounds:
            return
        if not force and (not self.autofit.checked) and getattr(self, camera_ready_attr):
            return
        mins = np.min([b[0] for b in region_bounds.values()], axis=0)
        maxs = np.max([b[1] for b in region_bounds.values()], axis=0)
        fit_bounds = getattr(self, fit_bounds_attr)
        if not force and fit_bounds is not None:
            fmin, fmax = fit_bounds
            if not (np.any(mins < fmin) or np.any(maxs > fmax)):
                return
        center = (mins + maxs) / 2.0
        half = np.maximum((maxs - mins) / 2.0, 1e-3)
        margin = 1.7
        emin, emax = center - half * margin, center + half * margin
        setattr(self, fit_bounds_attr, (emin, emax))
        bbox = o3d.geometry.AxisAlignedBoundingBox(emin, emax)
        try:
            scene_widget.setup_camera(60.0, bbox, bbox.get_center())
            setattr(self, camera_ready_attr, True)
        except Exception as e:  # noqa: BLE001
            print(f"[!] camera fit failed ({which}): {e}", file=sys.stderr)

    def run(self):
        self.gui.Application.instance.run()
        self._stop_worker.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.1.30",
                     help="IP address of the Windows mesh server (mesh_server_win.py) [default: 192.168.1.30] - "
                          "or 127.0.0.1 with --port 8780 if going through mesh_relay.py instead")
    ap.add_argument("--port", type=int, default=mesh_wire.DEFAULT_PORT,
                     help=f"server TCP port (default {mesh_wire.DEFAULT_PORT}; use 8780 if going through mesh_relay.py)")
    ap.add_argument("--retry", type=float, default=2.0)
    ap.add_argument("--ckpt", required=True, help="Path to a trained .ckpt")
    ap.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    ap.add_argument("--arch", choices=["upper", "lower"], default=None,
                     help="Which arch is being scanned - needed to show real Universal tooth "
                          "numbers (1-32) on the 17-class scheme; omit to show tooth type names "
                          "only (e.g. 'central_incisor') without a number, in the legend and "
                          "floating 3D labels alike. Not used for the 5-class scheme.")
    ap.add_argument("--target_density", type=float, default=5.0, help="faces/mm^2 - must match training density")
    ap.add_argument("--max_faces", type=int, default=20000,
                     help="hard cap on the merged mesh fed to the model per cycle, independent of voxel dedup "
                          "(default 20000, comfortably above a full arch's ~16-18k faces at target_density=5.0) - "
                          "protects against models/patch_layer.py's EdgeGraphConvBlock 2/3 always doing an "
                          "on-the-fly O(N^2) torch.cdist with no cap of its own; confirmed live testing let the "
                          "merged mesh grow past 160k faces and try to allocate 100+ GiB (see REALTIME.md)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--guidance_confirm_cycles", type=int, default=8,
                     help="Finding #13 (REALTIME.md): a tooth needs this many CONSECUTIVE cycles "
                          "at >= --guidance_min_faces before guidance treats it as confidently "
                          "found - filters brief misclassification spikes (e.g. a terminal molar "
                          "momentarily flagged while the scanner is still elsewhere) that used to "
                          "make the arrow flicker or the regimen report done prematurely. Default "
                          "(8) was empirically chosen via replay_guidance.py against real "
                          "hardware snapshots - lower it for a more responsive but noisier arrow, "
                          "raise it for a slower but more conservative one.")
    ap.add_argument("--snapshot_every", type=int, default=1,
                     help="save a (mesh, per-face predicted labels, area_mm2, dilation gate states) .npz "
                          "every N cycles to realtime/logs/snapshots/<run_id>/ - see "
                          "realtime/visualize_snapshots.py to turn a run's worth of these into "
                          "inspectable renders (default: every cycle)")
    ap.add_argument("--no_snapshots", action="store_true", help="disable snapshot saving entirely")
    ap.add_argument("--color_mode", choices=["varying", "flat"], default="varying",
                     help="Only meaningful for a color-trained checkpoint (feature_dim=27) - ignored "
                          "otherwise. 'varying' (default): feed the model real captured scanner color "
                          "as-is. 'flat': classify each face's real color as tooth-like or gum-like "
                          "(live_preprocessing.py's classify_and_flatten_colors) and repaint with ONE "
                          "flat value per class for the whole scan, matching training's own flat-per-"
                          "class-group color structure - an interim mitigation for a confirmed "
                          "training/inference color distribution mismatch (see Docs/REALTIME.md), not "
                          "a fix in itself.")
    args = ap.parse_args()

    log_path = setup_logging()
    print(f"[*] debug log: {log_path}")
    snapshot_dir = None
    if not args.no_snapshots:
        run_id = os.path.splitext(os.path.basename(log_path))[0]
        snapshot_dir = os.path.join(os.path.dirname(log_path), "snapshots", run_id)
        print(f"[*] prediction snapshots: {snapshot_dir} (every {args.snapshot_every} cycle(s))")

    if not _have_gui():
        print("[!] this Open3D build lacks the GUI framework -- pip install --upgrade open3d", file=sys.stderr)
        return

    if args.num_classes == 17 and args.arch is None:
        print("[*] no --arch given - tooth labels will show type names only, no Universal numbers "
              "(pass --arch upper or --arch lower to enable them)")

    print(f"[*] loading checkpoint {args.ckpt} on {args.device}...")
    model = PatchLitDilatedToothSegmentationNetwork.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval()
    model.to(args.device)
    segmenter = LiveSegmenter(model, args.device, num_classes=args.num_classes,
                               target_density=args.target_density, arch=args.arch,
                               max_model_faces=args.max_faces, snapshot_dir=snapshot_dir,
                               snapshot_every=args.snapshot_every, color_mode=args.color_mode,
                               guidance_confirm_cycles=args.guidance_confirm_cycles)

    conn = Conn(args.host, args.port, args.retry)
    viewer = SegmentedViewerApp(lambda cmd: conn.send({"type": "cmd", "cmd": cmd}, b""), segmenter,
                                 num_classes=args.num_classes, arch=args.arch)
    conn.on_state = viewer.set_status

    stop = threading.Event()
    frame_count = {"mesh": 0, "scan_control": 0, "status": 0, "other": 0}

    # Hypothesis #4 investigation (REALTIME.md): does heavy CPU work in the worker thread
    # (pyfqmr decimation, the model's on-the-fly cdist, etc.) starve this reader thread of the
    # GIL long enough to explain the "connection lost, will reconnect" cycling seen in earlier
    # real-hardware logs? _last_frame_recv_end tracks wall-clock time between successive frames
    # actually being processed here; a widening gap that lines up with a TIMING line elsewhere in
    # the log (same wall-clock window) is the empirical signal to look for - not by itself proof
    # (the server could just be pacing itself slower at that moment), but if gaps consistently
    # track OUR local CPU-heavy stages rather than anything server-side, that's GIL contention.
    _last_frame_recv_end = [time.time()]
    GAP_WARN_S = 0.2

    def reader():
        _debug("reader thread started")
        while not stop.is_set():
            t_recv_start = time.time()
            try:
                frame = conn.recv()
            except OSError:
                frame = None
            t_recv_end = time.time()
            recv_blocked_s = t_recv_end - t_recv_start
            if frame is None:
                if stop.is_set():
                    return
                _debug("connection lost, will reconnect")
                viewer.set_status("relay closed -- reconnecting…")
                conn.connect()
                continue
            gap_since_last_frame_s = t_recv_end - _last_frame_recv_end[0]
            _last_frame_recv_end[0] = t_recv_end
            gap_tag = " [GAP - possible GIL/scheduling stall, see reader() comment]" if gap_since_last_frame_s > GAP_WARN_S else ""
            header, blob = frame
            t = header.get("type")
            frame_count[t if t in frame_count else "other"] += 1
            if t == "mesh":
                try:
                    parsed = mesh_wire.parse_mesh_payload(header, blob)
                    _debug(f"recv mesh frame: mesh_id={parsed['mesh_id']} "
                           f"vertex_count={parsed['vertex_count']} face_count={parsed['face_count']} "
                           f"blob_bytes={len(blob)} seq={parsed.get('seq')} "
                           f"recv_blocked={recv_blocked_s:.3f}s gap_since_last_frame={gap_since_last_frame_s:.3f}s"
                           f"{gap_tag} (totals so far: {frame_count})")
                    viewer.push_raw(parsed)
                    # mesh_wire.parse_mesh_payload's colors are float64 in [0,1] (raw 3xu8/255 -
                    # see that file's own parsing); the color pipeline (live_preprocessing.py,
                    # dataset/patch_preprocessing_color.py) works in uint8 [0,255] throughout,
                    # matching training's synthetic-color convention, so convert once here at the
                    # entry point rather than at every downstream stage.
                    raw_colors = parsed.get("colors")
                    colors_u8 = (raw_colors * 255.0).round().astype(np.uint8) if raw_colors is not None else None
                    segmenter.update_chunk(parsed["mesh_id"], parsed["vertices"], parsed["triangles"],
                                            colors_u8)
                    viewer.notify_new_data()
                except Exception as e:  # noqa: BLE001
                    print(f"[!] parse mesh failed: {e}", file=sys.stderr)
            elif t == "scan_control":
                _debug(f"recv scan_control: {header}")
                if header.get("ok"):
                    viewer.set_status(f"scan {header.get('action')}")
                else:
                    viewer.set_status(f"scan control failed: {header.get('error')}")
            else:
                _debug(f"recv frame type={t}: {header}")

    threading.Thread(target=reader, daemon=True).start()
    print(f"[*] segmentation viewer up. Click 'Start Scan' to begin. (relay {args.host}:{args.port})")
    try:
        viewer.run()
    finally:
        stop.set()
        with conn._lock:
            s = conn.sock
        if s is not None:
            try:
                s.close()
            except Exception:
                pass
        segmenter._pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
