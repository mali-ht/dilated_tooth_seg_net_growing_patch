import time

import numpy as np
import pyfqmr
import torch
import trimesh
from scipy.spatial import cKDTree

from dataset.patch_color_augmentation import GUM_RGB_HIGH, GUM_RGB_LOW, TOOTH_RGB_HIGH, TOOTH_RGB_LOW
from dataset.patch_preprocessing import PatchPreTransform
from dataset.patch_preprocessing_real_color import PatchPreTransformWithRealColor
from debug_log import logger


def _debug(msg):
    logger.debug(f"[live_preprocessing] {msg}")


# Explicit "no real color for this vertex" sentinel - matches trimesh's own default placeholder
# (trimesh.visual.color.DEFAULT_COLOR[:3]) so it's visually consistent with what already showed up
# whenever color was unset, but used INTENTIONALLY here rather than leaking in by accident. Real
# hardware confirmed (2026-08-19) mesh_wire's wire protocol genuinely omits color on some frames
# (mesh_wire.parse_mesh_payload's colors=None is a normal, anticipated path, not an error) - a
# scan where ~75% of the accumulated mesh silently fell back to this exact value turned out to be
# a real bug (see merge_meshes/​_fold_in below), not the color-domain-gap finding this constant's
# neighbors are otherwise about.
NO_COLOR_SENTINEL = np.array([102, 102, 102], dtype=np.uint8)

# Bridges the live mesh_wire stream (realtime/mesh_wire.py, untouched) to the trained
# patch model. Two problems solved here, neither handled anywhere else in the pipeline:
#
# 1. Density mismatch: claude_code_prompt.md measured the live scanner's native density at
#    ~74 faces/mm^2, but the model was trained at target_density=5 faces/mm^2 (processed_w5,
#    dataset/mesh_dataset.py). Feeding raw scanner density straight to the model would be far
#    outside the training distribution. downsample_to_density mirrors mesh_dataset.py's
#    _donwscale_mesh (same pyfqmr quadric-decimation call, same target_count formula) but drops
#    the label-remapping step - there is no ground truth for live scanner data, so there is
#    nothing to remap - and skips the zero-face padding that method uses to hit an exact count
#    (that padding exists to keep the OLD fixed-N model happy; Component 4's model handles
#    variable N natively, so padding here would just inject meaningless degenerate faces into a
#    real inference mesh for no benefit).
#
# 2. Chunk merging: mesh_viewer_linux.py's "accumulating mesh" is actually N independently
#    updated named sub-meshes (one Open3D geometry per mesh_id, replaced whole on every update -
#    see that file's _apply()). Open3D's scene graph can render that directly with no merging
#    needed; our model needs ONE connected mesh with a single shared vertex/face indexing scheme.
#    merge_chunks does the standard concatenate-and-offset merge.
#
# 3. Cross-chunk overlap: confirmed against the real scanner (not assumed) that mesh_ids are
#    strictly increasing and never reused - every chunk is a small, NEW, spatially-overlapping
#    observation as the scanner sweeps (re-observing already-covered surface is normal scanning
#    motion), not a disjoint region that gets refined in place. Per-chunk decimation alone
#    (downsample_to_density above) correctly bounds each chunk's OWN density, but does nothing
#    about redundancy BETWEEN chunks - merging hundreds of overlapping chunks made the total grow
#    with elapsed scan time, not true surface area (measured live: 240,000+ faces after a couple
#    minutes, next to the ~16,000-18,000 a full arch at 5 faces/mm^2 should actually need),
#    which is what caused the CUDA OOM crashes (32GB, then 145GB, then 217GB allocation attempts)
#    and very likely the connection drops too (a multi-second CPU-side stall processing 200k+
#    points plausibly starved the reader thread long enough to look dead). claim_new_voxels fixes
#    this: a face only survives if it's the first thing to claim its voxel in a persistent, global
#    grid - total kept faces are bounded by true covered area, not by how many times it's been
#    re-observed.


def downsample_to_density(mesh: trimesh.Trimesh, target_density: float = 5.0,
                           vertex_colors: np.ndarray = None) -> trimesh.Trimesh:
    """Quadric-decimate mesh down to ~target_density faces/mm^2, matching processed_w5's training
    density exactly (same formula, same decimator settings as mesh_dataset.py._donwscale_mesh).
    Returns mesh unchanged if it's already at or below that density (nothing to remove).

    vertex_colors: optional (V, 3) uint8 RGB, matching mesh.vertices - real scanner color (see
    mesh_wire.parse_mesh_payload's "colors"). pyfqmr's simplifier only knows about raw vertex
    positions/faces - it has no concept of carrying an auxiliary per-vertex attribute through
    edge-collapse decimation, and the decimated output's vertices are genuinely NEW positions, not
    a subset of the input's. So when vertex_colors is given, this does a nearest-neighbor color
    transfer (cKDTree query of each decimated vertex against the ORIGINAL, pre-decimation vertex
    set) and attaches the result as the returned mesh's mesh.visual.vertex_colors - the standard
    technique for carrying per-vertex attributes through a decimator that doesn't support them
    natively. Attached via trimesh's own vertex_colors mechanism (not a separate parallel array)
    so it rides through later trimesh operations (concatenation, vertex merging) via trimesh's own
    bookkeeping instead of needing to be manually re-synced at every step."""
    target_count = max(round(mesh.area * target_density), 4)
    if target_count >= len(mesh.faces):
        _debug(f"downsample_to_density: {len(mesh.faces)} faces already <= target {target_count} "
               f"(area={mesh.area:.1f}mm2) - no-op")
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors
        return mesh

    t0 = time.time()
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=3, preserve_border=True,
                                   verbose=0, max_iterations=2000)
    new_positions, new_faces, _ = mesh_simplifier.getMesh()
    mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_faces)
    t_decimate = time.time() - t0
    # SLOW tag: makes the "periodic large re-batched chunk" hypothesis (REALTIME.md) grep-able -
    # 0.3s is well above the typical ~0.03-0.1s per-chunk cost seen in real-hardware logs, but
    # comfortably below the ~0.6-0.7s measured for the occasional 90-150k-triangle chunks
    slow_tag = " [SLOW]" if t_decimate > 0.3 else ""
    _debug(f"downsample_to_density: {len(mesh.faces)} -> {len(mesh_simple.faces)} faces "
           f"(target {target_count}, area={mesh.area:.1f}mm2) in {t_decimate:.3f}s{slow_tag}")

    if vertex_colors is not None:
        t0 = time.time()
        nearest_idx = cKDTree(mesh.vertices).query(mesh_simple.vertices, k=1)[1]
        mesh_simple.visual.vertex_colors = vertex_colors[nearest_idx]
        _debug(f"downsample_to_density: nearest-neighbor color transfer for {len(mesh_simple.vertices)} "
               f"decimated vertices in {time.time() - t0:.3f}s")

    if len(mesh_simple.faces) > target_count:
        # plain uniform random face subsample - NOT mesh_dataset.py's trimesh.sample.
        # sample_surface_even, which does rejection sampling for evenly-spaced surface coverage.
        # That's fine for offline, one-time training preprocessing but measured badly live: it
        # prints "only got X/Y samples!" and can stall for a long time as the input mesh grows,
        # which is exactly what was freezing the GUI. pyfqmr's decimation output is already close
        # to evenly distributed, so trimming the small remaining overshoot doesn't need rejection
        # sampling's stronger (and much more expensive) guarantee - a uniform random pick over
        # face indices is O(1)-ish and doesn't degrade with input size.
        rng = np.random.default_rng()
        keep = rng.choice(len(mesh_simple.faces), size=target_count, replace=False)
        kept_colors = mesh_simple.visual.vertex_colors if vertex_colors is not None else None
        mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[keep])
        if kept_colors is not None:
            mesh_simple.visual.vertex_colors = kept_colors  # same vertex buffer, only faces changed
        _debug(f"downsample_to_density: trimmed decimation overshoot {len(keep)} -> {target_count} "
               f"via uniform random subsample")

    return mesh_simple


def spatial_split(vertices: np.ndarray, faces: np.ndarray, n_splits: int, colors: np.ndarray = None):
    """Splits one raw chunk's faces into n_splits spatially-coherent groups (PCA-axis ordering of
    face centroids, then even chunks along that axis) - lets a single very large incoming chunk's
    decimation spread across n_splits pool workers instead of running as one pyfqmr call pinned to
    a single core (a single simplify_mesh() call can't itself be parallelized - REALTIME.md
    Finding #9 - so a chunk large enough on its own to dominate a cycle's wall time needs to be
    split before submission, not just submitted alongside others).

    Spatially-coherent splits, not a random face subset: this project's own earlier testing found
    a random subset fragments badly under pyfqmr's decimator (see downsample_to_density's
    docstring and realtime/test_live_segmenter.py's _spatial_chunks, which this mirrors).

    colors: optional (V, 3) uint8 RGB matching vertices - sliced by the same used_v index as
    vertices so each piece's colors line up with its own local vertex indexing.

    Returns a list of (piece_vertices, piece_faces, piece_colors) triples, each with its own local
    vertex indexing - piece_colors is None throughout when colors isn't given. May return fewer
    than n_splits pieces if the chunk is too small to split evenly."""
    if n_splits <= 1 or len(faces) <= 1:
        return [(vertices, faces, colors)]
    centroids = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).triangles_center
    centered = centroids - centroids.mean(axis=0)
    axis_pos = centered @ np.linalg.svd(centered, full_matrices=False)[2][0]
    order = np.argsort(axis_pos)
    pieces = []
    for face_idx in np.array_split(order, n_splits):
        if len(face_idx) == 0:
            continue
        sub_faces = faces[face_idx]
        used_v = np.unique(sub_faces)
        remap = -np.ones(len(vertices), dtype=np.int64)
        remap[used_v] = np.arange(len(used_v))
        piece_colors = colors[used_v] if colors is not None else None
        pieces.append((vertices[used_v], remap[sub_faces], piece_colors))
    return pieces


def decimate_chunk_worker(vertices: np.ndarray, faces: np.ndarray, target_density: float,
                           colors: np.ndarray = None):
    """Runs downsample_to_density in an isolated worker process (see LiveSegmenter's
    ProcessPoolExecutor, mesh_viewer_segmented.py) - pyfqmr's simplify_mesh() holds Python's GIL
    for its entire duration (verified by reading pyfqmr/Simplify.pyx directly: only its small
    vertex/face-copying helpers are marked nogil, the actual decimation call is not - REALTIME.md
    Finding #9), so running it in-process can freeze every other thread in the program, including
    the network reader, for as long as one call takes - up to 1+ seconds for the periodic large
    re-batched chunks seen on real hardware. A separate OS process has its own GIL, so no matter
    how long simplify_mesh() takes here, it cannot block anything in the main process.

    Takes/returns plain numpy arrays rather than a trimesh.Trimesh - keeps the pickled payload
    sent across the process boundary minimal and avoids depending on trimesh's pickling behavior
    for a type with cached lazy properties. Deliberately does no logging here (the child process's
    `logger` has no FileHandler attached - see debug_log.py) - the caller already has the
    mesh_id/cycle context to log a proper line after collecting this result, so it does that
    itself instead of coordinating log writes across process boundaries.

    colors: optional (V, 3) uint8 RGB matching vertices - see downsample_to_density's own
    docstring for how it survives decimation (nearest-neighbor transfer, since decimation
    produces genuinely new vertex positions). Returns down_colors as a 4th element (None if colors
    wasn't given) - (V_down, 3) uint8, matching down_vertices."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    t0 = time.time()
    down = downsample_to_density(mesh, target_density, vertex_colors=colors)
    down_colors = np.asarray(down.visual.vertex_colors)[:, :3] if colors is not None else None
    return down.vertices, down.faces, down_colors, time.time() - t0


def cap_face_count(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Hard cap on face count via uniform random subsample - deliberately NOT another
    downsample_to_density (quadric decimation) call. A first version of the live-viewer cap did
    call downsample_to_density on the merged mesh, and it reproduced the exact same "redo full
    expensive work every cycle" problem Finding #3 already fixed once: measured live, once the
    merged mesh passes the cap it does so EVERY cycle from then on, and re-decimating an
    ever-growing merged mesh (54,031 -> 20,000, then 68,196 -> 20,000, ...) from scratch each time
    cost 1.7s, then 2.1s and climbing - the dominant share of the multi-second cycle times that
    made the live view stop feeling real-time. The merged mesh reaching this point is already
    fairly evenly covered (every chunk was quadric-decimated to training density before merging,
    and voxel dedup only removes exact-duplicate coverage, not concentrated regions), so a plain
    random subsample loses density evenly across the whole mesh rather than needing decimation's
    shape-aware simplification - same reasoning as downsample_to_density's own overshoot-trim
    above, just applied to the final cap instead of a single chunk.

    Only ever subsets FACES, never touches the vertex buffer - so if mesh already carries
    mesh.visual.vertex_colors (real scanner color, see downsample_to_density), that gets copied
    onto the returned mesh too (a fresh trimesh.Trimesh() constructor call doesn't inherit the old
    object's .visual by itself)."""
    if len(mesh.faces) <= max_faces:
        return mesh
    t0 = time.time()
    rng = np.random.default_rng()
    keep = rng.choice(len(mesh.faces), size=max_faces, replace=False)
    has_colors = mesh.visual.kind == 'vertex'
    capped = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)
    if has_colors:
        capped.visual.vertex_colors = mesh.visual.vertex_colors
    _debug(f"cap_face_count: {len(mesh.faces)} -> {len(capped.faces)} faces via uniform random "
           f"subsample in {time.time() - t0:.3f}s")
    return capped


def voxel_size_for_density(target_density: float, fineness: float = 0.3) -> float:
    """Cell size for claim_new_voxels. A first version used voxel_size = 1/sqrt(target_density)
    directly (the spacing a single chunk decimated TO target_density would naturally have between
    its own distinct faces) - measured wrong: it made adjacent, genuinely-distinct faces within
    ONE chunk alias into the same voxel and get discarded as if they were redundant, losing ~30%
    of a single, non-overlapping chunk's faces for no reason. `fineness` scales the grid down well
    below that natural spacing (default 0.3x) so real neighbors stay distinguishable, while still
    being coarse enough to catch TRUE redundant re-observations (different chunks landing on
    close to the same physical point) - see realtime/test_live_segmenter.py for both
    properties verified together (single-chunk density preserved AND synthetic overlap removed)."""
    return fineness / (target_density ** 0.5)


def claim_new_voxels(claimed_voxels: set, mesh: trimesh.Trimesh, voxel_size: float) -> np.ndarray:
    """Returns a boolean mask over mesh.faces: True for faces that successfully claim a
    previously-unclaimed voxel (added to claimed_voxels as a side effect - a persistent set the
    caller keeps across the whole scan), False for faces whose voxel was already claimed (by this
    or an earlier chunk) and are therefore redundant with something already kept. First-writer-
    wins and permanent by design: once a voxel is claimed, it's never displaced, so already-shown
    surface doesn't flicker/change as later, redundant re-observations of it arrive."""
    centroids = mesh.triangles_center
    keys = np.floor(centroids / voxel_size).astype(np.int64)
    keep = np.zeros(len(mesh.faces), dtype=bool)
    for i, key in enumerate(map(tuple, keys)):
        if key not in claimed_voxels:
            claimed_voxels.add(key)
            keep[i] = True
    return keep


def merge_meshes(meshes: list) -> trimesh.Trimesh | None:
    """Concatenate already-independent trimesh.Trimesh objects into one, offsetting each one's
    face indices by the cumulative vertex count before it - same merge as merge_chunks, just
    operating on trimesh objects (used by the per-chunk decimation path: each chunk is decimated
    to target density INDEPENDENTLY and cached, then the already-small decimated pieces are
    merged here, rather than merging huge raw chunks and decimating the ever-growing whole every
    cycle - see mesh_viewer_segmented.py's LiveSegmenter for why that mattered).

    If ANY input mesh carries mesh.visual.vertex_colors (real scanner color - see
    downsample_to_density), colors get concatenated the same way and passed into the Trimesh
    constructor's own vertex_colors= kwarg - verified empirically that process=True's vertex
    dedup handles a colors array passed this way correctly (merged/deduped vertices keep their
    original color, not just first-N-truncated). A mesh that lacks colors contributes
    NO_COLOR_SENTINEL for its own vertices rather than silently discarding every OTHER mesh's real
    color too - fixed 2026-08-19 after a real hardware run showed the accumulated mesh's
    "captured color" was up to ~75% trimesh's own default placeholder: the wire protocol genuinely
    omits color on SOME frames (mesh_wire.parse_mesh_payload's colors=None is expected, not an
    error), and the old all-or-nothing check meant every fold that included even one color-less
    chunk wiped out all previously-accumulated real color, permanently, for the rest of the scan
    (self._accumulated gets fully REBUILT each fold in mesh_viewer_segmented.py's _fold_in, not
    incrementally patched) - only a genuinely all-none merge now produces None."""
    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        _debug("merge_meshes: nothing to merge yet")
        return None
    t0 = time.time()
    all_vertices, all_faces = [], []
    any_colors = any(m.visual.kind == 'vertex' for m in meshes)
    all_colors = [] if any_colors else None
    offset = 0
    for m in meshes:
        all_vertices.append(m.vertices)
        all_faces.append(m.faces + offset)
        if any_colors:
            if m.visual.kind == 'vertex':
                all_colors.append(np.asarray(m.visual.vertex_colors)[:, :3])
            else:
                all_colors.append(np.tile(NO_COLOR_SENTINEL, (len(m.vertices), 1)))
        offset += len(m.vertices)
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors = np.concatenate(all_colors, axis=0) if all_colors is not None else None
    merged = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, vertex_colors=colors)
    _debug(f"merge_meshes: {len(meshes)} chunk(s), {len(faces)} faces before cleanup -> "
           f"{len(merged.faces)} after (process=True vertex/degenerate cleanup) in {time.time()-t0:.3f}s")
    return merged


def merge_chunks(chunks: dict) -> trimesh.Trimesh | None:
    """chunks: {mesh_id: {"vertices": (V,3) float array, "triangles": (F,3) int array}}, each
    chunk's triangles indexed locally into its OWN vertices (that's what arrives over the wire -
    see mesh_wire.parse_mesh_payload). Concatenates every chunk's vertex buffer into one shared
    buffer, offsetting each chunk's face indices by the cumulative vertex count before it.
    process=True (trimesh's default - unlike dataset/patch_generator.py's load_cached_arch, which
    uses process=False to keep face indices aligned with a separate cached labels array) is used
    deliberately here: live scanner chunks can share near-duplicate boundary vertices where two
    reconstruction blocks meet, and there's no external label array whose alignment would break
    from trimesh's cleanup - letting it merge duplicate vertices only helps here.
    Returns None if there's nothing to merge yet (scan just started, no chunks received)."""
    all_vertices, all_faces = [], []
    offset = 0
    for mesh_id in sorted(chunks.keys()):
        v, f = chunks[mesh_id]["vertices"], chunks[mesh_id]["triangles"]
        if v is None or f is None or len(v) == 0 or len(f) == 0:
            continue
        all_vertices.append(np.asarray(v, dtype=np.float64))
        all_faces.append(np.asarray(f, dtype=np.int64) + offset)
        offset += len(v)
    if not all_vertices:
        return None
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def classify_and_flatten_colors(face_colors: np.ndarray, face_centroids: np.ndarray, tooth_color: np.ndarray,
                                 gum_color: np.ndarray, redness_threshold: float = 20.0) -> np.ndarray:
    """--color_mode flat (mesh_viewer_segmented.py): real captured color varies continuously
    across the mesh, but training's SyntheticColorPaint (dataset/patch_color_augmentation.py)
    only ever produced ONE flat RGB value shared by every tooth face and one shared by every gum
    face - confirmed empirically (2026-08-19) that this mismatch alone costs real accuracy (a
    controlled test: identical geometry/labels, only adding +-25/channel noise on top of otherwise
    flat training-matching color dropped mIoU by 0.085). This is the "for now" mitigation the user
    asked for instead of waiting on a retrain: classify each face as tooth-like or gum-like from
    its OWN real captured color, then repaint EVERY face in each group with the SAME ONE flat
    value (tooth_color/gum_color, expected to be drawn once per scan from
    dataset/patch_color_augmentation.py's TOOTH_RGB_LOW/HIGH and GUM_RGB_LOW/HIGH - see
    mesh_viewer_segmented.py's LiveSegmenter for where those get drawn and cached for the whole
    scan) - reproducing training's exact flat-per-class-group structure from real data instead of
    feeding the model real data's natural variance.

    Classifier: "redness" = R - mean(G, B). Calibrated against 2 real half-arch scans (2026-08-19,
    NOT a learned classifier, a documented heuristic) - genuine (non-sentinel) captured color's
    redness distribution has its main mass below ~15-20 (teeth: fairly neutral/slightly cool) with
    a secondary tail above ~20 (gum: visibly redder) - see Docs/REALTIME.md for the full
    distribution.

    face_centroids: (F, 3) mesh.triangles_center, same order as face_colors - needed for faces
    with NO real data (both source vertices at NO_COLOR_SENTINEL). analyze_snapshot_colors.py
    found real scans genuinely have this for 53-72% of faces (the wire protocol just doesn't send
    color on most frames - not a bug). Fixed 2026-08-19: these used to default to the tooth
    bucket unconditionally regardless of what they actually were, which given how much of a scan
    this affects likely explains why flat mode helped but stayed below the geometry-only
    baseline - any color-less GUM region would get silently painted with the flat TOOTH color,
    actively misleading the model in exactly the areas the color channel should help most. Now
    propagated from each color-less face's NEAREST face that DOES have real color (nearest by
    centroid distance, via cKDTree - same nearest-neighbor technique downsample_to_density uses to
    carry color through decimation) - a color-less patch of gum surrounded by real gum data should
    now correctly inherit "gum", not "tooth" by default. Only if NO face in the whole mesh has any
    real color at all (nothing to propagate from) does everything fall back to the tooth bucket,
    as a last resort.

    This is a coarse, unvalidated-against-ground-truth heuristic, not a trained classifier - some
    misclassification is expected. The point is only to get the model's color-channel INPUT
    distribution back in range, not to perfectly separate tooth from gum by color alone."""
    face_colors = face_colors.astype(np.float64)
    is_sentinel = np.all(face_colors == NO_COLOR_SENTINEL, axis=1)
    redness = face_colors[:, 0] - face_colors[:, 1:3].mean(axis=1)
    is_gum = redness > redness_threshold

    if is_sentinel.any():
        if is_sentinel.all():
            is_gum[:] = False  # nothing anywhere to propagate from - last-resort default
        else:
            real_idx = np.where(~is_sentinel)[0]
            nearest = cKDTree(face_centroids[real_idx]).query(face_centroids[is_sentinel], k=1)[1]
            is_gum[is_sentinel] = is_gum[real_idx[nearest]]

    out = np.empty_like(face_colors, dtype=np.uint8)
    out[~is_gum] = tooth_color
    out[is_gum] = gum_color
    return out


def mesh_to_model_inputs(mesh: trimesh.Trimesh, transform: PatchPreTransform):
    """Raw trimesh -> (pos, x, area_mm2) ready for the model, via the SAME PatchPreTransform
    (dataset/patch_preprocessing.py) used at training time - same 24-dim feature layout, same
    fixed-scale normalization. There's no ground truth at inference time, so a dummy all-zero
    labels array is passed through the transform (unused for computing pos/x) and discarded."""
    faces = mesh.faces
    mesh_faces = torch.from_numpy(faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[faces].copy()).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[faces].copy()).float()
    dummy_labels = torch.zeros(len(faces), dtype=torch.long)
    area_mm2 = float(mesh.area_faces.sum())

    raw = (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, dummy_labels)
    pos, x, _ = transform(raw)
    return pos, x, area_mm2


def mesh_to_model_inputs_with_color(mesh: trimesh.Trimesh, transform: PatchPreTransformWithRealColor,
                                     flat_colors: tuple = None):
    """Same as mesh_to_model_inputs, for a checkpoint trained with real per-face color
    (dataset/patch_preprocessing_color.py's PatchPreTransformWithColor, feature_dim=27) instead of
    the original 24-dim geometry-only layout. mesh must carry mesh.visual.vertex_colors (real
    scanner color, threaded through decimation/merging/capping by every function above - see
    downsample_to_density's docstring for the mechanism) - raises if it doesn't, since silently
    falling back to zeros would feed the model a color distribution it never trained on without
    any signal that something's wrong.

    Per-face color is the mean of its 3 vertex colors, matching how training data is structured
    (dataset/patch_dataset.py's _stage_tensors builds mesh_vertices_normals/mesh_face_normals the
    same per-face way from per-vertex attributes).

    flat_colors: optional (tooth_color, gum_color) pair - when given, --color_mode flat
    (mesh_viewer_segmented.py): the real per-face color is classified and repainted via
    classify_and_flatten_colors before being fed to the model, but the RETURNED face_colors is
    still the real, unflattened value - snapshots (see _save_snapshot) should always record what
    the scanner actually captured, for checking the real distribution, regardless of which mode
    the model itself is being fed.

    Returns (pos, x, area_mm2, face_colors) - the extra face_colors (F, 3) uint8 return (vs.
    mesh_to_model_inputs's 3-tuple) is what actually got fed to the model this cycle, for
    snapshotting/diagnostics (see mesh_viewer_segmented.py's _save_snapshot) - lets a saved
    snapshot be checked against the training color distribution after the fact, not just
    predictions."""
    if mesh.visual.kind != 'vertex':
        raise ValueError('mesh_to_model_inputs_with_color: mesh has no vertex_colors - the scanner '
                          'connection, decimation, or merge step upstream dropped it. Check that '
                          'update_chunk was called with colors, not just vertices/triangles.')
    faces = mesh.faces
    mesh_faces = torch.from_numpy(faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[faces].copy()).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[faces].copy()).float()
    dummy_labels = torch.zeros(len(faces), dtype=torch.long)
    area_mm2 = float(mesh.area_faces.sum())

    vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
    face_colors = vertex_colors[faces].mean(axis=1).astype(np.uint8)

    fed_colors = face_colors
    if flat_colors is not None:
        tooth_color, gum_color = flat_colors
        fed_colors = classify_and_flatten_colors(face_colors, mesh.triangles_center, tooth_color, gum_color)

    raw = (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, dummy_labels)
    pos, x, _ = transform(raw, fed_colors)
    return pos, x, area_mm2, face_colors
