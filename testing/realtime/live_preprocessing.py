import time

import numpy as np
import pyfqmr
import torch
import trimesh

from dataset.patch_preprocessing import PatchPreTransform
from debug_log import logger


def _debug(msg):
    logger.debug(f"[live_preprocessing] {msg}")

# Bridges the live mesh_wire stream (testing/realtime/mesh_wire.py, untouched) to the trained
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


def downsample_to_density(mesh: trimesh.Trimesh, target_density: float = 5.0) -> trimesh.Trimesh:
    """Quadric-decimate mesh down to ~target_density faces/mm^2, matching processed_w5's training
    density exactly (same formula, same decimator settings as mesh_dataset.py._donwscale_mesh).
    Returns mesh unchanged if it's already at or below that density (nothing to remove)."""
    target_count = max(round(mesh.area * target_density), 4)
    if target_count >= len(mesh.faces):
        _debug(f"downsample_to_density: {len(mesh.faces)} faces already <= target {target_count} "
               f"(area={mesh.area:.1f}mm2) - no-op")
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
        mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[keep])
        _debug(f"downsample_to_density: trimmed decimation overshoot {len(keep)} -> {target_count} "
               f"via uniform random subsample")

    return mesh_simple


def spatial_split(vertices: np.ndarray, faces: np.ndarray, n_splits: int):
    """Splits one raw chunk's faces into n_splits spatially-coherent groups (PCA-axis ordering of
    face centroids, then even chunks along that axis) - lets a single very large incoming chunk's
    decimation spread across n_splits pool workers instead of running as one pyfqmr call pinned to
    a single core (a single simplify_mesh() call can't itself be parallelized - REALTIME.md
    Finding #9 - so a chunk large enough on its own to dominate a cycle's wall time needs to be
    split before submission, not just submitted alongside others).

    Spatially-coherent splits, not a random face subset: this project's own earlier testing found
    a random subset fragments badly under pyfqmr's decimator (see downsample_to_density's
    docstring and testing/realtime/test_live_segmenter.py's _spatial_chunks, which this mirrors).
    Returns a list of (piece_vertices, piece_faces) tuples, each with its own local vertex
    indexing - may return fewer than n_splits pieces if the chunk is too small to split evenly."""
    if n_splits <= 1 or len(faces) <= 1:
        return [(vertices, faces)]
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
        pieces.append((vertices[used_v], remap[sub_faces]))
    return pieces


def decimate_chunk_worker(vertices: np.ndarray, faces: np.ndarray, target_density: float):
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
    itself instead of coordinating log writes across process boundaries."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    t0 = time.time()
    down = downsample_to_density(mesh, target_density)
    return down.vertices, down.faces, time.time() - t0


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
    above, just applied to the final cap instead of a single chunk."""
    if len(mesh.faces) <= max_faces:
        return mesh
    t0 = time.time()
    rng = np.random.default_rng()
    keep = rng.choice(len(mesh.faces), size=max_faces, replace=False)
    capped = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)
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
    close to the same physical point) - see testing/realtime/test_live_segmenter.py for both
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
    cycle - see mesh_viewer_segmented.py's LiveSegmenter for why that mattered)."""
    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        _debug("merge_meshes: nothing to merge yet")
        return None
    t0 = time.time()
    all_vertices, all_faces = [], []
    offset = 0
    for m in meshes:
        all_vertices.append(m.vertices)
        all_faces.append(m.faces + offset)
        offset += len(m.vertices)
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    merged = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
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
