import time
from dataclasses import dataclass

import numba
import numpy as np
import torch
from scipy.spatial import cKDTree

# Component 4 (claude_code_prompt.md) - see models/patch_layer.py's module docstring for the
# full reasoning. Short version: of the 4 O(N^2) pairwise-distance computations the network would
# otherwise do per forward pass, 2 are purely a function of metric face-center position - fixed
# before the network ever runs - so they never need to happen on GPU at all:
#   - EdgeGraphConvBlock 1's local kNN (it's given the true pos, unlike blocks 2/3 which use their
#     own learned features)
#   - all 3 DilatedEdgeGraphConvBlocks' neighbor selection - candidate gathering AND the
#     subsequent FPS down-sample to k, since both operate on pos alone (confirmed in the paper:
#     "the dilated neighborhood graph is computed on the face coordinates in metric space and
#     therefore remains static")
# This module precomputes both, on CPU, via a KD-tree (scipy.spatial.cKDTree - O(N log N) per
# query, replacing an O(N^2) dense distance matrix) plus a small numpy farthest-point-sampling
# implementation (the installed CUDA FPS kernel, pointnet2_ops, only runs on CUDA tensors - see
# TRAINING_CONCERNS.md's torch_cluster note - so a CPU-only patch needs its own FPS). Only the 2
# genuinely dynamic feature-space kNN lookups (blocks 2/3) are left for the network to compute
# on-the-fly; there's no way to precompute those, since they depend on the network's own
# mid-forward activations.
#
# Nearest-neighbor RANKING is unaffected by PatchPreTransform's normalization (a fixed per-patch
# translation + uniform positive scale preserves distance order), so running the KD-tree on the
# already-normalized `pos` the dataset returns gives identical neighbor indices to running it on
# raw mm coordinates - no special-casing needed.
#
# Both the KD-tree query and the FPS step are batched across all N faces at once rather than
# looped per-face in Python: a first version that looped per face measured ~20s to collate a
# single ~24,000-face patch, and even after batching the KD-tree query (workers=-1) and
# vectorizing FPS with numpy broadcasting, a single large patch still took ~25-35s - almost
# entirely inside FPS itself. Chasing it down: a plain numpy-vectorized FPS iteration allocates a
# fresh (N, dilation_k, 3) difference array every one of its k=32 iterations (e.g. ~625MB for a
# 29k-face patch at dilation_k=1800) and computes an einsum reduction over all of it - correct,
# but memory-bandwidth-bound, and repeated 32 times per dilated block, 3 blocks per patch. A
# numba-JIT'd version (below) fuses the whole per-point distance+argmax computation into one pass
# with no large temporaries, and parallelizes across the N faces with numba.prange - measured
# ~90-100x faster than the numpy-vectorized version at realistic patch sizes (0.04s vs 3.6s at
# N=5083, m=1800, k=32; 0.18s vs an extrapolated ~18s at N=25000). numba is a pure CPU JIT
# compiler (LLVM, compiled for whatever CPU it runs on) - unlike torch_cluster, there's no CUDA
# architecture/wheel-compatibility risk, and it was already present in this environment.


@numba.njit(parallel=True, cache=True)
def _gather_candidate_positions_numba(pos_np, cand_idx_all):
    """pos_np: (total_N, 3) float32 - every face's position, the same array cKDTree was built on.
    cand_idx_all: (N, max_k) int64 - each row is one face's candidate neighbor indices, ascending
    distance. Returns (N, max_k, 3) float32: cand_idx_all's indices resolved to actual positions.

    Same gather plain numpy already did (pos_np[cand_idx_all]) - added because profiling a live
    cycle (REALTIME.md) found it was the single largest cost in the whole pipeline (bigger than
    the model's entire forward pass), and it was running single-threaded: numpy fancy indexing
    isn't itself multi-threaded. The KD-tree query right before this already needed workers=-1 for
    the identical reason (this file's own comment: "~4.3s single-threaded vs ~0.3s parallel"); this
    is the same fix applied to the next line's gather, following models.layer.fps's C-order-loop
    style and numba.prange like _fps_batched_numba below already does for the FPS step itself."""
    n, m = cand_idx_all.shape
    out = np.empty((n, m, 3), dtype=np.float32)
    for i in numba.prange(n):
        for j in range(m):
            idx = cand_idx_all[i, j]
            out[i, j, 0] = pos_np[idx, 0]
            out[i, j, 1] = pos_np[idx, 1]
            out[i, j, 2] = pos_np[idx, 2]
    return out


@numba.njit(parallel=True, cache=True)
def _fps_batched_numba(points, k, farthest0):
    """points: (N, M, 3) float32, N independent candidate pools of M points each. farthest0: (N,)
    int64 starting index per pool (the random first pick). Returns (N, k) int64 indices into the
    M dimension - one iterative farthest-point-sampling run per pool, all N run in parallel via
    numba.prange. Same algorithm as models.layer.fps's pure-Python fallback and this module's
    prior pure-numpy _fps_batched (random start, then repeatedly take the point farthest from
    everything selected so far) - just compiled, with the per-iteration distance update and
    argmax fused into a single pass instead of separate large-array numpy operations."""
    n, m, _ = points.shape
    selected = np.empty((n, k), dtype=np.int64)
    distance = np.full((n, m), np.inf, dtype=np.float32)
    for b in numba.prange(n):
        farthest = farthest0[b]
        for i in range(k):
            selected[b, i] = farthest
            cx = points[b, farthest, 0]
            cy = points[b, farthest, 1]
            cz = points[b, farthest, 2]
            best_d = -1.0
            best_j = 0
            for j in range(m):
                dx = points[b, j, 0] - cx
                dy = points[b, j, 1] - cy
                dz = points[b, j, 2] - cz
                d = dx * dx + dy * dy + dz * dz
                if d < distance[b, j]:
                    distance[b, j] = d
                if distance[b, j] > best_d:
                    best_d = distance[b, j]
                    best_j = j
            farthest = best_j
    return selected


def _fps_batched(points, k, rng):
    """points: (N, M, 3). Returns (N, k) indices into the M dimension - see
    _fps_batched_numba for the actual (compiled) implementation; this just handles the
    k >= M passthrough case and draws the random starting point per pool."""
    n, m, _ = points.shape
    if k >= m:
        return np.broadcast_to(np.arange(m, dtype=np.int64), (n, m)).copy()
    farthest0 = rng.integers(0, m, size=n).astype(np.int64)
    return _fps_batched_numba(points, k, farthest0)


def precompute_neighbor_indices(pos_np, k, dilation_ks, rng, profile=False):
    """
    One KD-tree build + one query, sliced to serve block 1's local kNN and all len(dilation_ks)
    dilated blocks' candidate-gather-then-FPS. Returns (local_idx, dilated_idx_list):
      - local_idx: (N, k_eff) int64, self excluded (matches models.patch_layer.knn's convention)
      - dilated_idx_list[i]: (N, fps_k_eff) int64 of ORIGINAL face indices, self INCLUDED in the
        candidate pool before FPS (matches DilatedEdgeGraphConvBlock's on-the-fly
        torch.topk(cd, dilation_k) call, which has no self-exclusion, unlike the local blocks)
    k_eff / fps_k_eff are the usual min(..., N)-style clamps for small patches.

    profile=False (default, used by training's PatchCollator.__call__) returns the usual 2-tuple
    with no overhead. profile=True (live-inference debug investigation only, see REALTIME.md)
    additionally returns a per-stage timing dict as a third element - kdtree build vs. query vs.
    each dilation level's FPS pass, so a slow collate step can be attributed to a specific stage
    instead of just the single bundled t_collate number LiveSegmenter already tracked.
    """
    timings = {} if profile else None
    n = pos_np.shape[0]
    max_k = min(max(k, max(dilation_ks)), n)

    t0 = time.time()
    tree = cKDTree(pos_np)
    if profile:
        timings['kdtree_build'] = time.time() - t0

    # workers=-1: use all CPU cores - single-threaded query was the dominant cost by far (~4.3s
    # vs ~0.3s parallel, measured on a 24-core machine for a 29k-face patch at k=1800)
    t0 = time.time()
    _, cand_idx_all = tree.query(pos_np, k=max_k, workers=-1)
    if profile:
        timings['kdtree_query'] = time.time() - t0

    # cand_pos_all materializes an (N, max_k, 3) float32 array - at N~17k, max_k~1800 that's
    # ~370MB. Timed separately from kdtree_query above: a first profiling pass left this untimed
    # and the numbers didn't add up (t_collate noticeably bigger than kdtree_query + all FPS
    # stages combined) - this gather is where the rest of it was, and it turned out to be the
    # single largest cost in the whole live-inference cycle (see REALTIME.md) - bigger than the
    # model's entire forward pass. Two fixes, both verified live in that profiling pass:
    #   1. copy=False on both astype calls (not the plain .astype(dtype) used elsewhere in this
    #      file, which always copies even when the dtype already matches): pos_np is a torch
    #      tensor's .numpy() view of PatchPreTransform's `torch.zeros(s, 24).float()` output, so
    #      it's already float32 - the original .astype(np.float32) was silently doing a SECOND
    #      full-array copy of this exact 370MB result for nothing (cKDTree.query's index dtype,
    #      np.intp, is likewise already int64 on this platform). This alone roughly halved the
    #      cost.
    #   2. The actual gather now runs through _gather_candidate_positions_numba (parallel via
    #      numba.prange) instead of plain numpy fancy indexing (single-threaded) - same fix this
    #      file's own kdtree_query already needed (see its comment) applied one line later.
    t0 = time.time()
    cand_idx_all = np.atleast_2d(cand_idx_all).astype(np.int64, copy=False)  # (N, max_k), ascending distance
    cand_pos_all = _gather_candidate_positions_numba(pos_np.astype(np.float32, copy=False), cand_idx_all)
    if profile:
        timings['candidate_gather'] = time.time() - t0

    k_eff = max(0, min(k, n - 1))
    local_idx = cand_idx_all[:, 1:k_eff + 1]  # drop self (column 0, distance 0)

    dilated_idx = []
    for i, dk in enumerate(dilation_ks):
        dk_eff = min(dk, n)
        fps_k_eff = min(k, dk_eff)
        t0 = time.time()
        local_sel = _fps_batched(cand_pos_all[:, :dk_eff], fps_k_eff, rng)  # (N, fps_k_eff)
        if profile:
            timings[f'fps_dilation{i}_k{dk}'] = time.time() - t0
        dilated_idx.append(np.take_along_axis(cand_idx_all[:, :dk_eff], local_sel, axis=1))

    if profile:
        return local_idx, dilated_idx, timings
    return local_idx, dilated_idx


@dataclass
class PatchBatch:
    """Everything models.patch_dilated_tooth_seg_network.PatchDilatedToothSegmentationNetwork's
    forward() needs, already batch_size=1-shaped and ready to unpack:
        model(batch.x, batch.pos, area_mm2=batch.area_mm2, local_idx=batch.local_idx,
              dilated_idx=batch.dilated_idx)
    labels stays separate (for the loss), not part of the model call."""
    pos: torch.Tensor
    x: torch.Tensor
    labels: torch.Tensor
    area_mm2: float
    local_idx: torch.Tensor
    dilated_idx: tuple

    def to(self, device):
        """PatchBatch is a plain dataclass, not a tensor/dict/tuple - Lightning's Trainer only
        knows how to auto-move those to the GPU. Implementing .to(device) is Lightning's
        documented pattern for custom batch types, so Trainer.fit's automatic CPU->GPU transfer
        works without needing to override transfer_batch_to_device."""
        return PatchBatch(
            pos=self.pos.to(device),
            x=self.x.to(device),
            labels=self.labels.to(device),
            area_mm2=self.area_mm2,
            local_idx=self.local_idx.to(device),
            dilated_idx=tuple(d.to(device) for d in self.dilated_idx),
        )


class PatchCollator:
    """
    collate_fn for DataLoader(patch_dataset, batch_size=1, collate_fn=PatchCollator(...)) - see
    the Component 4 batching decision (TRAINING_CONCERNS.md): batch_size=1 + gradient
    accumulation, not padding, so there's exactly one patch per collate call and no masking to
    thread through. Expects the dataset's default (pos, x, labels, area_mm2) 4-tuple output (i.e.
    PatchTeeth3DSDataset with its default PatchPreTransform, not transform=None).

    Build with the SAME k / dilation_ks as the network instance this feeds - easiest via
    PatchCollator.for_network(model), which reads them directly off the model so the two can't
    silently drift apart.
    """

    def __init__(self, k=32, dilation_ks=(200, 900, 1800), seed=None):
        self.k = k
        self.dilation_ks = tuple(dilation_ks)
        self.seed = seed

    @classmethod
    def for_network(cls, model, seed=None):
        return cls(k=model.k, dilation_ks=model.dilation_ks, seed=seed)

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError(f'PatchCollator only supports batch_size=1 (got {len(batch)}) - '
                              'see the Component 4 batching decision in TRAINING_CONCERNS.md')
        pos, x, labels, area_mm2 = batch[0]
        rng = np.random.default_rng(self.seed)
        pos_np = pos.numpy()

        local_idx, dilated_idx = precompute_neighbor_indices(pos_np, self.k, self.dilation_ks, rng)

        return PatchBatch(
            pos=pos.unsqueeze(0),
            x=x.unsqueeze(0),
            labels=labels.unsqueeze(0),
            area_mm2=float(area_mm2),
            local_idx=torch.from_numpy(local_idx).unsqueeze(0),
            dilated_idx=tuple(torch.from_numpy(d).unsqueeze(0) for d in dilated_idx),
        )
