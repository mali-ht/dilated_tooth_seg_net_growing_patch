import numpy as np
from scipy.spatial import cKDTree

from dataset.patch_generator import order_teeth_from_seed

# New, additive-only file - dataset/patch_generator.py and dataset/patch_dataset.py are both
# untouched. Implements the "Whole-tooth / multi-tooth patch curriculum" proposed in
# Docs/MAA_ToDo.md (steps 1-3 of that proposal): patches bounded by GROUND-TRUTH TOOTH LABEL
# instead of the existing scan-sweep's local rectangular footprint - giving the model training
# examples that are explicitly "this is one (or a few) COMPLETE tooth/teeth, cleanly bounded",
# which the local-footprint sweep patches never provide regardless of how large they grow.
#
# Boundary margin: user-decided 2026-08-19 (Docs/MAA_ToDo.md step 2's open design question) -
# a pure tooth-label mask is unlike anything a real scan produces (always some margin of
# adjacent gum/tooth at a real capture's edge), so DEFAULT_MARGIN_MM dilates each tooth's mask
# spatially before use, matching how the existing scan-sweep footprints already admit adjacent
# gum at their own edges.

DEFAULT_MARGIN_MM = 1.5


def whole_tooth_mask(labels: np.ndarray, tooth_ids) -> np.ndarray:
    """Boolean mask: faces whose label is one of tooth_ids (may be a single tooth or several)."""
    return np.isin(labels, list(tooth_ids))


def dilate_mask_by_margin(mesh, mask: np.ndarray, margin_mm: float) -> np.ndarray:
    """Spatially dilate a face mask: any face whose centroid is within margin_mm of the NEAREST
    already-in-mask face's centroid also gets included. margin_mm<=0 or an empty mask is a no-op
    (returns mask unchanged) - the pure-tooth-label-boundary case is just this function called
    with margin_mm=0, not a separate code path."""
    if margin_mm <= 0 or not mask.any():
        return mask
    centroids = mesh.triangles_center
    in_tree = cKDTree(centroids[mask])
    dists, _ = in_tree.query(centroids, k=1)
    return mask | (dists <= margin_mm)


def grow_patch_whole_teeth(mesh, labels, seed_label, rng, margin_mm: float = DEFAULT_MARGIN_MM):
    """Sequential "one whole tooth at a time" growth curriculum, directly analogous to
    grow_patch_full_sweep's occlusal/facial/lingual growth stages but bounded by tooth LABEL
    instead of a geometric footprint. Reuses order_teeth_from_seed (dataset/patch_generator.py,
    untouched) for the visiting order - same seed-outward, random-side, missing-tooth-aware
    traversal the existing scan-sweep uses - so stage i's mask is the (margin-dilated) union of
    the first i+1 teeth in that same anatomical order, giving "mostly single tooth, occasionally
    a few CONTIGUOUS teeth" for free once the caller applies its own early-bias stage sampling
    (PatchTeeth3DSDataset._sample_stage_index already does this - no separate tooth-count sampler
    needed, mirroring the doc proposal's own noted parallel to the scan-sweep's stage bias).

    Each tooth's margin dilation is computed ONCE per tooth (not once per stage) and stages are
    built via cumulative OR of those - mathematically identical to dilating the growing union
    directly (nearest-point-distance dilation distributes over union), just without redundantly
    re-querying the same already-dilated teeth at every larger stage.

    Returns (masks, stages): masks[i] is a (F,) bool array, stages[i] is the tuple of tooth
    labels included at that stage (for debugging/visualization - same shape of return as
    grow_patch_full_sweep's own (masks, stages) pair, so callers don't need to know which
    curriculum produced them)."""
    present = set(int(l) for l in np.unique(labels) if l != 0)
    order = order_teeth_from_seed(seed_label, present, rng)

    per_tooth_dilated = {t: dilate_mask_by_margin(mesh, whole_tooth_mask(labels, [t]), margin_mm)
                          for t in order}

    masks, stages = [], []
    cumulative = np.zeros(len(labels), dtype=bool)
    included = []
    for tooth in order:
        cumulative = cumulative | per_tooth_dilated[tooth]
        included.append(tooth)
        masks.append(cumulative.copy())
        stages.append(tuple(included))
    return masks, stages
