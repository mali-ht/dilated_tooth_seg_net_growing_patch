import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import (INCISOR_LABELS, TOOTH_LABELS, compute_occlusality, grow_patch_coverage,
                                      grow_patch_full_sweep, grow_patch_scanner_sweep, load_cached_arch,
                                      pick_incisor_seed_label, rectangular_footprint, tooth_local_frame,
                                      tooth_tangent_direction)
from testing.visualize_patch_growth import local_view_axes
from utils.teeth_numbering import label_to_coarse_label


def _n_components(mesh, mask):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0
    a, b = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    n = len(mesh.faces)
    adj = csr_matrix((np.ones(len(a) * 2), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
    sub = adj[idx][:, idx]
    n_comp, _ = connected_components(sub, directed=False)
    return n_comp


def _check_stage_sequence(mesh, labels, masks, stage_descriptions, whole_mask_connected=False):
    """
    whole_mask_connected=True (coverage growth, grow_patch_coverage): a single best-first flood
    fill from one seed, so the WHOLE accumulated mask is provably one connected component by
    construction at every threshold - a real, checkable invariant.

    whole_mask_connected=False (extent/tooth-sweep growth, sweep_footprints-based): each stage
    unions in an INDEPENDENT footprint from a different tooth's own local frame.
    rectangular_footprint() guarantees that single new footprint is one connected blob (see
    _check_footprint_connectivity below for a direct test of that) - it does NOT guarantee the
    footprint touches whatever was already accumulated, nor that subtracting the accumulated
    mask from it leaves a connected remainder (a raw footprint that partially overlaps the prior
    stage can have that overlap sit in the middle of it, splitting the non-overlapping remainder
    into >1 piece - checked and confirmed benign on this exact failure case: tooth 9's raw
    170-face footprint was one component; 50 of those faces already existed in tooth 10's
    footprint, and removing them split the other 120 into two). So connectivity isn't a
    meaningful thing to assert here at all - only monotonic growth and label validity are.
    """
    prev_mask = None
    for mask, desc in zip(masks, stage_descriptions):
        n_faces = int(mask.sum())
        is_superset = prev_mask is None or np.all(mask[prev_mask])
        if whole_mask_connected:
            n_comp = _n_components(mesh, mask)
            print(f"  {desc}: faces={n_faces:6d}  components={n_comp}  monotonic_growth={is_superset}")
            assert n_comp <= 1, f"{desc} is not a single connected component (found {n_comp})"
        else:
            print(f"  {desc}: faces={n_faces:6d}  monotonic_growth={is_superset}")
        assert is_superset, f"{desc} dropped faces present in the previous stage"
        if n_faces > 0:
            assert np.all(np.isin(labels[mask], list(range(17)))), f"{desc} has an out-of-range label"
        prev_mask = mask


def _check_footprint_connectivity(mesh, labels, min_occlusality=0.6, width_mm=14.0, height_mm=14.0, slab_mm=16.0):
    """
    Direct regression test for the isolated-island bug (see rectangular_footprint's docstring):
    for every present tooth's own occlusal footprint, computed independently (no accumulated
    mask involved, so no overlap-splitting confound), the result must be a single connected
    component landing on/near that tooth - not a stray disconnected sliver.
    """
    present = sorted(set(np.unique(labels).tolist()) & set(TOOTH_LABELS))
    for tooth_label in present:
        origin, normal_axis = tooth_local_frame(mesh, labels, tooth_label)
        tangent_axis = tooth_tangent_direction(mesh, labels, tooth_label, present, normal_axis)
        footprint = rectangular_footprint(mesh, origin, normal_axis, tangent_axis, width_mm, height_mm, slab_mm,
                                           min_occlusality)
        n_faces = int(footprint.sum())
        n_comp = _n_components(mesh, footprint)
        assert n_comp <= 1, f"tooth {tooth_label}'s own footprint is not a single connected component (found {n_comp})"
        assert n_faces > 0, f"tooth {tooth_label} got an empty footprint"
    print(f"  ok - all {len(present)} present teeth's own occlusal footprints are single connected components")


def main(pt_path):
    mesh, labels = load_cached_arch(pt_path)

    rng = np.random.default_rng(0)
    seed_label = pick_incisor_seed_label(labels, rng)
    assert seed_label in INCISOR_LABELS, "seed must be an incisor"

    print("rectangular_footprint: per-tooth connectivity (isolated-island regression check):")
    _check_footprint_connectivity(mesh, labels)

    print("extent growth (scanner sweep, tooth by tooth):")
    extent_masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng)
    extent_masks, order = extent_masks[:6], order[:6]
    _check_stage_sequence(mesh, labels, extent_masks,
                           [f"position {i + 1} (tooth {t} added)" for i, t in enumerate(order)])

    print("coverage growth (against the position-4 extent, descending threshold):")
    _, secondary_axis, occlusal_axis = local_view_axes(mesh, extent_masks[-1])
    occlusality = compute_occlusality(mesh, occlusal_axis)
    fixed_extent = extent_masks[3]
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.0, -0.3]
    coverage_masks = [grow_patch_coverage(mesh, fixed_extent, occlusality, t, labels) for t in thresholds]
    _check_stage_sequence(mesh, labels, coverage_masks, [f"threshold {t:+.1f}" for t in thresholds],
                           whole_mask_connected=True)
    for mask in coverage_masks:
        assert np.all(mask <= fixed_extent), "coverage growth escaped its extent mask"

    print("full 3-phase sweep (occlusal -> facial -> lingual):")
    full_masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng)
    _check_stage_sequence(mesh, labels, full_masks, [f"[{p}] tooth {t}" for p, t in stages])
    last_index_of = {phase: max(i for i, (p, t) in enumerate(stages) if p == phase)
                      for phase in ("occlusal", "facial", "lingual")}
    occlusal_final = full_masks[last_index_of["occlusal"]]
    facial_final = full_masks[last_index_of["facial"]]
    lingual_final = full_masks[last_index_of["lingual"]]
    facial_new = facial_final & ~occlusal_final
    lingual_new = lingual_final & ~facial_final
    assert facial_new.sum() > 0, "facial phase added nothing new over occlusal"
    assert lingual_new.sum() > 0, "lingual phase added nothing new over facial"
    assert (facial_new & lingual_new).sum() == 0, "facial and lingual additions overlap"
    print(f"  ok - occlusal={occlusal_final.sum()}  +facial={facial_new.sum()}  +lingual={lingual_new.sum()}"
          f"  (no overlap between facial/lingual additions)")

    print("coarse (5-class) label remap:")
    coarse = label_to_coarse_label(labels)
    assert coarse.shape == labels.shape
    assert np.all((coarse >= 0) & (coarse <= 4)), "coarse label out of [0,4] range"
    for l in range(17):
        m = labels == l
        if m.sum() > 0:
            assert len(set(coarse[m].tolist())) == 1, f"fine label {l} maps to multiple coarse labels"
    print(f"  ok - {len(set(coarse.tolist()))} distinct coarse classes present")

    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the patch generator on one cached arch")
    parser.add_argument("--pt_path", default="data/3dteethseg/processed_w5/data_00OMSZGW_lower.pt")
    args = parser.parse_args()
    main(args.pt_path)
