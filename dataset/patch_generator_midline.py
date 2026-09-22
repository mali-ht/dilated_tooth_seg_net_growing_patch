import numpy as np

from dataset.patch_generator import rectangular_footprint, tooth_local_frame

# New, additive-only file - dataset/patch_generator.py and dataset/patch_dataset.py are both
# untouched. Implements a MIDLINE-seeded early-growth stage: real-hardware testing (2026-08-24,
# Docs/REALTIME.md) showed the live scanner's very first captured patch is typically centered ON
# THE BOUNDARY between the two central incisors (labels 8/9), roughly evenly split between them -
# something the existing scan-sweep curriculum never produces, since it always seeds on ONE
# tooth's own local frame (pick_incisor_seed_label + tooth_local_frame) and only lets the
# neighbor bleed in as a small minority fraction. Measured empirically against 15 real cached
# arches: the OTHER central incisor makes up only ~7-8% of a single-tooth-seeded stage-0 patch on
# average (2-3/30 samples over 15%) - nothing like the ~41/36% near-even split actually observed
# live. This is also not a new observation: the same failure (5 classes predicted on a ~53mm2
# patch that's really just 2 teeth) was first flagged 2026-08-15 (see train_patch_network_color.py's
# own header comment) and "lowering area_thresholds/early_bias_power" was tried as a mitigation
# since then (both are now tuned lower) - the failure still reproduces on 2026-08-19/21/24 runs,
# including checkpoints trained with those tuned values, so a reweighting-only fix wasn't enough;
# this targets the actual missing DATA SHAPE instead.
#
# Empirically tuned against 15 real cached upper arches (both central-incisor seeds each):
# min_size_mm=8/max_size_mm=22/n_stages=8 gives stage 0-2 (~25-58mm2, brackets the live ~54mm2
# observation) as PURE {8, 9, gum} - no 6/7/10/11 bleed at all - then grows smoothly into the
# lateral incisors (stage 3+) and canines (stage 6+), so there's no hard cliff into the existing
# scan-sweep curriculum's own territory. slab_mm/min_occlusality sweeps (4-8mm / 0.6-0.85) showed
# the ~45% gum fraction at stage 0 doesn't come from over-generous filtering - it's the real
# interdental papilla, anatomically unavoidable at the exact midline - so those were left at the
# base curriculum's own defaults rather than hand-tuned to chase a lower number.

CENTRAL_INCISOR_LABELS = (8, 9)


def midline_local_frame(mesh, labels, tooth_a=8, tooth_b=9):
    """Local frame straddling the boundary between two adjacent teeth (default: the central
    incisors) - origin at their midpoint, normal_axis the average of their own occlusal normals,
    tangent_axis pointing from tooth_a's origin toward tooth_b's (i.e. along the arch, across the
    midline), projected into the plane perpendicular to normal_axis. Mirrors tooth_local_frame's/
    tooth_tangent_direction's own conventions but built from two teeth at once instead of one."""
    origin_a, normal_a = tooth_local_frame(mesh, labels, tooth_a)
    origin_b, normal_b = tooth_local_frame(mesh, labels, tooth_b)
    origin = (origin_a + origin_b) / 2
    normal_axis = normal_a + normal_b
    normal_axis = normal_axis / np.linalg.norm(normal_axis)

    tangent_axis = origin_b - origin_a
    tangent_axis = tangent_axis - np.dot(tangent_axis, normal_axis) * normal_axis
    tangent_axis = tangent_axis / np.linalg.norm(tangent_axis)
    return origin, normal_axis, tangent_axis


def grow_patch_midline_seed(mesh, labels, min_size_mm=8.0, max_size_mm=22.0, n_stages=8,
                             slab_mm=16.0, min_occlusality=0.6, tooth_a=8, tooth_b=9):
    """
    A short growth sequence of nested, symmetric footprints centered on the tooth_a/tooth_b
    boundary (see midline_local_frame), widening from min_size_mm to max_size_mm over n_stages -
    the earliest, smallest stages are the whole point (see module docstring); later stages exist
    so this sequence can feed the same early_bias_power-weighted stage sampler
    (PatchTeeth3DSDataset._sample_stage_index) the base scan-sweep curriculum does, without a
    hard cliff at the end. Sizes are square (width_mm == height_mm == size) - this is deliberately
    NOT trying to reproduce the full arch sweep; once a stage has grown past a couple of teeth's
    worth of extent, the base scan-sweep/whole-tooth curricula already cover that territory.

    Caller must confirm both tooth_a and tooth_b are present in this arch first (some real arches
    are missing one) - unlike the scan-sweep functions this mirrors, there's no fallback here.
    """
    origin, normal_axis, tangent_axis = midline_local_frame(mesh, labels, tooth_a, tooth_b)
    sizes = np.linspace(min_size_mm, max_size_mm, n_stages)

    masks = []
    accumulated = np.zeros(len(mesh.faces), dtype=bool)
    for size_mm in sizes:
        footprint = rectangular_footprint(mesh, origin, normal_axis, tangent_axis, size_mm, size_mm,
                                           slab_mm, min_occlusality)
        accumulated = accumulated | footprint
        masks.append(accumulated.copy())
    return masks
