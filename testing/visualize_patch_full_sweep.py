import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import (TOOTH_LABELS, compute_arch_center, grow_patch_full_sweep,
                                      load_cached_arch, pick_incisor_seed_label, tooth_facial_direction,
                                      tooth_local_frame, tooth_tangent_direction)
from testing.visualize_patch_growth import local_view_axes, project_triangles, render_stage


def sample_indices(n, k):
    """k evenly-spaced indices into range(n), always including the first and last (the phase's
    start and its fully-swept end state)."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def compute_lingual_view_axis(mesh, labels, secondary_axis):
    """
    A top-down projection can't show the facial/lingual walls - the tooth's own bulk occludes
    them from directly above. Pick the sign of secondary_axis (the "side" PCA axis - roughly the
    arch's buccal-lingual spread) that points lingually (inward, toward the tongue), by comparing
    against the average facial direction across all present teeth. Viewing along the negation of
    this (camera outside the arch looking in) puts the facial wall face-on; viewing along this
    axis itself (camera inside, in the tongue space, looking out) puts the lingual wall face-on.
    """
    present = set(np.unique(labels).tolist()) & set(TOOTH_LABELS)
    arch_center = compute_arch_center(mesh, labels)
    facial_dirs = []
    for t in present:
        origin, normal_axis = tooth_local_frame(mesh, labels, t)
        tangent_axis = tooth_tangent_direction(mesh, labels, t, present, normal_axis)
        facial_dirs.append(tooth_facial_direction(origin, normal_axis, tangent_axis, arch_center))
    avg_facial = np.mean(facial_dirs, axis=0)
    return secondary_axis if np.dot(secondary_axis, avg_facial) < 0 else -secondary_axis


def main(pt_path, out_path, seed, per_phase, num_classes):
    mesh, labels = load_cached_arch(pt_path)

    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)

    masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng)

    # sample each phase's stage range independently so all three phases stay represented
    # regardless of how many teeth each one covers
    phase_names = [s[0] for s in stages]
    panels = []
    for phase in ["occlusal", "facial", "lingual"]:
        idx_in_phase = [i for i, p in enumerate(phase_names) if p == phase]
        for i in sample_indices(len(idx_in_phase), per_phase):
            panels.append(idx_in_phase[i])

    md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, masks[-1])
    lingual_axis = compute_lingual_view_axis(mesh, labels, secondary_axis)

    # top-down camera: looks along occlusal_axis, screen = the other two axes. Shows occlusal
    # clearly; facial/lingual walls are occluded by the tooth's own bulk from directly above.
    top_projected = project_triangles(mesh, md_axis, lingual_axis)
    top_depth = mesh.triangles_center @ occlusal_axis
    top_shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)

    # side camera looking from outside the arch inward (along -lingual_axis, i.e. facially):
    # screen = (md_axis, occlusal_axis), the two axes perpendicular to lingual_axis. Puts the
    # facial wall face-on.
    facial_projected = project_triangles(mesh, md_axis, occlusal_axis)
    facial_depth = mesh.triangles_center @ (-lingual_axis)
    facial_shade = np.clip(mesh.face_normals @ (-lingual_axis), 0.25, 1.0)

    # side camera looking from inside the arch (tongue space) outward, along lingual_axis: same
    # screen plane as facial, opposite viewing direction. Puts the lingual wall face-on.
    lingual_projected = project_triangles(mesh, md_axis, occlusal_axis)
    lingual_depth = mesh.triangles_center @ lingual_axis
    lingual_shade = np.clip(mesh.face_normals @ lingual_axis, 0.25, 1.0)

    views = {
        "occlusal": ("top-down", top_projected, top_depth, top_shade),
        "facial": ("outside-in", facial_projected, facial_depth, facial_shade),
        "lingual": ("inside-out", lingual_projected, lingual_depth, lingual_shade),
    }

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 3.6))
    if n_panels == 1:
        axes = [axes]
    for ax, stage_idx in zip(axes, panels):
        mask = masks[stage_idx]
        phase, tooth = stages[stage_idx]
        view_name, projected, depth, shade = views[phase]
        n_faces = int(mask.sum())
        actual_area = mesh.area_faces[mask].sum()
        render_stage(ax, projected, depth, shade, mask, labels,
                     f"[{phase}, {view_name} view] stage {stage_idx + 1}/{len(masks)}: tooth {tooth}\n"
                     f"{n_faces} faces, {actual_area:.0f} mm²", num_classes=num_classes)

    fig.suptitle(f"Full 3-phase sweep (occlusal -> facial -> lingual), {num_classes}-class - "
                 f"{pt_path.split('/')[-1]}, seed tooth {seed_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the full occlusal->facial->lingual scan sweep")
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--out", default="patch_full_sweep.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per_phase", type=int, default=4, help="How many stages to show per phase")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    args = parser.parse_args()
    main(args.pt_path, args.out, args.seed, args.per_phase, args.num_classes)
