import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import (compute_occlusality, grow_patch_coverage, grow_patch_scanner_sweep,
                                      load_cached_arch, pick_incisor_seed_label)
from testing.visualize_patch_growth import local_view_axes, project_triangles, render_stage


def main(pt_path, out_path, seed, n_stages, threshold_start, threshold_end, num_classes=17):
    mesh, labels = load_cached_arch(pt_path)

    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)

    # extent grows tooth-by-tooth (the scanner-sweep strategy); coverage threshold relaxes in
    # lockstep from restrictive (early scan: just the tips) to permissive (late/full scan: full
    # wall depth) - together this approximates the actual curriculum idea: small extent + low
    # coverage is the common/early state, growing toward full arch + full depth. Coverage's own
    # combine logic (grow_patch_coverage) is unchanged here - only the extent source changed.
    extent_masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng)
    n_stages = min(n_stages, len(extent_masks))
    extent_masks, order = extent_masks[:n_stages], order[:n_stages]
    thresholds = np.linspace(threshold_start, threshold_end, n_stages)

    md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, extent_masks[-1])
    occlusality = compute_occlusality(mesh, occlusal_axis)
    combined_masks = [grow_patch_coverage(mesh, extent_masks[i], occlusality, thresholds[i], labels)
                       for i in range(n_stages)]

    projected = project_triangles(mesh, md_axis, secondary_axis)
    depth = mesh.triangles_center @ occlusal_axis
    shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)

    fig, axes = plt.subplots(1, n_stages, figsize=(3.4 * n_stages, 3.6))
    if n_stages == 1:
        axes = [axes]
    for i, (ax, mask) in enumerate(zip(axes, combined_masks)):
        n_faces = int(mask.sum())
        actual_area = mesh.area_faces[mask].sum()
        present_labels = sorted(set(labels[mask].tolist()))
        render_stage(ax, projected, depth, shade, mask, labels,
                     f"Position {i + 1}: tooth {order[i]} added, t={thresholds[i]:+.2f}\n"
                     f"{n_faces} faces, {actual_area:.0f} mm²\nlabels: {present_labels}", num_classes=num_classes)

    fig.suptitle(f"Combined extent+coverage curriculum ({num_classes}-class) - {pt_path.split('/')[-1]}, "
                 f"seed tooth {seed_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize extent and coverage growing together as one curriculum")
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--out", default="patch_combined.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_stages", type=int, default=6, help="How many tooth-positions to show")
    parser.add_argument("--threshold_start", type=float, default=0.85, help="Coverage threshold at position 1 (restrictive)")
    parser.add_argument("--threshold_end", type=float, default=-0.4, help="Coverage threshold at the last position (permissive)")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17],
                         help="Color patches by the fine 17-class or coarse 5-class label scheme")
    args = parser.parse_args()
    main(args.pt_path, args.out, args.seed, args.n_stages, args.threshold_start, args.threshold_end, args.num_classes)
