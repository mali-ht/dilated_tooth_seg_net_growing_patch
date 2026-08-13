import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import (compute_occlusality, grow_patch_coverage, grow_patch_scanner_sweep,
                                      load_cached_arch, pick_incisor_seed_label)
from testing.visualize_patch_growth import local_view_axes, project_triangles, render_stage


def main(pt_path, out_path, seed, n_teeth, thresholds, num_classes=17):
    mesh, labels = load_cached_arch(pt_path)

    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)

    # fix ONE extent region (a few teeth, via the scanner-sweep extent strategy), then vary the
    # coverage threshold against it - coverage's own combine logic is unchanged/revisited later
    extent_masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng)
    extent_mask = extent_masks[min(n_teeth, len(extent_masks)) - 1]

    md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, extent_mask)
    occlusality = compute_occlusality(mesh, occlusal_axis)

    projected = project_triangles(mesh, md_axis, secondary_axis)
    depth = mesh.triangles_center @ occlusal_axis
    shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)

    n_stages = len(thresholds)
    fig, axes = plt.subplots(1, n_stages, figsize=(3.4 * n_stages, 3.6))
    for i, (ax, t) in enumerate(zip(axes, thresholds)):
        mask = grow_patch_coverage(mesh, extent_mask, occlusality, t, labels)
        n_faces = int(mask.sum())
        frac_of_extent = n_faces / extent_mask.sum() * 100
        present_labels = sorted(set(labels[mask].tolist()))
        render_stage(ax, projected, depth, shade, mask, labels,
                     f"threshold={t:+.1f}: {n_faces} faces ({frac_of_extent:.0f}% of extent)\n"
                     f"labels: {present_labels}", num_classes=num_classes)

    fig.suptitle(f"Coverage threshold sweep ({num_classes}-class, fixed extent, {int(extent_mask.sum())} faces, "
                 f"teeth {order[:n_teeth]}) - {pt_path.split('/')[-1]}, seed tooth {seed_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the coverage-threshold sweep against a fixed extent")
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--out", default="patch_coverage.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_teeth", type=int, default=4, help="How many scanner-sweep tooth-positions to fix as the extent")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3, 0.0, -0.3],
                         help="Occlusality thresholds, high (restrictive) to low (permissive)")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17],
                         help="Color patches by the fine 17-class or coarse 5-class label scheme")
    args = parser.parse_args()
    main(args.pt_path, args.out, args.seed, args.n_teeth, args.thresholds, args.num_classes)
