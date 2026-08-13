import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_generator import grow_patch_scanner_sweep, load_cached_arch, pick_incisor_seed_label
from utils.teeth_numbering import coarse_label_to_colors, label_to_colors, label_to_coarse_label

GUM_COLOR = np.array([1.0, 0.75, 0.8])  # light pink - gum's own teeth_numbering (17-class) color
                                          # is a near-invisible gray, too close to the background


def local_view_axes(mesh, mask):
    """
    PCA-derived (tangent, secondary, normal) axes, restricted to the vertices of `mask`'s faces
    rather than the whole mesh. A flat linear projection along whole-mesh axes distorts a
    sharply-curving horseshoe arch badly for any zoomed-in region far from the arch's own
    "center" (e.g. front teeth vs. molars) - fitting the view to the region actually being shown
    keeps growth sequences visually legible instead of exaggerating the arch's curvature.
    """
    idx = np.unique(mesh.faces[np.where(mask)[0]].reshape(-1))
    verts = mesh.vertices[idx]
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal_axis = eigvecs[:, 0]
    secondary_axis = eigvecs[:, 1]
    tangent_axis = eigvecs[:, 2]
    avg_normal = mesh.face_normals[np.where(mask)[0]].mean(axis=0)
    if np.dot(normal_axis, avg_normal) < 0:
        normal_axis = -normal_axis
    return tangent_axis, secondary_axis, normal_axis


def project_triangles(mesh, axis_u, axis_v):
    """Project every face's 3 vertices onto the (axis_u, axis_v) plane. Returns (F, 3, 2)."""
    tri = mesh.triangles  # (F, 3, 3)
    u = tri @ axis_u
    v = tri @ axis_v
    return np.stack([u, v], axis=-1)


def render_stage(ax, projected_tris, depth, shade, mask, labels, title, num_classes=17):
    # Some tooth-label colors (gum especially) are close in hue to a flat gray background, so a
    # patch made mostly of gum can visually camouflage into "background" even though it's
    # solidly connected - style background as a faint wireframe and the patch as solid fill with
    # a dark outline so the boundary is unambiguous regardless of label color.
    #
    # A flat top-down projection alone doesn't read as 3D: occlusal-facing tooth crowns and
    # near-vertical buccal/lingual walls end up the same flat color. `shade` (face normal dot
    # the viewing axis - standard Lambertian flat shading) darkens faces that face away from the
    # viewer, which is what actually makes individual tooth crowns visually pop out.
    #
    # The mesh also has real depth along the occlusal axis (~26mm), so faces can genuinely
    # overlap in projection - draw far-to-near (painter's algorithm) so the near surface
    # correctly occludes what's behind it.
    n = len(mask)
    facecolors = np.tile(np.array([0.96, 0.96, 0.97, 1.0]), (n, 1))
    edgecolors = np.tile(np.array([0.7, 0.7, 0.72, 1.0]), (n, 1))
    linewidths = np.full(n, 0.15)

    patch_idx = np.where(mask)[0]
    if len(patch_idx) > 0:
        if num_classes == 5:
            # the coarse palette already gives gum its own light-pink entry, no override needed
            facecolors[patch_idx, :3] = coarse_label_to_colors(label_to_coarse_label(labels[patch_idx])) / 255.0
        else:
            facecolors[patch_idx, :3] = label_to_colors(labels[patch_idx]) / 255.0
            gum_idx = patch_idx[labels[patch_idx] == 0]
            facecolors[gum_idx, :3] = GUM_COLOR
        edgecolors[patch_idx] = [0.15, 0.15, 0.15, 0.5]
        linewidths[patch_idx] = 0.1

    facecolors[:, :3] *= shade[:, None]

    order = np.argsort(depth)
    pc = PolyCollection(projected_tris[order], facecolors=facecolors[order],
                         edgecolors=edgecolors[order], linewidths=linewidths[order])
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)


def main(pt_path, out_path, seed, n_stages, width_mm, height_mm, slab_mm, min_occlusality, num_classes=17):
    mesh, labels = load_cached_arch(pt_path)

    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)

    masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng, width_mm=width_mm, height_mm=height_mm,
                                             slab_mm=slab_mm, min_occlusality=min_occlusality)
    n_stages = min(n_stages, len(masks))
    masks, order = masks[:n_stages], order[:n_stages]

    # fit the view to the region actually shown (the final/largest mask), not the whole arch
    md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, masks[-1])
    projected = project_triangles(mesh, md_axis, secondary_axis)
    depth = mesh.triangles_center @ occlusal_axis
    shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)

    fig, axes = plt.subplots(1, n_stages, figsize=(3.4 * n_stages, 3.6))
    if n_stages == 1:
        axes = [axes]
    for i, (ax, mask) in enumerate(zip(axes, masks)):
        n_faces = int(mask.sum())
        actual_area = mesh.area_faces[mask].sum()
        render_stage(ax, projected, depth, shade, mask, labels,
                     f"Position {i + 1}: tooth {order[i]} added\n{n_faces} faces, {actual_area:.0f} mm²\n"
                     f"teeth so far: {order[:i + 1]}", num_classes=num_classes)

    fig.suptitle(f"Scanner sweep, one tooth-footprint at a time ({num_classes}-class) - "
                 f"{pt_path.split('/')[-1]}, seed tooth {seed_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the scanner-sweep patch growth sequence on one cached arch")
    parser.add_argument("--pt_path", required=True, help="Path to a cached data_*.pt file")
    parser.add_argument("--out", default="patch_growth.png")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for picking the incisor seed tooth")
    parser.add_argument("--n_stages", type=int, default=6, help="How many tooth-positions to show")
    parser.add_argument("--width_mm", type=float, default=14.0)
    parser.add_argument("--height_mm", type=float, default=14.0)
    parser.add_argument("--slab_mm", type=float, default=16.0)
    parser.add_argument("--min_occlusality", type=float, default=0.6,
                         help="Occlusal-facing cutoff (higher = more restrictive to top-only, "
                              "lower/negative admits more of the buccal/lingual walls)")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17],
                         help="Color patches by the fine 17-class or coarse 5-class label scheme")
    args = parser.parse_args()
    main(args.pt_path, args.out, args.seed, args.n_stages, args.width_mm, args.height_mm, args.slab_mm,
         args.min_occlusality, args.num_classes)
