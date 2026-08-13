import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_augmentation import SimToRealAugment
from dataset.patch_generator import grow_patch_scanner_sweep, load_cached_arch, pick_incisor_seed_label
from testing.visualize_patch_growth import local_view_axes, project_triangles, render_stage


def main(pt_path, out_path, seed, stage, num_classes):
    mesh, labels = load_cached_arch(pt_path)
    rng = np.random.default_rng(seed)
    seed_label = pick_incisor_seed_label(labels, rng)
    masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng)
    stage = min(stage, len(masks) - 1)
    mask = masks[stage]
    print(f"base patch (stage {stage}, teeth {order[:stage + 1]}): {int(mask.sum())} faces")

    aug = SimToRealAugment()
    aug_rng = np.random.default_rng(seed + 1000)
    aug_mask = aug.augment_mask(mesh, mask, aug_rng)
    aug_face_idx = np.where(aug_mask)[0]
    aug_patch_faces = mesh.faces[aug_face_idx]
    vertices, vertex_normals, face_normals = aug.augment_geometry(
        mesh.vertices, mesh.vertex_normals, mesh.face_normals, aug_patch_faces, aug_face_idx, aug_rng)
    print(f"augmented patch: {int(aug_mask.sum())} faces "
          f"({int(mask.sum()) - int(aug_mask.sum())} removed by holes/erosion/density subsample)")

    # build a second trimesh sharing the original topology but with jittered vertex positions, so
    # the existing render_stage/project_triangles helpers (which read from a trimesh object) work
    # unmodified on the augmented geometry too
    aug_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    aug_mesh.face_normals = face_normals  # override trimesh's own recomputed normals with ours

    # fit the view to the ORIGINAL (larger) mask so both panels share the same framing - otherwise
    # a shrunk augmented mask would get re-centered/re-scaled and the comparison wouldn't read as
    # "the same patch, perturbed"
    md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, mask)
    projected = project_triangles(mesh, md_axis, secondary_axis)
    aug_projected = project_triangles(aug_mesh, md_axis, secondary_axis)
    depth = mesh.triangles_center @ occlusal_axis
    aug_depth = aug_mesh.triangles_center @ occlusal_axis
    shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)
    aug_shade = np.clip(aug_mesh.face_normals @ occlusal_axis, 0.25, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    render_stage(axes[0], projected, depth, shade, mask, labels,
                 f"Original\n{int(mask.sum())} faces", num_classes=num_classes)
    render_stage(axes[1], aug_projected, aug_depth, aug_shade, aug_mask, labels,
                 f"Sim-to-real augmented\n{int(aug_mask.sum())} faces "
                 f"(holes/erosion/density + position/normal noise)", num_classes=num_classes)

    # zoom both panels to the (original) patch's own bounding box, padded a bit - at whole-arch
    # scale the hole/erosion/noise effects are too small to see clearly
    patch_uv = projected[mask].reshape(-1, 2)
    pad = 2.0
    xlim = (patch_uv[:, 0].min() - pad, patch_uv[:, 0].max() + pad)
    ylim = (patch_uv[:, 1].min() - pad, patch_uv[:, 1].max() + pad)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    fig.suptitle(f"Sim-to-real augmentation ({num_classes}-class) - {pt_path.split('/')[-1]}, "
                 f"seed tooth {seed_label}, stage {stage}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sim-to-real augmentation (before/after) on one patch")
    parser.add_argument("--pt_path", default="data/3dteethseg/processed_w5/data_00OMSZGW_lower.pt")
    parser.add_argument("--out", default="patch_augmentation.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, default=4, help="Which scanner-sweep stage (tooth count) to augment")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    args = parser.parse_args()
    main(args.pt_path, args.out, args.seed, args.stage, args.num_classes)
