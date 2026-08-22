import argparse
import os
import sys

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_patch_network_color import get_datasets
from utils.teeth_numbering import coarse_label_to_colors, label_to_colors, label_to_coarse_label

# New, additive-only file - nothing existing is edited. Renders REAL items pulled straight out of
# train_dataset[index] (dataset/patch_dataset_whole_tooth.py's PatchTeeth3DSDatasetWithWholeTooth,
# via train_patch_network_color.py's own get_datasets() - the literal function training calls, not
# a reimplementation that could drift from it) as ACTUAL 3D meshes (.ply export + an interactive
# Open3D window), not a 2D matplotlib projection - an earlier version of this file did the latter;
# rewritten 2026-08-22 per explicit request for real, rotatable/zoomable 3D output instead.
#
# Geometry comes from x[:, :9] (dataset/patch_preprocessing.py's PatchPreTransform: each face's 3
# vertex positions, patch-centroid-centered and /scale_mm normalized) reshaped to (F, 3, 3) and
# turned into an UNSHARED-vertex mesh (3 fresh vertices per face, faces=arange(3F).reshape(F,3))
# so each face can carry its own flat, non-interpolated color with sharp boundaries - same
# technique as realtime/mesh_viewer_segmented.py's flat_shaded_mesh (not imported from there to
# avoid pulling in that file's Open3D GUI-app-level dependencies just for a 10-line reshape; the
# reconstruction below already produces the same unshared-vertex structure that helper builds).
#
# Color comes from x[:, 24:27] (only present for a color-trained --color_style run, feature_dim=27
# - see dataset/patch_preprocessing_color.py / _realistic_color_hsv.py) decoded back from the
# [-1,1] range every color transform in this codebase uses (x*127.5+127.5) to real 0-255 RGB - if
# --color_dropout_prob > 0, dropped-out faces will render as the flat sentinel gray, which is
# itself useful to see (confirms the dropout is doing what it's supposed to).


def reconstruct_mesh_and_colors(pos, x, labels, num_classes, color_by):
    n_faces = x.shape[0]
    tris = x[:, :9].reshape(-1, 3, 3).numpy().astype(np.float64)
    verts = tris.reshape(-1, 3)
    faces = np.arange(3 * n_faces).reshape(n_faces, 3)

    labels_np = labels.numpy()
    if color_by == 'labels':
        if num_classes == 5:
            face_colors = coarse_label_to_colors(label_to_coarse_label(labels_np))
        else:
            face_colors = label_to_colors(labels_np)
            face_colors[labels_np == 0] = [255, 191, 204]  # light pink - same GUM_COLOR override
            # visualize_patch_growth.py uses (17-class gum's real palette entry is a near-invisible
            # gray, easy to lose against the background)
    else:
        face_colors = ((x[:, 24:27] + 1.0) * 127.5).clamp(0, 255).numpy()

    vertex_colors = np.repeat(face_colors, 3, axis=0).astype(np.float64) / 255.0
    return verts, faces, vertex_colors


def to_o3d_mesh(verts, faces, vertex_colors):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    m.compute_vertex_normals()
    return m


def main(args):
    train_dataset, _ = get_datasets(
        args.root, args.processed_folder, args.train_test_split, args.num_classes,
        args.early_bias_power, args.max_faces, args.color_style, args.whole_tooth_patch_prob,
        color_dropout_prob=args.color_dropout_prob)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(train_dataset), size=args.n_examples, replace=False) if args.random \
        else range(args.start_index, args.start_index + args.n_examples)

    os.makedirs(args.out_dir, exist_ok=True)
    o3d_meshes = []
    x_cursor = 0.0
    gap_mm = 5.0

    for index in indices:
        # ONE draw per index, both colorings built from it - train mode draws fresh randomness
        # every call (PatchTeeth3DSDataset's own docstring), so calling train_dataset[index] twice
        # (once per coloring) would show two DIFFERENT patches under the same index, not the same
        # patch colored two ways.
        pos, x, labels, area_mm2 = train_dataset[int(index)]
        n_faces = x.shape[0]
        width = None

        for color_by in ('labels', 'fed_color'):
            verts, faces, vcolors = reconstruct_mesh_and_colors(pos, x, labels, args.num_classes, color_by)

            out_path = os.path.join(args.out_dir, f"patch_{int(index):05d}_{color_by}.ply")
            trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=(vcolors * 255).astype(np.uint8),
                             process=False).export(out_path)
            print(f"arch-index {index} ({color_by}): {n_faces} faces, area={area_mm2:.1f}mm2 -> {out_path}")

            if args.show:
                mesh = to_o3d_mesh(verts, faces, vcolors)
                if width is None:
                    width = verts[:, 0].max() - verts[:, 0].min()
                y_row = 0.0 if color_by == 'labels' else -(verts[:, 1].max() - verts[:, 1].min() + gap_mm)
                mesh.translate((x_cursor - verts[:, 0].min(), y_row - verts[:, 1].min(), 0))
                o3d_meshes.append(mesh)

        if args.show:
            x_cursor += width + gap_mm

    if args.show:
        print(f"\nOpening an interactive window with {len(indices)} patch(es) side by side - top "
              f"row = ground-truth labels, bottom row = fed color. Close the window to exit.")
        o3d.visualization.draw_geometries(o3d_meshes, window_name="train_dataset[i]: labels (top) / fed color (bottom)",
                                           mesh_show_back_face=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export/view real items pulled from train_patch_network_color.py's own "
                     "train_dataset[index] - the literal tensors the model is fed, decoded back "
                     "into real 3D meshes (.ply export, optional interactive Open3D window).")
    parser.add_argument("--root", default="data/3dteethseg")
    parser.add_argument("--processed_folder", default="processed_w5")
    parser.add_argument("--train_test_split", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    parser.add_argument("--early_bias_power", type=float, default=2.5)
    parser.add_argument("--max_faces", type=int, default=20000)
    parser.add_argument("--color_style", choices=["flat", "mosaic"], default="mosaic")
    parser.add_argument("--whole_tooth_patch_prob", type=float, default=0.3)
    parser.add_argument("--color_dropout_prob", type=float, default=0.0)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--start_index", type=int, default=0, help="First arch index (ignored if --random)")
    parser.add_argument("--random", action="store_true", help="Pick --n_examples random indices instead of sequential")
    parser.add_argument("--seed", type=int, default=0, help="Seeds --random's index choice; each drawn "
                         "item still draws its own fresh randomness (train mode - see "
                         "PatchTeeth3DSDataset's docstring), so a re-run looks different on purpose.")
    parser.add_argument("--out_dir", default="training_input_meshes", help="Where .ply files are written")
    parser.add_argument("--show", action="store_true", help="Also open an interactive Open3D window "
                         "with all examples placed side by side")
    args = parser.parse_args()
    main(args)
