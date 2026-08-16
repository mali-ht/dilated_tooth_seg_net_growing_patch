import argparse
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_generator import grow_patch_full_sweep, load_cached_arch, pick_incisor_seed_label
from dataset.patch_preprocessing import PatchPreTransform
from models.patch_collate import PatchCollator
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork
from testing.visualize_patch_growth import local_view_axes, project_triangles, render_stage

# Qualitative check of a trained checkpoint on real validation patches: ground truth vs
# predicted vs a correct/wrong map, for a handful of val arches. Reuses get_full_sequence's own
# internal logic (val_seed + index, ds.min_occlusality - the FIXED value, not the randomized
# training range) rather than PatchTeeth3DSDataset.__getitem__ directly, since visualization
# wants the full-arch mesh + boolean mask for contextual rendering (background arch behind the
# highlighted patch), which __getitem__'s already-cropped tensors don't carry.


def render_correctness(ax, projected_tris, depth, shade, mask, correct, title):
    n = len(mask)
    facecolors = np.tile(np.array([0.96, 0.96, 0.97, 1.0]), (n, 1))
    edgecolors = np.tile(np.array([0.7, 0.7, 0.72, 1.0]), (n, 1))
    linewidths = np.full(n, 0.15)

    patch_idx = np.where(mask)[0]
    green = np.array([0.25, 0.7, 0.3])
    red = np.array([0.85, 0.2, 0.2])
    facecolors[patch_idx, :3] = np.where(correct[:, None], green, red)
    edgecolors[patch_idx] = [0.15, 0.15, 0.15, 0.5]
    linewidths[patch_idx] = 0.1
    facecolors[:, :3] *= shade[:, None]

    order = np.argsort(depth)
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(projected_tris[order], facecolors=facecolors[order],
                         edgecolors=edgecolors[order], linewidths=linewidths[order])
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)


def main(ckpt_path, root, processed_folder, num_classes, n_examples, start_index, stage_fraction, out_path, device):
    model = PatchLitDilatedToothSegmentationNetwork.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    ds = PatchTeeth3DSDataset(root, processed_folder=processed_folder, num_classes=num_classes, is_train=False,
                               train_test_split=1, val_seed=0, transform=None)
    collator = PatchCollator.for_network(model.model, seed=0)
    transform = PatchPreTransform()

    fig, axes = plt.subplots(n_examples, 3, figsize=(13, 4.2 * n_examples), squeeze=False)

    for row, index in enumerate(range(start_index, start_index + n_examples)):
        pt_path = join(ds.root, ds.processed_folder, ds.file_names[index])
        mesh, labels = load_cached_arch(pt_path)
        rng = np.random.default_rng(ds.val_seed + index)
        seed_label = pick_incisor_seed_label(labels, rng)
        masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, ds.width_mm, ds.height_mm,
                                               ds.slab_mm, ds.min_occlusality)
        stage_idx = int(round(stage_fraction * (len(masks) - 1)))
        mask = masks[stage_idx]
        face_idx = np.where(mask)[0]

        raw = ds._stage_tensors(mesh, labels, mask)
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, patch_labels, area_mm2 = raw
        pos, x, _ = transform(raw[:5])

        batch = collator([(pos, x, patch_labels, area_mm2)])
        batch = batch.to(device)
        with torch.no_grad():
            pred = model.predict_labels(batch)[0].cpu().numpy()

        pred_labels_full = labels.copy()
        pred_labels_full[face_idx] = pred
        correct = pred == patch_labels.numpy()
        acc = correct.mean()

        # per-patch mIoU over classes actually present, matching the training metric's own
        # zero-union exclusion (see models/patch_lightning_module.py)
        import torchmetrics as tm
        miou_metric = tm.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        miou = miou_metric(torch.from_numpy(pred), patch_labels).item()

        print(f"arch {index} ({ds.file_names[index]}), stage {stage_idx}/{len(masks) - 1} "
              f"({stages[stage_idx]}), {len(face_idx)} faces, area={area_mm2:.1f}mm2: "
              f"accuracy={acc:.1%}  mIoU={miou:.3f}")

        md_axis, secondary_axis, occlusal_axis = local_view_axes(mesh, mask)
        projected = project_triangles(mesh, md_axis, secondary_axis)
        depth = mesh.triangles_center @ occlusal_axis
        shade = np.clip(mesh.face_normals @ occlusal_axis, 0.25, 1.0)

        render_stage(axes[row][0], projected, depth, shade, mask, labels,
                     f"Ground truth\narch {index}, {len(face_idx)} faces", num_classes=num_classes)
        render_stage(axes[row][1], projected, depth, shade, mask, pred_labels_full,
                     f"Predicted\nacc={acc:.1%}  mIoU={miou:.3f}", num_classes=num_classes)
        render_correctness(axes[row][2], projected, depth, shade, mask, correct,
                            "Correct (green) / wrong (red)")

        patch_uv = projected[mask].reshape(-1, 2)
        pad = 2.0
        xlim = (patch_uv[:, 0].min() - pad, patch_uv[:, 0].max() + pad)
        ylim = (patch_uv[:, 1].min() - pad, patch_uv[:, 1].max() + pad)
        for ax in axes[row]:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    fig.suptitle(f"Validation predictions ({num_classes}-class) - {os.path.basename(ckpt_path)}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained checkpoint's predictions on real val patches")
    parser.add_argument("--ckpt", required=True, help="Path to a .ckpt file")
    parser.add_argument("--root", default="data/3dteethseg")
    parser.add_argument("--processed_folder", default="processed_w5")
    parser.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    parser.add_argument("--n_examples", type=int, default=4, help="How many val arches to show")
    parser.add_argument("--start_index", type=int, default=0, help="First val arch index")
    parser.add_argument("--stage_fraction", type=float, default=0.6,
                         help="0=earliest/smallest stage, 1=final full occlusal+facial+lingual stage")
    parser.add_argument("--out", default="patch_predictions.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.ckpt, args.root, args.processed_folder, args.num_classes, args.n_examples, args.start_index,
         args.stage_fraction, args.out, args.device)
