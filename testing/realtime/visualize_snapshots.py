#!/usr/bin/env python3
"""visualize_snapshots.py -- offline analysis/rendering of a live run's snapshot sequence
(testing/logs/snapshots/<run_id>/cycle_XXXX.npz, saved by LiveSegmenter._save_snapshot -
mesh_viewer_segmented.py enables this by default, see its --snapshot_every/--no_snapshots).

Built for a specific live-investigation question: a real run showed a small early patch (e.g. a
tooth #8/#9 occlusal surface) predicted with NEIGHBORING tooth numbers first, correcting itself
as the accumulated patch grew. The model's own area-gated dilation
(models/patch_dilated_tooth_seg_network.py, area_thresholds=(40, 180, 360)) is a concrete,
testable explanation: a small patch runs with the wider-context dilated blocks gated OFF, so it
only has local geometry to go on - exactly the situation where visually similar neighbor teeth
would be hardest to tell apart. This script renders the accumulated mesh colored by predicted
class at each snapshotted cycle, in a FIXED projection (fit once on the largest snapshot, reused
for every earlier one) so the same physical region lines up across frames - letting the actual
moment predictions "settle" be inspected directly against area_mm2/gate state, not guessed at.

Deliberately does NOT import mesh_viewer_segmented.py (open3d/torch/model stack) - only needs
utils/teeth_numbering.py's color tables, so this stays a fast, display-independent CLI tool.

Usage:
    python3 visualize_snapshots.py --snapshot_dir testing/logs/snapshots/<run_id>
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.teeth_numbering import coarse_label_to_colors, label_to_colors  # noqa: E402

GUM_COLOR_255 = np.array([255, 192, 203], dtype=np.uint8)  # matches mesh_viewer_segmented.py's palette


def colors_for_predictions(pred_labels, num_classes):
    """Same palette as the live viewer (mesh_viewer_segmented.py) - kept as a standalone copy
    here rather than importing that module, since importing it drags in open3d/torch/the model
    stack just for this one small, self-contained function."""
    if num_classes == 5:
        return coarse_label_to_colors(pred_labels).astype(np.uint8)
    colors = label_to_colors(pred_labels).astype(np.uint8)
    colors[pred_labels == 0] = GUM_COLOR_255
    return colors


def _load_snapshots(snapshot_dir):
    paths = sorted(glob.glob(os.path.join(snapshot_dir, "cycle_*.npz")))
    if not paths:
        raise SystemExit(f"no cycle_*.npz files found in {snapshot_dir}")
    snaps = []
    for p in paths:
        d = np.load(p)
        # guidance_* fields are only present in snapshots saved after that logging was added -
        # older runs' .npz files still load fine, just with guidance_state defaulting to "none"
        has_guidance = "guidance_state" in d.files
        snaps.append({
            "cycle": int(d["cycle"]), "vertices": d["vertices"], "faces": d["faces"],
            "pred_labels": d["pred_labels"], "area_mm2": float(d["area_mm2"]),
            "raw_faces": int(d["raw_faces"]), "gates": d["gates"].tolist(),
            "guidance_state": str(d["guidance_state"]) if has_guidance else "none",
            "guidance_target_label": int(d["guidance_target_label"]) if has_guidance else -1,
            "guidance_target_universal": int(d["guidance_target_universal"]) if has_guidance else -1,
            "guidance_distance_mm": float(d["guidance_distance_mm"]) if has_guidance else float("nan"),
            "guidance_compass": str(d["guidance_compass"]) if has_guidance else "",
            "guidance_current_pos": d["guidance_current_pos"] if has_guidance else None,
            "guidance_target_pos": d["guidance_target_pos"] if has_guidance else None,
        })
    return snaps


def _fit_projection(vertices):
    """Top-2 PCA axes of a point cloud -> a (3,2) projection matrix, same technique this project
    already uses elsewhere (spatial_split/live_preprocessing.py) for a coordinate-frame-agnostic
    'natural' 2D layout - here it doubles as keeping every panel in the SAME fixed orientation
    (fit once, reused for every snapshot) so the same physical region lines up across cycles."""
    centered = vertices - vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:2].T  # (3, 2)


def _render_panel(ax, snap, projection, num_classes):
    verts2d = snap["vertices"] @ projection
    face_xy = verts2d[snap["faces"]]  # (F, 3, 2)
    colors = colors_for_predictions(snap["pred_labels"], num_classes).astype(np.float64) / 255.0
    coll = PolyCollection(face_xy, facecolors=colors, edgecolors="none")
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    # guidance arrow overlay - same fixed projection as the mesh itself, so the arrow lines up
    # spatially with whatever it was actually pointing at, at that same cycle (the whole point of
    # logging scan/prediction/guidance together - see LiveSegmenter._save_snapshot's docstring)
    if snap["guidance_state"] == "active" and snap["guidance_current_pos"] is not None:
        cur2d = snap["guidance_current_pos"] @ projection
        tgt2d = snap["guidance_target_pos"] @ projection
        ax.annotate("", xy=tuple(tgt2d), xytext=tuple(cur2d),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14))

    gate_str = "".join("1" if g else "0" for g in snap["gates"])
    guidance_str = (f"-> label{snap['guidance_target_label']} "
                     f"({snap['guidance_distance_mm']:.0f}mm)" if snap["guidance_state"] == "active"
                     else f"[{snap['guidance_state']}]")
    ax.set_title(f"cycle {snap['cycle']}  area={snap['area_mm2']:.0f}mm2  gates={gate_str}\n"
                 f"raw={snap['raw_faces']:,} down={len(snap['faces']):,}  guidance {guidance_str}", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=17, choices=[5, 17])
    ap.add_argument("--max_panels", type=int, default=12, help="max cycles to render in the grid (evenly sampled)")
    ap.add_argument("--out", default=None, help="output PNG path (default: <snapshot_dir>/evolution_grid.png)")
    args = ap.parse_args()

    snaps = _load_snapshots(args.snapshot_dir)
    print(f"loaded {len(snaps)} snapshot(s), cycles {snaps[0]['cycle']}-{snaps[-1]['cycle']}")

    largest = max(snaps, key=lambda s: len(s["vertices"]))
    projection = _fit_projection(largest["vertices"])

    if len(snaps) > args.max_panels:
        idx = sorted(set(np.linspace(0, len(snaps) - 1, args.max_panels).round().astype(int).tolist()))
        sampled = [snaps[i] for i in idx]
    else:
        sampled = snaps

    n = len(sampled)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, snap in zip(axes, sampled):
        _render_panel(ax, snap, projection, args.num_classes)
    for ax in axes[len(sampled):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = args.out or os.path.join(args.snapshot_dir, "evolution_grid.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")

    # quantitative summary: per-cycle area/gates/histogram/guidance-target - a simple proxy for
    # "when did the predicted class set stop changing" (i.e. when did it stop being confused),
    # cross-referenced against gates/area_mm2 AND what the arrow was pointing at that same cycle,
    # without needing to look at any image at all
    print("\ncycle  area_mm2  gates  raw_faces   down_faces  guidance_target  distance_mm  classes_present")
    for s in snaps:
        present = sorted(int(c) for c in np.unique(s["pred_labels"]) if c != 0)
        gate_str = "".join("1" if g else "0" for g in s["gates"])
        if s["guidance_state"] == "active":
            target_str = f"label{s['guidance_target_label']}"
            dist_str = f"{s['guidance_distance_mm']:.1f}"
        else:
            target_str = f"[{s['guidance_state']}]"
            dist_str = "-"
        print(f"{s['cycle']:5d}  {s['area_mm2']:8.1f}  {gate_str}   {s['raw_faces']:9,d}  "
              f"{len(s['faces']):9,d}  {target_str:>15s}  {dist_str:>11s}  {present}")


if __name__ == "__main__":
    main()
