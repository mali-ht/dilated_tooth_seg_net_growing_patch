import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -
# this file lives in realtime/ (one level below root), same pattern testing/*.py already uses
from dataset.patch_color_augmentation import GUM_RGB_HIGH, GUM_RGB_LOW, TOOTH_RGB_HIGH, TOOTH_RGB_LOW  # noqa: E402

# Same value as realtime/live_preprocessing.py's NO_COLOR_SENTINEL - inlined rather than
# imported to avoid that module's own sys.path dependency on being run from within
# realtime/ (it does `from debug_log import logger`, a bare import that only resolves
# with that directory already on sys.path).
NO_COLOR_SENTINEL = np.array([102, 102, 102], dtype=np.int64)

# New, additive-only script - reads real captured color out of realtime/mesh_viewer_
# segmented.py's snapshot .npz files (face_colors, added 2026-08-19 specifically for this) across
# one or more live-hardware runs, and reports per-channel mean/std/min/max for teeth vs. gum,
# directly comparable against dataset/patch_color_augmentation.py's TOOTH_RGB_LOW/HIGH and
# GUM_RGB_LOW/HIGH training ranges. The point: SyntheticColorPaint currently paints a single flat
# value per class-group with zero intra-class variance - confirmed empirically that this mismatch
# with real scanner color's natural variance costs real accuracy (Docs/REALTIME.md). These stats
# are meant to inform a future SyntheticColorPaint revision that draws from a distribution
# actually shaped like what the scanner produces (a real mean/std per class, not an arbitrary
# guessed noise magnitude) instead of a flat value.
#
# IMPORTANT CAVEAT: there is no ground truth for real hardware data, so faces are bucketed into
# "teeth" vs "gum" using the model's OWN predicted labels (pred_labels, already saved in every
# snapshot) - this is circular to a degree (the predictions being used to interpret the color data
# are themselves a product of a model with a known color-domain-gap issue) and the bucketing is
# only as trustworthy as those predictions were. Prefer aggregating a run where predictions looked
# qualitatively reasonable (e.g. a --color_mode flat run, which scored better than --color_mode
# varying per the live testing this script was built to inform) over a run known to have been
# struggling badly.
#
# Sentinel-contaminated faces (NO_COLOR_SENTINEL - a real bug fixed 2026-08-19, see
# realtime/live_preprocessing.py's merge_meshes docstring: chunks the wire protocol sent
# with no color data at all) are excluded entirely - they are not real captured color and would
# just add a spurious [102,102,102] spike to both buckets.


def load_run_colors_and_labels(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "*.npz")))
    all_colors, all_labels = [], []
    for f in files:
        d = np.load(f)
        if not bool(d["has_face_colors"]):
            continue
        fc = d["face_colors"].astype(np.int64)
        pl = d["pred_labels"].astype(np.int64)
        if len(fc) != len(pl):
            print(f"  [!] {f}: face_colors ({len(fc)}) / pred_labels ({len(pl)}) length mismatch - skipped")
            continue
        all_colors.append(fc)
        all_labels.append(pl)
    if not all_colors:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_colors, axis=0), np.concatenate(all_labels, axis=0)


def summarize(name, colors, low_range=None, high_range=None):
    if len(colors) == 0:
        print(f"  {name}: no faces")
        return
    colors = colors.astype(np.float64)
    mean = colors.mean(axis=0)
    std = colors.std(axis=0)
    print(f"  {name}: n={len(colors)}")
    print(f"    mean RGB = {mean.round(1)}")
    print(f"    std  RGB = {std.round(1)}   <- use this to shape a future SyntheticColorPaint's noise")
    print(f"    min  RGB = {colors.min(axis=0).astype(int)}")
    print(f"    max  RGB = {colors.max(axis=0).astype(int)}")
    if low_range is not None:
        in_range = np.all((colors >= low_range) & (colors <= high_range), axis=1)
        print(f"    fraction within current training range {low_range}-{high_range}: {in_range.mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze real captured color from live-run snapshots, bucketed into teeth/gum "
                     "via the model's OWN predictions (no ground truth exists for real hardware data - "
                     "see this file's own module docstring for that caveat)")
    parser.add_argument("run_dirs", nargs="+",
                         help="one or more realtime/logs/snapshots/<run_id>/ directories")
    args = parser.parse_args()

    all_colors, all_labels = [], []
    for run_dir in args.run_dirs:
        c, l = load_run_colors_and_labels(run_dir)
        print(f"{run_dir}: {len(c)} faces with real color loaded")
        all_colors.append(c)
        all_labels.append(l)
    colors = np.concatenate(all_colors, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    is_sentinel = np.all(colors == NO_COLOR_SENTINEL, axis=1)
    print(f"\ntotal faces: {len(colors)}, sentinel/no-real-data (excluded): "
          f"{is_sentinel.sum()} ({is_sentinel.mean()*100 if len(colors) else 0:.1f}%)")
    colors, labels = colors[~is_sentinel], labels[~is_sentinel]

    is_gum = labels == 0
    print(f"\n=== TEETH (predicted label != 0), n={(~is_gum).sum()} ===")
    summarize("teeth", colors[~is_gum], TOOTH_RGB_LOW, TOOTH_RGB_HIGH)
    print(f"  current training TOOTH range: {TOOTH_RGB_LOW} - {TOOTH_RGB_HIGH}")

    print(f"\n=== GUM (predicted label == 0), n={is_gum.sum()} ===")
    summarize("gum", colors[is_gum], GUM_RGB_LOW, GUM_RGB_HIGH)
    print(f"  current training GUM range: {GUM_RGB_LOW} - {GUM_RGB_HIGH}")


if __name__ == "__main__":
    main()
