#!/bin/bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
# RENAMED 2026-08-22 (from run_experiment_matrix.sh) - the comprehensive independent-aspect
# ablation matrix (gating/RGB/dilation_ks/whole-tooth/color-dropout/early_bias_power/
# area_thresholds/both Transformer architectures, 11 runs total) now targets the 4-GPU server
# instead, split across run_experiment_matrix_gpu0.sh..gpu3.sh (one parallel lane per GPU - see
# those files' own headers). THIS file is kept as a single-GPU fallback / this-machine reference,
# not the primary matrix going forward - a smaller 6-run subset of the same axes, still valid to
# run here if the server isn't available.
#
# Rewritten 2026-08-22 for THIS machine (24-core / RTX 5090, torch 2.12.1+cu132) - the version
# previously on disk here was written by a SEPARATE Claude Code session on the 12-core/RTX-3090
# SERVER box (its `dtsegnet` conda env, ~7-hr/epoch estimates etc. don't apply on this machine).
# That earlier 3-run matrix (mosaic+tuned-thresholds, +whole-tooth, +dilation_ks) is DONE - see
# checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned{,_wholetooth,
# _wholetooth_dilationks}/ - CMD1/CMD2 results below, CMD3 (dilation_ks on top of whole-tooth) was
# never actually run on this machine; superseded by run [4] below testing dilation_ks in isolation
# instead of stacked on whole-tooth, now that we have a working RGB/gating/whole-tooth ablation
# design to be consistent with.
#
# THIS is the real ablation matrix requested 2026-08-22: isolate gating, RGB, and dilation_ks
# effectiveness (the three open unknowns explicitly asked for), each as ONE change against a
# single fixed ANCHOR config, at LONG (200-epoch) schedules - run [1]'s own 200-epoch extension
# (run_extend_run1_200ep.sh) showed val_miou still climbing at epoch 199, not plateaued, so a
# 100-epoch comparison between any two configs here risks the same "which one would've won at
# 200?" ambiguity that made run [2]'s (whole-tooth) 100-epoch result inconclusive against run [1].
#
# NOT INCLUDED - deliberately, per explicit direction 2026-08-22: the two new Transformer
# architectures (models/patch_dilated_tooth_seg_transformer_network.py and
# ..._transformer_deep_network.py, --architecture dilated_transformer[--add_late_global]) are NOT
# part of this matrix. They stay as their own standalone runs, launched separately, e.g.:
#   python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic \
#     --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 \
#     --architecture dilated_transformer --epochs 200 \
#     --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer
# (add --add_late_global for the deep variant) - not scripted here since they're explicitly out of
# scope for this matrix, not because anything about running them differs.
#
# ============================================================================================
# THE ANCHOR CONFIG (every run below changes exactly ONE thing away from this):
#   17-class, dilation_gating=on, color_style=mosaic (RealisticColorPaintHSV - see
#   dataset/patch_color_augmentation.py), area_thresholds=20/90/180 (tuned), early_bias_power=1.5
#   (tuned), whole_tooth_patch_prob=0.0, dilation_ks=200/900/1800 (paper default),
#   color_dropout_prob=0.0, architecture=dilated (no transformer), 200 epochs.
#
# Run [1] below (the anchor itself, tagged _hsv) is NOT just a placeholder baseline - it's a real,
# open question in its own right: the mosaic color augmentation changed from RealisticColorPaint
# (independent-per-channel RGB noise) to RealisticColorPaintHSV (correlated hue/sat/val noise)
# AFTER the existing Patch_Seg_17class_gateOn_color_mosaic_tuned checkpoint (0.8317 @ epoch 199)
# was trained - see dataset/patch_preprocessing_realistic_color_hsv.py's docstring for why. That
# checkpoint used the OLD color scheme and is deliberately left alone (not overwritten - see the
# distinct _hsv-suffixed experiment_version below) so it stays available as a direct before/after
# comparison point for the color-augmentation fix itself, alongside whatever this matrix finds.
#
# RUNTIME (THIS machine, 24-core/RTX 5090): [1]'s own config (tuned thresholds, mosaic color,
# whole_tooth=0.0) measured ~7:44-7:54/epoch steady-state, 2026-08-19/20 (TensorBoard event-log
# timestamps, Patch_Seg_17class_gateOn_color_mosaic_tuned's first 3 epochs) - call it ~7:50/epoch,
# ~26 hours for 200 epochs. [4] (dilation_ks widened) measured ~33% slower per-batch from
# dilation_ks alone (the OTHER machine's own note, directionally expected to hold here too, exact
# multiplier not re-measured on THIS box) - budget ~34 hours. The rest are within noise of the
# anchor's own pace. TOTAL FOR ALL 6 RUNS: roughly 26+27+26+34+26+26 =~ 165 HOURS =~ 7 DAYS
# of continuous sequential compute. This is a real, substantial commitment - if that's too much,
# trim runs from the bottom of this file (least-novel-question-first) rather than shortening every
# run's epoch count, given the plateau problem explained above.
#
# Run from the REPO ROOT, inside tmux/screen (a dropped session over multiple days will otherwise
# kill it):
#   bash training_launch/run_experiment_matrix_this_machine.sh
# If a run dies, resume it rather than restarting - checkpoints/<experiment_version>/last.ckpt is
# written every epoch:
#   python3 train_patch_network_color.py <same flags as that run's CMD> --ckpt <path to last.ckpt>
# ============================================================================================

set -e          # stop the queue on the first failure, don't burn hours on a broken config
set -o pipefail # WITHOUT this, `python3 ... | tee ...`'s exit status is tee's (always 0), so
                # set -e alone never sees a crashed training run - confirmed the hard way
                # 2026-08-19: a run crashed with a CUDA error partway through and the script
                # silently moved on to the next one anyway instead of stopping.
                # Piping python3's stdout straight into tee also makes it a non-terminal
                # (isatty()==False), which is why Lightning's rich progress bar only showed static
                # snapshots instead of a live-updating epoch bar - confirmed empirically. Every
                # python3 invocation below runs through `script -qec "cmd" /dev/null` instead of
                # plain `python3 cmd`: it allocates a real pty so the live bar renders, while still
                # piping to tee for logging. `-e` makes script exit with the CHILD's real exit
                # code (verified: propagates true/false/exit N correctly), so set -e/pipefail above
                # still catch a real crash, not just a wrapper failure.

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

ANCHOR_FLAGS="--num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200"

echo "[1/6] ANCHOR: tuned thresholds, mosaic color (HSV-fixed), whole_tooth=0.0, dilation_ks default, gating on - 200 epochs"
# Also the real HSV-color-fix validation - see the header's own note on why this isn't just a
# placeholder baseline. Everything else below is measured AGAINST this run.
CMD1="python3 train_patch_network_color.py $ANCHOR_FLAGS --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv.log

echo "[2/6] GATING OFF: dilation_gating=off, everything else = anchor"
# Tests whether area-gating the 3 dilated blocks (skip a block on patches too small to have
# reached area_thresholds[i]) is actually earning its keep, or just adding inference-time
# complexity for no real accuracy gain - dilation_gating=off means all 3 dilated blocks always
# run regardless of patch area (models/patch_dilated_tooth_seg_network.py's _resolve_gates).
# Directly answers the "gating effectiveness" question from the 2026-08-22 request.
CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating off --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --experiment_version Patch_Seg_17class_gateOff_color_mosaic_tuned_hsv"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOff_color_mosaic_tuned_hsv.log

echo "[3/6] RGB ABLATION: color_style=none (feature_dim=24, no color channel at all), everything else = anchor"
# Tests RGB's actual contribution to accuracy, isolated from every other axis this project has
# tuned around color (thresholds/whole-tooth/etc. all stay at their tuned values - only the color
# CHANNEL itself is removed). Uses --color_style none (dataset/patch_preprocessing.py's plain
# PatchPreTransform, added 2026-08-22 specifically for this run - train_patch_network.py, the
# original pre-color script, was NOT used: it never received any of the tuned-threshold/whole-
# tooth/dilation_ks flags added to train_patch_network_color.py over this whole project, so it
# can't run an apples-to-apples comparison against the anchor's config).
CMD3="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style none --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --experiment_version Patch_Seg_17class_gateOn_colorNone_tuned"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_colorNone_tuned.log

echo "[4/6] DILATION_KS WIDENED: 200/1200/3000 (was 200/900/1800), everything else = anchor"
# Re-tests the "wider local context helps distinguish arch-position-confused teeth (4 vs 5, 12 vs
# 13)" hypothesis from Docs/REALTIME.md's premolar-confusion finding, in ISOLATION this time (not
# stacked on whole-tooth, unlike the earlier deferred CMD3 in this file's history) - block 1
# (dilation_ks[0]=200) unchanged (near-field, not the block the arch-position hypothesis is
# about), block 2 900->1200 (+33%), block 3 1800->3000 (+67%, the block already active at the
# >4000mm^2 area regime where REALTIME.md's confusion persists even with full dilated context).
# See models/patch_dilated_tooth_seg_network.py's dilation_ks docstring for why 200/900/1800 are
# the paper's own values, never re-derived for this project's patch distribution.
CMD4="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --dilation_ks 200 1200 3000 --epochs 200 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_dilationks"
script -qec "$CMD4" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_dilationks.log

echo "[5/6] WHOLE-TOOTH CURRICULUM ON: whole_tooth_patch_prob=0.3, everything else = anchor"
# Re-test of the SAME comparison run [2/3] (old matrix) made at 100 epochs (0.7471 @ 90, below
# run [1]'s old 0.7730 @ 84) - inconclusive then because neither run had reached its actual
# ceiling (run [1]'s own 200-epoch extension kept climbing well past epoch 100). This time both
# the anchor [1/6] and this run get the full 200-epoch budget, so whichever wins does so at a
# comparable point in each config's own trajectory, not an arbitrary earlier checkpoint.
CMD5="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --epochs 200 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_wholetooth"
script -qec "$CMD5" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_wholetooth.log

echo "[6/6] COLOR DROPOUT ON: color_dropout_prob=0.6, everything else = anchor"
# Tests the live-deployment robustness fix (dataset/patch_preprocessing_color_dropout.py, added
# 2026-08-21 after the trimesh process=False bug was found and fixed - see realtime/
# live_preprocessing.py) at its intended real-world value: --color_dropout_prob applies to BOTH
# train and val (get_datasets()'s own deliberate choice - see that function's docstring), so
# val_miou here measures something DIFFERENT from every other run in this matrix (a distribution
# that includes color-less faces, matching what real live inference actually sees) - NOT directly
# comparable to the anchor's val_miou number as an apples-to-apples "which config is better" read.
# The point of this run is to see whether training WITH dropout costs meaningful accuracy on this
# harder, more realistic validation distribution, not to rank it against the other 5 runs' easier
# always-has-color val set.
CMD6="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --color_dropout_prob 0.6 --epochs 200 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_colordropout"
script -qec "$CMD6" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_colordropout.log

echo "Matrix complete (anchor, gating-off, RGB-ablation, dilation_ks-widened, whole-tooth-on, color-dropout-on). Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."

# ---------------------------------------------------------------------------------------------
# EARLIER MATRIX (2026-08-19/20) - ALREADY RUN, kept for reproducibility, not executed here:
#   [1/3] mosaic + tuned thresholds, whole_tooth=0.0: COMPLETED 100 epochs, 0.7730 @ 84
#         (since extended to 200 epochs with the OLD non-HSV color - run_extend_run1_200ep.sh -
#         0.8317 @ 199, still climbing)
#   [2/3] same + whole_tooth=0.3: COMPLETED 100 epochs, 0.7471 @ 90
#   [3/3] same as [2/3] + dilation_ks=200/1200/3000: never run on this machine (superseded by
#         [4/6] above, which tests dilation_ks against the anchor in isolation instead)
# OLDCMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned"
# OLDCMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth"
# ---------------------------------------------------------------------------------------------
