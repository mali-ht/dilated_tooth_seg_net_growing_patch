#!/bin/bash
# Lane 3/4 of the comprehensive independent-aspect ablation matrix - see
# run_experiment_matrix_gpu0.sh for the full shared header (goal, anchor config, why 200 epochs,
# all 11 runs across all 4 lanes, DDP-not-needed note, cost-balancing caveat). Launch all 4
# gpu0..gpu3 scripts SIMULTANEOUSLY (one per GPU, e.g. 4 tmux panes) for real parallelization -
# each is fully independent, pinned to its own GPU via --devices.
#
# This lane: the ANCHOR itself, gating off, RGB=none.
# Estimated total (dev-machine-relative): ~78h.
#
# Run from the REPO ROOT, inside tmux/screen: bash run_experiment_matrix_gpu3.sh
# If a run dies, resume via --ckpt <path>/last.ckpt rather than restarting.

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[gpu3 1/3] ANCHOR: tuned thresholds, mosaic color (HSV-fixed), whole_tooth=0.0, dilation_ks default, gating on - 200 epochs"
# Every other run across all 4 lanes is measured against this one. Also the real validation of
# the HSV color-augmentation fix in its own right, not just a placeholder baseline: the mosaic
# color style changed from RealisticColorPaint (independent-per-channel RGB noise, confirmed to
# occasionally produce implausible saturated colors like purple/magenta "enamel") to
# RealisticColorPaintHSV (correlated hue/sat/val noise) AFTER the existing
# Patch_Seg_17class_gateOn_color_mosaic_tuned checkpoint (0.8317 @ epoch 199) was trained - see
# dataset/patch_preprocessing_realistic_color_hsv.py's docstring. That old checkpoint used the
# OLD color scheme and is deliberately left alone (this run uses a distinct _hsv-suffixed
# experiment_version) so it stays available as a direct before/after comparison for the color fix
# itself, independent of whatever this matrix finds.
# --------------------------------------------------------------------------------------------
# --num_workers 20 (added 2026-08-22, server deployment): the script's own default of 8 was tuned
# on the 24-core dev machine, alongside the NUMBA_NUM_THREADS=1 fix documented at the top of
# train_patch_network_color.py ("concurrency should come from num_workers"). This server has 128
# cores. Measured on the first launch attempt with the default 8: 4 lanes x 8 train workers = 32
# active worker processes, GPUs averaging 4-17% utilisation, 58% of CPU idle and %wa = 0.0 (so
# starved by the collate pipeline, not by disk). 20 per lane puts 80 of 128 cores to work while
# leaving headroom for the 4 main processes and for each lane's persistent val worker pool waking
# up during validation. Raise further only with the same measurement in hand - the header comment
# in train_patch_network_color.py documents what over-subscription looked like when it went wrong.
# --------------------------------------------------------------------------------------------

CMD1="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 3 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv.log

echo "[gpu3 2/3] GATING OFF: dilation_gating=off, everything else = anchor"
# Tests whether area-gating the 3 dilated blocks (skip a block on patches too small to have
# reached area_thresholds[i] - models/patch_dilated_tooth_seg_network.py's _resolve_gates) is
# actually earning its keep, or just adding inference-time complexity for no real accuracy gain.
# dilation_gating=off means all 3 dilated blocks always run regardless of patch area.
CMD2="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating off --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 3 --experiment_version Patch_Seg_17class_gateOff_color_mosaic_tuned_hsv"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOff_color_mosaic_tuned_hsv.log

echo "[gpu3 3/3] RGB ABLATION: color_style=none (feature_dim=24, no color channel at all), everything else = anchor"
# Tests RGB's actual contribution to accuracy, isolated from every other axis this project has
# tuned around color (thresholds/whole-tooth/etc. stay at their tuned values - only the color
# CHANNEL is removed). Uses --color_style none (dataset/patch_preprocessing.py's plain
# PatchPreTransform, added 2026-08-22 specifically for this - the original pre-color script,
# train_patch_network.py, was NOT used since it never received any of the tuned-threshold/
# whole-tooth/dilation_ks flags added to train_patch_network_color.py over this project's
# history, so it can't run an apples-to-apples comparison against the anchor's config).
CMD3="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style none --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 3 --experiment_version Patch_Seg_17class_gateOn_colorNone_tuned"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_colorNone_tuned.log

echo "[gpu3] lane complete. Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
