#!/bin/bash
# Standalone script, deliberately separate from run_experiment_matrix.sh (that file is currently
# mid-execution in another terminal on run 2 - editing a live-running bash script risks bash's
# read-buffer desyncing when it reaches the not-yet-executed lines, so this is its own file rather
# than a change to CMD3 there). Once run 2 (and whatever's left of the original CMD3 there) is
# fully done, run_experiment_matrix.sh's CMD3 should be updated to match this and this file
# retired - see run_experiment_matrix.sh's own header for that history.
#
# REPLACES the originally-planned run 3 (dilation_ks tuning). Decided 2026-08-20: instead of
# testing a 3rd new axis, extend run 1 (Patch_Seg_17class_gateOn_color_mosaic_tuned) from its
# completed 100 epochs out to 200, to test whether more training alone helps - run 1's val_miou
# hadn't clearly plateaued (best 0.7730 was at epoch 84/100, not epoch ~50), so this is a cheap
# way to check "does longer help" before building any exhaustive-coverage dataset changes.
#
# RESUMES via --ckpt rather than retraining from scratch: run 1's checkpoint already represents
# ~13hrs of compute at epochs 0-99. Same --experiment_version as run 1 on purpose - Lightning
# restores full trainer + ModelCheckpoint state (including current best_model_score) from
# last.ckpt, so this keeps writing into run 1's existing checkpoint dir/TensorBoard run and will
# only replace best-epoch=84-val_miou=0.7730.ckpt if a later epoch actually beats it.
#
# Run from the REPO ROOT:
#   bash run_extend_run1_200ep.sh

set -e
set -o pipefail

mkdir -p logs/experiments

echo "Extending Patch_Seg_17class_gateOn_color_mosaic_tuned from epoch 100 -> 200 (resumed from last.ckpt)"
CMD="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --ckpt checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned/last.ckpt --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned"
script -qec "$CMD" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_200ep.log

echo "Done. Checkpoints under checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned/, log at logs/experiments/17class_gateOn_color_mosaic_tuned_200ep.log."
