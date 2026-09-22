#!/bin/bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
# Local (24-core/RTX 5090) training for the two Transformer-augmented architectures - both
# also scheduled on the server matrix (run_experiment_matrix_gpu1.sh's early+late run,
# run_experiment_matrix_gpu2.sh's early-only run) at 200 epochs; this script runs them here at
# 100 epochs instead, for faster turnaround (~half the wall-clock) - user-chosen tradeoff
# 2026-08-22. Experiment_versions are tagged _100ep specifically so these don't collide with or
# get confused for the server's 200-epoch runs of the same configs - they're genuinely different
# experiments now (different epoch budgets), not the same one run twice.
#
# CAVEAT carried over from the matrix's own 200-epoch reasoning, now reintroduced by shortening
# back to 100: run 1's own 200-epoch extension (run_extend_run1_200ep.sh) was still climbing at
# epoch 199, not plateaued - so any single 100-epoch number here is a snapshot mid-climb, not a
# ceiling. Fine for a fast first read on whether either Transformer variant looks promising; if
# one does, it's worth extending it further (same --ckpt .../last.ckpt --epochs 200 pattern used
# for run 1) before trusting the number as final.
#
# Same anchor config as the server matrix otherwise (run_experiment_matrix_gpu0.sh's header):
# tuned thresholds, mosaic color (HSV-fixed), whole_tooth_patch_prob=0.0, dilation_ks default,
# gating on - only --architecture/--add_late_global differ between the two runs below and against
# the matrix's own dilated-only anchor.
#
# Run 1 (early-global only) then run 2 (early+late-global) - sequential, not parallel (one GPU on
# this machine). Estimated ~14.5h and ~15.5h respectively (half the 200-epoch estimates) - roughly
# 30 hours combined.
#
# Run from the REPO ROOT, inside tmux/screen: bash training_launch/run_transformer_this_machine.sh
# If a run dies, resume rather than restart - checkpoints/<experiment_version>/last.ckpt is
# written every epoch:
#   python3 train_patch_network_color.py <same flags as that run's CMD> --ckpt <path to last.ckpt>

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[1/2] TRANSFORMER (early-global only): --architecture dilated_transformer - 100 epochs"
CMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --architecture dilated_transformer --epochs 100 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_100ep"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer_100ep.log

echo "[2/2] TRANSFORMER (early + late global): --architecture dilated_transformer --add_late_global - 100 epochs"
CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --architecture dilated_transformer --add_late_global --epochs 100 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal_100ep"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal_100ep.log

echo "Both Transformer runs complete. Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
