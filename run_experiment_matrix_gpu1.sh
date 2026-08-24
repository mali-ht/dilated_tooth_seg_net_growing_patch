#!/bin/bash
# Lane 1/4 of the comprehensive independent-aspect ablation matrix - see
# run_experiment_matrix_gpu0.sh for the full shared header (goal, anchor config, why 200 epochs,
# all 11 runs across all 4 lanes, DDP-not-needed note, cost-balancing caveat). Launch all 4
# gpu0..gpu3 scripts SIMULTANEOUSLY (one per GPU, e.g. 4 tmux panes) for real parallelization -
# each is fully independent, pinned to its own GPU via --devices.
#
# This lane: Transformer (early+late-global), whole-tooth on, area_thresholds=default.
# Estimated total (dev-machine-relative): ~79h - heaviest lane (deep-transformer + whole-tooth +
# area_thresholds-default all land here), expect this to be the slowest lane to finish.
#
# Run from the REPO ROOT, inside tmux/screen: bash run_experiment_matrix_gpu1.sh
# If a run dies, resume via --ckpt <path>/last.ckpt rather than restarting.

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[gpu1 1/3] TRANSFORMER (early + late global): --architecture dilated_transformer --add_late_global, everything else = anchor"
# Tests whether either global-attention addition (models/patch_transformer_layer.py's
# GlobalTokenTransformerBlock, run once on raw local features and again on the fused
# local+dilated representation - see models/patch_dilated_tooth_seg_transformer_deep_network.py's
# own module docstring for the full design) beats the dilated-blocks-only baseline. This is the
# "most global understanding" variant - both injection points active at once, not just one.
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

CMD1="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --architecture dilated_transformer --add_late_global --epochs 200 --devices 1 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal.log

echo "[gpu1 2/3] WHOLE-TOOTH CURRICULUM ON: whole_tooth_patch_prob=0.3, everything else = anchor"
# Re-test of the same comparison an earlier 100-epoch run made (0.7471 @ 90, below the
# no-whole-tooth anchor's old 0.7730 @ 84) - inconclusive then since neither run had reached its
# real ceiling (the anchor's own 200-epoch extension kept climbing well past epoch 100). Both the
# anchor [gpu3] and this run now get the full 200-epoch budget, so whichever wins does so at a
# comparable point in each config's own trajectory.
CMD2="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --epochs 200 --devices 1 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_wholetooth"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_wholetooth.log

echo "[gpu1 3/3] AREA_THRESHOLDS AT DEFAULT: 40/180/360 (was 20/90/180 tuned), early_bias_power STAYS TUNED (1.5)"
# The original "tuned" config changed area_thresholds AND early_bias_power together - nobody has
# isolated which one actually did the work. This run reverts ONLY area_thresholds to the paper's
# original default while keeping early_bias_power at its tuned value - if this scores close to
# the anchor, early_bias_power was doing most of the lifting; if it drops close to the old
# untuned baseline (0.6505), area_thresholds tuning was the real driver. See
# run_experiment_matrix_gpu2.sh's early_bias_power-default run for the complementary isolation.
CMD3="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 40 180 360 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 1 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_earlyBiasTunedOnly"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_earlyBiasTunedOnly.log

echo "[gpu1] lane complete. Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
