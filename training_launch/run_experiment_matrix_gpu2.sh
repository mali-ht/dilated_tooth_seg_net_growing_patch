#!/bin/bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
# Lane 2/4 of the comprehensive independent-aspect ablation matrix - see
# run_experiment_matrix_gpu0.sh for the full shared header (goal, anchor config, why 200 epochs,
# all 11 runs across all 4 lanes, DDP-not-needed note, cost-balancing caveat). Launch all 4
# gpu0..gpu3 scripts SIMULTANEOUSLY (one per GPU, e.g. 4 tmux panes) for real parallelization -
# each is fully independent, pinned to its own GPU via --devices.
#
# This lane: Transformer (early-global only), RGB=flat, early_bias_power=default.
# Estimated total (dev-machine-relative): ~78h.
#
# Run from the REPO ROOT, inside tmux/screen: bash training_launch/run_experiment_matrix_gpu2.sh
# If a run dies, resume via --ckpt <path>/last.ckpt rather than restarting.

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[gpu2 1/3] TRANSFORMER (early global only): --architecture dilated_transformer, everything else = anchor"
# Tests GlobalTokenTransformerBlock (models/patch_transformer_layer.py) run once, in parallel
# with the dilated blocks, on raw local features (models/patch_dilated_tooth_seg_transformer_
# network.py) - true all-pairs global context, something no existing block in the original
# architecture provides (every other block is either a fixed-k-nearest-neighbor gather or a
# per-point-independent channel gate). Paired against [gpu1]'s early+late variant to isolate
# whether the SECOND (late) attention pass adds anything beyond this one.
CMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --architecture dilated_transformer --epochs 200 --devices 2 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer.log

echo "[gpu2 2/3] RGB STYLE: flat (SyntheticColorPaint), everything else = anchor"
# 'flat' has never been tested against the TUNED thresholds/early_bias_power - only against the
# old untuned baseline, long before this matrix existed. One shared RGB draw per class-group per
# patch (dataset/patch_color_augmentation.py's SyntheticColorPaint, hand-picked TOOTH_RGB_LOW/
# HIGH and GUM_RGB_LOW/HIGH ranges), vs 'mosaic' HSV's per-face correlated draw. Completes the
# 3-way RGB-style comparison alongside the anchor (mosaic) and [gpu3]'s none.
CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style flat --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 2 --experiment_version Patch_Seg_17class_gateOn_colorFlat_tuned"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_colorFlat_tuned.log

echo "[gpu2 3/3] EARLY_BIAS_POWER AT DEFAULT: 2.5 (was 1.5 tuned), area_thresholds STAYS TUNED (20/90/180)"
# Complementary isolation to [gpu1]'s area_thresholds-default run - reverts ONLY
# early_bias_power to PatchTeeth3DSDataset's original default (more skew toward small/early
# growth stages) while keeping area_thresholds at its tuned value. Together, [gpu1]'s and this
# run bracket which of the two originally-co-tuned knobs actually mattered.
CMD3="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 2.5 --whole_tooth_patch_prob 0.0 --epochs 200 --devices 2 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_areaThreshTunedOnly"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_areaThreshTunedOnly.log

echo "[gpu2] lane complete. Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
