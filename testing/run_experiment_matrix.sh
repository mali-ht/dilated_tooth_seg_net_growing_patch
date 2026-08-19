#!/bin/bash
# Runs the 4-combination experiment matrix from claude_code_prompt.md Component 5
# (num_classes in {17,5} x dilation_gating in {on,off}), preceded by two synthetic-color runs -
# default area_thresholds/early_bias_power vs. the tuned values - back to back on one GPU.
# The third promised axis, norm scheme, isn't a runnable CLI flag yet - GroupNorm is
# the settled design (see models/patch_lightning_module.py's own comment) - so the 4-run
# matrix covers the 2 axes that are actually wired up today.
#
# Run from the REPO ROOT, not from inside testing/ (this file moved there 2026-08-19, but its
# own python3 calls below are plain `python3 train_patch_network_color.py ...` - resolved
# relative to the CALLER's cwd, not this script's own location, so it must be invoked as:
#   bash testing/run_experiment_matrix.sh
# Recommended inside tmux/screen so it survives a disconnected terminal - see REALTIME.md-
# style session notes for why: each of the 4 matrix runs is ~4.6hrs on an RTX 5090
# (~18-19hrs total); the color run adds roughly the same again on top of that.

set -e          # stop the queue on the first failure, don't burn hours on a broken config
set -o pipefail # WITHOUT this, `python3 ... | tee ...`'s exit status is tee's (always 0), so
                # set -e alone never sees a crashed training run - confirmed the hard way
                # 2026-08-19: run [1/6] crashed with a CUDA error partway through and the script
                # silently moved on to [2/6] anyway instead of stopping.
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

echo "[1/6] 17-class, dilation_gating=on, synthetic color, area_thresholds=40/180/360 (default), early_bias_power=2.5 (default)"
# Baseline color run - same area_thresholds/early_bias_power as PatchDilatedToothSegmentationNetwork
# / PatchTeeth3DSDataset's own defaults (omitted here, so the flags fall back to them). Run back to
# back with [2/6] below (the tuned-threshold variant) so the two are directly comparable - same
# data, same seed, same everything except this one axis - rather than comparing against the older
# Patch_Seg_081226 checkpoint (no color, different dilation_ks too).
CMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --experiment_version Patch_Seg_17class_gateOn_color_baseline"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_baseline.log

echo "[2/6] 17-class, dilation_gating=on, synthetic color, area_thresholds=20/90/180, early_bias_power=1.5 (tuned)"
# Real hardware finding (cycle_0001, realtime/logs/snapshots/mesh_viewer_segmented_20260815_122810/)
# showed the model predicting 5 different classes on a 52.6mm2 (~2-tooth) patch, where only the
# first dilated block was gated on and its own candidate pool was already close to the whole patch
# - see Docs/REALTIME.md. NOTE: sweep_area_thresholds.py's no-retrain val-set sweep against the
# OLD (40/180/360-trained, no-color) Patch_Seg_081226 checkpoint actually showed these lower values
# performing WORSE at low area (0-100mm2: acc 0.745->0.698, mIoU 0.592->0.542; 100-400mm2: acc
# 0.772->0.742, mIoU 0.549->0.512 - see logs/experiments/threshold_sweep_081226.csv) - expected,
# since that checkpoint's dilated blocks 2/3 were never calibrated for the wider activation regime
# these thresholds newly open up. That result does NOT validate these values - it can't, on old
# weights - so this run is the actual test: compare its val_miou against [1/6]'s baseline once both
# finish to judge whether tuning these before training helps.
CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --area_thresholds 20 90 180 --early_bias_power 1.5 --experiment_version Patch_Seg_17class_gateOn_color_tuned"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_tuned.log

# Rest of the matrix - commented out for tonight's run (RGB/gateOn only, per instruction). Re-enable
# by uncommenting when ready to run the full comparison matrix again. NOTE: these use
# train_patch_network.py, not the _color variant - that file does NOT yet have the
# numba/torch thread-oversubscription fix from train_patch_network_color.py's header (see that
# file's own comment for the trace); worth porting over before running these for real.
# echo "[3/6] 17-class, dilation_gating=on"
# CMD3="python3 train_patch_network.py --num_classes 17 --dilation_gating on --experiment_version Patch_Seg_17class_gateOn"
# script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn.log
#
# echo "[4/6] 17-class, dilation_gating=off"
# CMD4="python3 train_patch_network.py --num_classes 17 --dilation_gating off --experiment_version Patch_Seg_17class_gateOff"
# script -qec "$CMD4" /dev/null 2>&1 | tee logs/experiments/17class_gateOff.log
#
# echo "[5/6] 5-class, dilation_gating=on"
# CMD5="python3 train_patch_network.py --num_classes 5 --dilation_gating on --experiment_version Patch_Seg_5class_gateOn"
# script -qec "$CMD5" /dev/null 2>&1 | tee logs/experiments/5class_gateOn.log
#
# echo "[6/6] 5-class, dilation_gating=off"
# CMD6="python3 train_patch_network.py --num_classes 5 --dilation_gating off --experiment_version Patch_Seg_5class_gateOff"
# script -qec "$CMD6" /dev/null 2>&1 | tee logs/experiments/5class_gateOff.log

echo "Run complete (color baseline + tuned-threshold tonight). Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
