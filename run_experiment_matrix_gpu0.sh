#!/bin/bash
# ============================================================================================
# COMPREHENSIVE INDEPENDENT-ASPECT ABLATION MATRIX - SERVER (4 GPUs), lane 0/4
# Written 2026-08-22. This is lane 0 of 4 - see run_experiment_matrix_gpu1.sh/gpu2.sh/gpu3.sh
# for the other three. All 4 are meant to be launched SIMULTANEOUSLY (one per GPU, e.g. 4 tmux
# panes) - each is a fully independent sequential queue pinned to its own GPU via --devices, NOT
# a distributed/DDP job split across GPUs. Running N independent single-GPU experiments in
# parallel needs none of the DDP machinery (gradient sync, class_alpha-per-rank consistency,
# sync_dist) that this project's separate multi-GPU-SETUP task (a single model trained faster
# across multiple GPUs together) is about - that's a different concern, not a prerequisite for
# this script. All that's needed here is the server environment installed (setup_server.sh) and
# 4 GPUs to exist.
#
# GOAL (user-directed 2026-08-22): test EVERY meaningful aspect of this project's own bespoke
# architecture/pipeline design INDEPENDENTLY (one change from a single fixed ANCHOR config per
# run, not a full factorial of every combination) so that afterward the winning setting from each
# independent test can be COMBINED into one final model. Not a hyperparameter search (lr/
# optimizer/weight_decay stay untouched throughout) - specifically the axes this project has
# tuned or added over its own history without ever isolating them individually:
#   gating, RGB/color (none vs flat vs mosaic), dilation_ks, whole-tooth curriculum, color
#   dropout, early_bias_power, area_thresholds, and (both) Transformer architectures.
#
# THE ANCHOR CONFIG (every run across all 4 lanes changes exactly ONE thing away from this):
#   17-class, dilation_gating=on, color_style=mosaic (RealisticColorPaintHSV),
#   area_thresholds=20/90/180 (tuned), early_bias_power=1.5 (tuned), whole_tooth_patch_prob=0.0,
#   dilation_ks=200/900/1800 (paper default), color_dropout_prob=0.0, architecture=dilated
#   (no transformer), 200 epochs. Lane 3's first run below IS this anchor, unchanged - see that
#   file.
#
# WHY 200 EPOCHS: run [1]'s own 200-epoch extension (this repo's run_extend_run1_200ep.sh, on the
# 24-core/RTX-5090 dev machine) showed val_miou still climbing at epoch 199, not plateaued - a
# shorter budget risks the same "which config would've won by 200?" ambiguity that made an
# earlier 100-epoch whole-tooth comparison inconclusive against its own anchor.
#
# ALL 11 RUNS ACROSS ALL 4 LANES (for reference - only this lane's 2 are launched from here):
#   [gpu3] anchor (HSV color validation), gating off, RGB=none
#   [gpu2] Transformer (early-global), RGB=flat, early_bias_power=default(2.5)
#   [gpu1] Transformer (early+late-global), whole-tooth on, area_thresholds=default(40/180/360)
#   [gpu0] dilation_ks widened, color dropout on           <- THIS FILE
# Lanes are balanced by ESTIMATED cost (not run count) using per-epoch timings measured on the
# 24-core/RTX-5090 DEV machine (NOT this server - the server has more VRAM/FLOPs per the user, so
# real wall-clock here should be faster across the board; relative proportions between runs -
# e.g. dilation_ks widening costing ~33% more - are what's carried over, not absolute hours).
# Estimated lane totals (dev-machine-hours, for relative balance only): gpu0=~60h (heaviest single
# run), gpu1=~79h, gpu2=~78h, gpu3=~78h - reasonably even, expect this lane to also be the FASTEST
# to finish since dilation_ks widening's cost is concentrated in this lane alone.
#
# Run from the REPO ROOT, inside tmux/screen (a dropped session over multiple days will otherwise
# kill it): bash run_experiment_matrix_gpu0.sh
# If a run dies, resume rather than restart - checkpoints/<experiment_version>/last.ckpt is
# written every epoch: python3 train_patch_network_color.py --num_workers 20 <same flags> --ckpt <path>
# ============================================================================================

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[gpu0 1/2] DILATION_KS WIDENED: 200/1200/3000 (was 200/900/1800), everything else = anchor"
# Re-tests the "wider local context helps distinguish arch-position-confused teeth (4 vs 5, 12 vs
# 13)" hypothesis from Docs/REALTIME.md's premolar-confusion finding - block 1 unchanged
# (near-field, not the block the arch-position hypothesis is about), block 2 900->1200 (+33%),
# block 3 1800->3000 (+67%, the block already active at the >4000mm^2 area regime where
# REALTIME.md's confusion persists even with full dilated context). See
# models/patch_dilated_tooth_seg_network.py's dilation_ks docstring for why 200/900/1800 are the
# paper's own values, never re-derived for this project's patch distribution. Heaviest single run
# in the whole matrix (~33% more expensive per batch, measured on the dev machine) - given its
# own lane here rather than paired with another heavy run.
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

CMD1="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --dilation_ks 200 1200 3000 --epochs 200 --devices 0 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_dilationks"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_dilationks.log

echo "[gpu0 2/2] COLOR DROPOUT ON: color_dropout_prob=0.6, everything else = anchor"
# Tests the live-deployment robustness fix (dataset/patch_preprocessing_color_dropout.py, added
# 2026-08-21 after the trimesh process=False color-loss bug was found and fixed - see
# realtime/live_preprocessing.py) at its real-world value. --color_dropout_prob applies to BOTH
# train and val (get_datasets()'s own deliberate choice), so val_miou here measures a DIFFERENT,
# harder distribution than every other run in this matrix (matching what live inference actually
# sees) - NOT directly comparable to the anchor's val_miou as an apples-to-apples ranking. The
# point is whether training WITH dropout costs meaningful accuracy on this harder distribution,
# not to rank it against the other 10 runs' easier always-has-color val set.
CMD2="python3 train_patch_network_color.py --num_workers 20 --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --color_dropout_prob 0.6 --epochs 200 --devices 0 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_colordropout"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_colordropout.log

echo "[gpu0] lane complete. Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
