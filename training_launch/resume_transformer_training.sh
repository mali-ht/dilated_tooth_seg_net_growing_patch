#!/bin/bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
# ONE run tonight: resumes the early-global-ONLY Transformer's 100-epoch checkpoint
# (Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_100ep/last.ckpt) up to 200 epochs,
# with the midline-seeded curriculum (dataset/patch_generator_midline.py,
# dataset/patch_dataset_midline.py) now turned on for the rest of training - NOT a fresh run.
#
# Early-only, not early+late (changed 2026-08-24): the original choice of early+late was based
# only on the LOCAL 100-epoch numbers (its last-20-epoch average sat closer to its own peak). The
# server matrix's full 200-epoch run reversed that - early-only finished at 0.8472 vs early+late's
# 0.8400, the best result in the whole matrix - so the midline curriculum gets stacked on the
# actually-stronger architecture.
#
# --experiment_version is a NEW name (not the original _100ep one) since adding the midline
# curriculum partway through makes this a genuinely different experiment from that point on;
# Lightning restores the model/optimizer/epoch-count/best_model_score state from --ckpt regardless
# of --experiment_version matching (checkpoint_dir/TensorBoard version are derived purely from
# --experiment_version, independent of --ckpt - see train_patch_network_color.py), so resuming
# under a different name works cleanly, it just starts a fresh checkpoint dir/TensorBoard run
# rather than appending to the original _100ep one.
#
# --whole_tooth_patch_prob 0.0 stays explicit since it defaults to 0.3 and is mutually exclusive
# with --midline_seed_prob > 0 (train_patch_network_color.py's own guard).
#
# Run from the REPO ROOT, inside tmux/screen: bash training_launch/resume_transformer_training.sh

set -e
set -o pipefail

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

SOURCE_EXPERIMENT_VERSION=Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_100ep
EXPERIMENT_VERSION=Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed

echo "Resuming $SOURCE_EXPERIMENT_VERSION (100ep) -> $EXPERIMENT_VERSION: 100 -> 200 epochs, midline-seeded curriculum added"
CMD="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --midline_seed_prob 0.3 --architecture dilated_transformer --epochs 200 --ckpt $CKPT_ROOT/$SOURCE_EXPERIMENT_VERSION/last.ckpt --experiment_version $EXPERIMENT_VERSION"
script -qec "$CMD" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed.log

echo "Done. Checkpoint under $CKPT_ROOT/$EXPERIMENT_VERSION/, log at logs/experiments/17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed.log"
