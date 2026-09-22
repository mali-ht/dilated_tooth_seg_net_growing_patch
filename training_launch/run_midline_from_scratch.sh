#!/usr/bin/env bash
# Same configuration as resume_transformer_training.sh, with midline sampling
# enabled from epoch 0 and random initialization (deliberately no --ckpt).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1

EXPERIMENT_VERSION="Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed_scratch_200ep_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME=patch_dilated_tooth_seg_net
LOG="logs/experiments/${EXPERIMENT_VERSION}.log"
mkdir -p logs/experiments

CMD=(python3 -u train_patch_network_color.py
    --num_classes 17 --dilation_gating on --color_style mosaic
    --area_thresholds 20 90 180 --early_bias_power 1.5
    --whole_tooth_patch_prob 0.0 --midline_seed_prob 0.3
    --midline_seed_min_mm 8 --midline_seed_max_mm 22 --midline_seed_stages 8
    --architecture dilated_transformer --epochs 200 --devices 0
    --experiment_name "$EXPERIMENT_NAME" --experiment_version "$EXPERIMENT_VERSION")

{
    echo "Starting fresh training: $EXPERIMENT_VERSION"
    echo "Checkpoints: checkpoints/$EXPERIMENT_NAME/$EXPERIMENT_VERSION/"
    echo "TensorBoard: logs/tensorboard/$EXPERIMENT_NAME/$EXPERIMENT_VERSION/"
    printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
    # Give Lightning a terminal so its progress bar remains visible through tee.
    printf -v command_line '%q ' "${CMD[@]}"
    script -qec "$command_line" /dev/null
    echo "Training completed successfully."
} 2>&1 | tee "$LOG"
