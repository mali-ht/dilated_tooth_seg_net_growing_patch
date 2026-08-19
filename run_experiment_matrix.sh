#!/bin/bash
# Tonight's (2026-08-19) 3-run plan, replacing the earlier 2-run color matrix (both of those
# already completed/crashed - see logs/experiments/17class_gateOn_color_{baseline,tuned}.log and
# the finding below).
#
# Context: the earlier "tuned area_thresholds" run (20/90/180, flat color) CRASHED at epoch
# 56/99 with a CUDA launch timeout (torch.AcceleratorError: "the launch timed out and was
# terminated" - a GPU driver watchdog kill, same failure class --max_faces already guards
# against, apparently not sufficient here) - see logs/experiments/17class_gateOn_color_tuned.log.
# BUT its last checkpoint (epoch 44) had already reached val_miou=0.7045, clearly ahead of the
# completed baseline's best (0.6505, epoch 93) - see checkpoints/patch_dilated_tooth_seg_net/. So
# the tuned-threshold direction looks real but is UNCONFIRMED (never finished, never run with
# mosaic color). [1/3] and [2/3] below keep --dilation_ks at its default (200 900 1800) - only
# [3/3] adds that as a 3rd axis, layered on top of [2/3]'s already-tuned config, once the first
# two runs have shown the tuned-threshold+whole-tooth combination is stable. If a run below dies
# the same way, resume it directly rather than restarting from scratch -
# checkpoints/<experiment_version>/last.ckpt is saved every epoch:
#   python3 train_patch_network_color.py <same flags as the crashed CMD> --ckpt <path to last.ckpt>
#
# Run from the REPO ROOT:
#   bash run_experiment_matrix.sh
# Recommended inside tmux/screen so it survives a disconnected terminal. Each run is
# ~4.5-5.5hrs on an RTX 5090; ~15-16hrs combined - plan for a full overnight-plus window.

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

echo "[1/3] 17-class, dilation_gating=on, MOSAIC color, area_thresholds=20/90/180 (tuned), early_bias_power=1.5 (tuned), whole_tooth_patch_prob=0.0"
# Clean re-confirmation of the promising-but-crashed/flat-color tuned-threshold result, now with
# mosaic color (RealisticColorPaint - per-face draws from the measured real scanner distribution,
# dataset/patch_preprocessing_realistic_color.py) instead of the flat synthetic color the earlier
# crashed run used, and run to full completion. whole_tooth_patch_prob left at 0.0 (pure
# scan-sweep, same distribution the earlier baseline/tuned runs trained on) so this result is
# directly comparable to both of them - only the color_style + area_thresholds/early_bias_power
# axes differ from baseline.
CMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned"
script -qec "$CMD1" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned.log

echo "[2/3] 17-class, dilation_gating=on, MOSAIC color, area_thresholds=20/90/180 (tuned), early_bias_power=1.5 (tuned), whole_tooth_patch_prob=0.3 (whole-tooth curriculum)"
# Same config as [1/2] plus the whole-tooth/multi-tooth curriculum (dataset/patch_generator_whole_tooth.py,
# dataset/patch_dataset_whole_tooth.py - additive, dataset/patch_dataset.py itself untouched):
# whole_tooth_patch_prob=0.3 of TRAIN draws use ground-truth-label-bounded, 1.5mm-margin-dilated,
# one-whole-tooth-at-a-time growth instead of the scan-sweep local footprint, so the model also
# sees complete, cleanly-bounded tooth shapes during training - not just growing local patches.
# val stays pinned to whole_tooth_patch_prob=0.0 regardless (see train_patch_network_color.py's
# get_datasets() comment) so val_miou here is directly comparable against [1/2]'s: same real
# scan-sweep distribution, only the TRAINING data mix differs. This isolates whole-tooth's own
# contribution on top of [1/2]'s already-tuned thresholds + mosaic color, rather than changing
# multiple things against the old baseline at once.
CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth"
script -qec "$CMD2" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_wholetooth.log

echo "[3/3] 17-class, dilation_gating=on, MOSAIC color, area_thresholds=20/90/180 (tuned), early_bias_power=1.5 (tuned), whole_tooth_patch_prob=0.3, dilation_ks=200/1200/3000 (tuned - global-context test)"
# Same config as [2/3] plus dilation_ks tuning - the axis deliberately deferred from [1/3]/[2/3] to
# avoid stacking a 3rd untested change on a config with a known crash history; now layered on top
# once [1/3]/[2/3] have shown that combination is stable. Tests the "more global context could
# help distinguish tooth 4 from 5 by relative arch position" hypothesis directly - motivated by
# Docs/REALTIME.md's own premolar-confusion finding (Finding, ~line 855: labels 4/5 and 12/13
# repeatedly swap dominance even at area>4000mm^2, where ALL THREE area_thresholds gates are
# already open - so the confusion persists even with the full dilated context CURRENTLY available,
# meaning the fix (if there is one on this axis) has to come from WIDENING that context, not just
# unlocking it earlier).
#
# 200/900/1800 are the ORIGINAL PAPER's values (models/dilated_tooth_seg_network.py, untouched),
# tuned for a fixed 16,000-face full-arch mesh (Docs/TRAINING_CONCERNS.md) - not something this
# project ever re-derived for our own patch size distribution (a few hundred faces up to
# --max_faces=20000). dilation_k is the CANDIDATE-neighborhood size each dilated block gathers
# (via KD-tree, in true metric/mm position) before FPS-downsampling to its actual k=32 graph, i.e.
# it controls how far in mm a block reaches, not what fraction of the patch it covers - so this
# widens absolute spatial reach regardless of patch size.
#   - Block 1 (dilation_ks[0]=200): left UNCHANGED - this is the near-field block, not the one the
#     "arch position" hypothesis is about.
#   - Block 2 (dilation_ks[1]=900->1200, +33%): moderate widening.
#   - Block 3 (dilation_ks[2]=1800->3000, +67%): the block most relevant to the hypothesis gets the
#     largest increase, since it's already the one active at the exact area regime (>4000mm^2)
#     where REALTIME.md's finding shows confusion persisting today.
# Deliberately NOT doubling everything - larger dilation_ks means a larger CPU KD-tree
# gather+FPS per activated block (models/patch_collate.py), stacked on top of [2/3]'s
# already-once-crashed-at-these-thresholds config and whole-tooth's own occasionally-larger
# patches - moderate values first, not a shot in the dark maximized for reach at max crash risk.
CMD3="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --dilation_ks 200 1200 3000 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth_dilationks"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_wholetooth_dilationks.log

echo "Run complete (mosaic+tuned-thresholds confirmation, then + whole-tooth curriculum, then + dilation_ks tuning). Checkpoints under $CKPT_ROOT/<experiment_version>/, logs under logs/experiments/."
