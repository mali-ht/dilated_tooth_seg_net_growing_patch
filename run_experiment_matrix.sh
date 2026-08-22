#!/bin/bash
# Run [3/3] ONLY. Runs [1/3] and [2/3] were executed on the 24-core/RTX-5090 machine and their
# checkpoints+logs copied here, so this script no longer re-runs them (2026-08-20). Their commands
# are preserved commented-out at the bottom for reproducibility.
#
# RESULTS OF THE FIRST TWO RUNS, as copied into this repo - read before trusting [3/3]'s premise:
#   [1/3] mosaic + tuned thresholds, whole_tooth_patch_prob=0.0
#         COMPLETED all 100 epochs. Best val_miou=0.7730 @ epoch 84.
#         (checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned/)
#   [2/3] same + whole_tooth_patch_prob=0.3
#         *** DID NOT COMPLETE *** - its log stops dead mid-epoch 56/99 at batch 591/1200 with no
#         traceback, no CUDA error and no Lightning shutdown line (i.e. the process was killed
#         abruptly - SIGKILL/OOM-killer/host died - NOT the torch.AcceleratorError launch-timeout
#         that killed the earlier 17class_gateOn_color_tuned run at a coincidentally similar
#         epoch). Best val_miou=0.6695 @ epoch 54.
#         (checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth/)
#
# *** CAVEAT ON RUNNING [3/3] NOW ***
# [3/3]'s original design rule was "layer dilation_ks on top of [2/3] ONCE THE FIRST TWO RUNS HAVE
# SHOWN the tuned-threshold+whole-tooth combination is stable". That precondition is NOT met:
# [2/3] never finished, and at the epoch it died its val_miou (0.6695 @ 54) was still well BELOW
# [1/3]'s (0.7730 @ 84) - though that comparison is not conclusive either, since [1/3] needed until
# epoch 84 to reach its best, and [2/3] never got past 56. So whole_tooth_patch_prob=0.3 is
# currently UNPROVEN, not proven-good - and [3/3] below inherits it. If [3/3] underperforms [1/3],
# the cause is ambiguous between the whole-tooth curriculum and dilation_ks. Finishing [2/3] first
# (resume from its last.ckpt, see below) would disambiguate; running [3/3] now trades that
# cleanliness for wall-clock. Deliberate choice - just don't read a bad [3/3] as "dilation_ks bad".
#
# *** RUNTIME ON THIS MACHINE (12-core / RTX 3090) - MUCH SLOWER THAN THE 5090 BOX ***
# Measured directly with this exact config, 2026-08-20: ~1.0-1.4 s per train batch steady-state
# (vs the ~0.25 s/batch in [1/3]'s log from the other machine). Projected ~35-40 min/epoch
# (1200 train + 600 val batches) => ROUGHLY 2.5-3 DAYS for the full 100 epochs, not the ~5 hrs the
# old header claimed for the 5090 box. The bottleneck is CPU, not the GPU:
#   - GPU utilization averaged only ~21% (peak VRAM 7.0 GB of 24 GB - memory is NOT a constraint)
#   - load average 12-21 on 12 cores, i.e. saturated/oversubscribed
#   - this machine has nproc=12; the machine the header's timings came from has nproc=24
# num_workers was measured at 5 vs 8 and made no meaningful difference (1.40 vs 1.33 s/batch, within
# run-to-run noise), so it is left at its default 8. Note persistent_workers=True keeps BOTH the
# train and val pools alive, so num_workers=8 means 16 live worker processes on 12 cores.
# Widening dilation_ks is itself ~33% of the per-batch cost (0.75 s/batch at 200/900/1800 vs
# 1.0 s/batch at 200/1200/3000, same machine) - the rest of the gap is just the smaller box.
#
# Run from the REPO ROOT, inside tmux/screen (a dropped SSH/desktop session over ~3 days will
# otherwise kill it - which is the most likely explanation for how [2/3] died):
#   bash run_experiment_matrix.sh
#
# If this run dies, resume it rather than restarting - checkpoints/<experiment_version>/last.ckpt
# is written every epoch:
#   python3 train_patch_network_color.py <same flags as CMD3 below> --ckpt <path to last.ckpt>

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

# Requires the `dtsegnet` conda env (python 3.10 / torch 2.1.0+cu121 / CUDA 12.1 / numba):
#   conda activate dtsegnet
# Fail early and loudly rather than 3 days later, or with the system python:
python3 -c "import torch, numba, lightning; assert torch.cuda.is_available(), 'CUDA not available'" \
  || { echo "ERROR: wrong/incomplete python env - run 'conda activate dtsegnet' first." >&2; exit 1; }

CKPT_ROOT=checkpoints/patch_dilated_tooth_seg_net
mkdir -p logs/experiments

echo "[3/3] 17-class, dilation_gating=on, MOSAIC color, area_thresholds=20/90/180 (tuned), early_bias_power=1.5 (tuned), whole_tooth_patch_prob=0.3, dilation_ks=200/1200/3000 (tuned - global-context test)"
# Same config as [2/3] plus dilation_ks tuning - the axis deliberately deferred from [1/3]/[2/3] to
# avoid stacking a 3rd untested change on a config with a known crash history. See the CAVEAT above:
# that deferral condition was never actually satisfied, since [2/3] did not finish.
# Tests the "more global context could help distinguish tooth 4 from 5 by relative arch position"
# hypothesis directly - motivated by Docs/REALTIME.md's own premolar-confusion finding (Finding,
# ~line 855: labels 4/5 and 12/13 repeatedly swap dominance even at area>4000mm^2, where ALL THREE
# area_thresholds gates are already open - so the confusion persists even with the full dilated
# context CURRENTLY available, meaning the fix (if there is one on this axis) has to come from
# WIDENING that context, not just unlocking it earlier).
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
# NB: that CPU KD-tree cost is exactly what the measured 0.75 -> 1.0 s/batch above is.
CMD3="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --dilation_ks 200 1200 3000 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth_dilationks"
script -qec "$CMD3" /dev/null 2>&1 | tee logs/experiments/17class_gateOn_color_mosaic_tuned_wholetooth_dilationks.log

echo "Run complete (mosaic + tuned thresholds + whole-tooth curriculum + dilation_ks tuning). Checkpoint under $CKPT_ROOT/Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth_dilationks/, log under logs/experiments/."

# ---------------------------------------------------------------------------------------------
# ALREADY RUN ON THE OTHER MACHINE - kept for reproducibility, intentionally not executed here.
# To finish the incomplete [2/3] (recommended before drawing conclusions from [3/3]), append
#   --ckpt checkpoints/patch_dilated_tooth_seg_net/Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth/last.ckpt
# to CMD2 and run it.
#
# CMD1="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned"
# CMD2="python3 train_patch_network_color.py --num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.3 --experiment_version Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth"
# ---------------------------------------------------------------------------------------------
