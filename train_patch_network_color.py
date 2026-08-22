import argparse
import os

# MUST run before numpy/torch/scipy get imported anywhere (including transitively, via
# lightning) - these libraries size their internal BLAS/OpenMP thread pool from these env vars
# at import/first-use time, not something that can be changed reliably afterward. Root-caused
# 2026-08-19: a resumed training run was running far slower than the original pre-crash run, not
# hung - `ps`/`/proc/<pid>/status` showed each of the 8 (x2 pools: train+val, both
# persistent_workers) DataLoader worker PROCESSES ALSO carrying ~26 of its own OS threads (the
# main process had 47), all with no thread-count limit set anywhere in this codebase - so every
# worker was independently trying to use up to nproc=24 threads for numpy/torch CPU ops, on top
# of the num_workers=8 process-level parallelism already providing the real concurrency. Result:
# load average 40-56 on a 24-core machine, GPU sitting mostly idle waiting on a CPU pipeline that
# was thrashing on context switches instead of doing useful work. This was very likely ALSO true
# during the original overnight run - it just didn't matter as much with the desktop otherwise
# idle overnight vs. competing with interactive daytime use for the same cores. Standard fix:
# cap each worker to 1 internal thread - concurrency should come from num_workers (an existing,
# already-tuned CLI knob), not from each worker also parallelizing internally.
for _env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_env_var, '1')
# NUMBA_NUM_THREADS is the one that actually mattered here, confirmed empirically: models/patch_collate.py
# (untouched) has two @numba.njit(parallel=True) functions (_gather_candidate_positions_numba,
# _fps_batched_numba) that run inside every DataLoader worker's collate step. numba's parallel
# backend is a SEPARATE thread pool from torch/BLAS - torch.set_num_threads() and the OMP/MKL/
# OPENBLAS env vars above do NOT touch it. Verified in isolation: an identical @njit(parallel=True)
# function spun up 24 threads (numba.get_num_threads()==24, matching nproc) with this env var
# unset, and exactly 1 with it set - this was the actual majority contributor to the ~26
# threads/worker and ~22-core aggregate CPU load measured on the live training process, not the
# BLAS-level env vars alone (those were necessary but not sufficient).

import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset.patch_dataset_whole_tooth import DEFAULT_MARGIN_MM, PatchTeeth3DSDatasetWithWholeTooth
from dataset.patch_face_cap import DEFAULT_MAX_FACES, MaxFaceCapTransform
from dataset.patch_losses import compute_class_alpha
from dataset.patch_preprocessing import PatchPreTransform
from dataset.patch_preprocessing_color import PatchPreTransformWithColor
from dataset.patch_preprocessing_color_dropout import ColorDropoutWrapper
from dataset.patch_preprocessing_realistic_color_hsv import PatchPreTransformWithRealisticColorHSV
from models.patch_collate import PatchCollator
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork
from models.patch_lightning_module_transformer import PatchLitDilatedToothSegTransformerNetwork
from models.patch_lightning_module_transformer_deep import PatchLitDilatedToothSegTransformerDeepNetwork

torch.set_num_threads(1)  # same reasoning as the env vars above - belt-and-suspenders for any
# torch CPU op that doesn't consult the env vars (interpreter-level torch threading, distinct
# from the BLAS backend's own pool)
torch.set_num_interop_threads(1)  # confirmed empirically this is a SEPARATE pool from the above -
# torch.get_num_interop_threads() defaults to nproc (24 here) and set_num_threads() does not touch
# it; must be called before any interop-parallel work happens (torch raises if called twice /
# after first use), so this has to stay this early in the file

# Synthetic-color-input variant of train_patch_network.py (untouched) - same reproducibility
# setup, dataloader pattern, checkpoint/logging structure, and CLI, copied rather than imported
# since train_patch_network.py's own body isn't reusable piecewise (its dataset/model
# construction is inlined under `if __name__ == "__main__":`). What's different:
#   - get_datasets() passes transform=PatchPreTransformWithColor(...) instead of leaving
#     PatchTeeth3DSDataset's default (PatchPreTransform); everything else about the dataset class
#     is unchanged and unmodified (dataset/patch_dataset.py) - the transform= constructor
#     argument is the existing extension point this whole feature is built on
#   - the model is constructed with feature_dim=27 (24 base + 3 synthetic RGB), also an existing,
#     already-parameterized constructor argument on PatchLitDilatedToothSegmentationNetwork
#     (models/patch_lightning_module.py) - no other model file needed touching
#   - --experiment_version defaults to a name tagged "_color" so runs never collide with the
#     geometry-only experiment matrix's checkpoints/logs
#   - --area_thresholds / --early_bias_power expose two more already-existing constructor
#     parameters (PatchLitDilatedToothSegmentationNetwork's area_thresholds,
#     PatchTeeth3DSDataset's early_bias_power) as CLI flags, defaulted to the SAME values those
#     classes already default to (40/180/360mm2, 2.5) - so a run that doesn't pass them behaves
#     identically to before. Added to investigate a real hardware finding: cycle_0001 of
#     realtime/logs/snapshots/mesh_viewer_segmented_20260815_122810/ shows the model predicting 5
#     different classes (0,7,8,9,10) on a 52.6mm2 patch that's realistically just 2 teeth - at
#     that area only the first dilated block is gated on (area_thresholds[0]=40), and its own
#     candidate pool (dilation_k=200) is already close to the whole patch's face count, so the
#     network is effectively reasoning from local shape alone at exactly the point in a scan where
#     adjacent-tooth geometric self-similarity is most likely to be mistaken for a neighbor.
#     Lowering these thresholds and/or early_bias_power (currently skewed hard toward the very
#     smallest patches - _sample_stage_index in dataset/patch_dataset.py uses
#     rng.random()**early_bias_power) are two candidate mitigations to try in the NEXT run, not
#     this one - see Docs/REALTIME.md for the full finding.
#   - --max_faces (default 20000, dataset/patch_face_cap.py's MaxFaceCapTransform) caps patch size
#     during training - added after both color runs crashed overnight (2026-08-19) with
#     torch.AcceleratorError: CUDA error: the launch timed out and was terminated. Real cause,
#     confirmed via journalctl: NVRM Xid 8 ("RC watchdog: GPU is probably locked!") - this
#     machine's GPU also drives the desktop, so the driver kills any single CUDA kernel running
#     longer than ~7s. models/layer.py's knn() (protected) does an uncapped O(N^2) torch.cdist for
#     the local edge-conv blocks whenever no precomputed idx is passed, and no face-count cap
#     existed anywhere in the training path - an occasional very large patch (late-stage/full-arch
#     growth stages) made a single kernel run long enough to trip the watchdog. Live inference
#     already solved this exact problem with max_model_faces=20000
#     (realtime/mesh_viewer_segmented.py) - reused here for the same reason and to keep
#     train/inference face-count distributions consistent.

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_datasets(root, processed_folder, train_test_split, num_classes, early_bias_power, max_faces,
                  color_style='flat', whole_tooth_patch_prob=0.3, whole_tooth_margin_mm=DEFAULT_MARGIN_MM,
                  color_dropout_prob=0.0):
    color_transform_cls = {
        # Added 2026-08-22 for the RGB-effectiveness ablation (run_experiment_matrix.sh): the
        # plain 24-dim PatchPreTransform (dataset/patch_preprocessing.py, untouched) - no color
        # channel appended at all, not even a zeroed one. Same __init__(scale_mm, classes)
        # signature as every *WithColor* transform below, so it drops into color_transform_cls(
        # classes=num_classes) identically - no special-casing needed at the call site.
        'none': PatchPreTransform,
        'flat': PatchPreTransformWithColor,
        # Switched 2026-08-22 to the HSV-space paint (dataset/patch_color_augmentation.py's
        # RealisticColorPaintHSV) - the original RealisticColorPaint's independent-per-channel RGB
        # noise was confirmed (testing/visualize_training_input.py) to occasionally produce
        # implausible saturated colors (a sampled tooth face came out as visible purple/magenta).
        # RealisticColorPaint/PatchPreTransformWithRealisticColor are left untouched as a fallback,
        # same precedent as SyntheticColorPaint being kept when RealisticColorPaint was added.
        'mosaic': PatchPreTransformWithRealisticColorHSV,
    }[color_style]

    if color_style == 'none' and color_dropout_prob > 0.0:
        raise ValueError("--color_dropout_prob > 0 has no meaning with --color_style none - there's "
                          "no color channel to drop out of a 24-dim (geometry-only) feature vector.")

    def make_transform():
        inner = color_transform_cls(classes=num_classes)
        # 0.0 (default): unchanged from before this flag existed - no wrapper, byte-identical
        # behavior. See dataset/patch_preprocessing_color_dropout.py for why this exists at all.
        if color_dropout_prob > 0.0:
            inner = ColorDropoutWrapper(inner, dropout_prob=color_dropout_prob)
        return MaxFaceCapTransform(inner, max_faces=max_faces)

    # PatchTeeth3DSDatasetWithWholeTooth (dataset/patch_dataset_whole_tooth.py, additive subclass -
    # PatchTeeth3DSDataset itself untouched): at whole_tooth_patch_prob=0.0 every draw still takes
    # the original scan-sweep growth path, just via a subclass __getitem__ that duplicates it rather
    # than the base class's own - see that file's own docstring on why this isn't byte-identical to
    # plain PatchTeeth3DSDataset output even at 0.0 (same distribution, different specific rng draws).
    train = PatchTeeth3DSDatasetWithWholeTooth(
        root, processed_folder=processed_folder, num_classes=num_classes,
        is_train=True, train_test_split=train_test_split, early_bias_power=early_bias_power,
        transform=make_transform(), whole_tooth_patch_prob=whole_tooth_patch_prob,
        whole_tooth_margin_mm=whole_tooth_margin_mm)
    # val stays pinned to whole_tooth_patch_prob=0.0 (pure scan-sweep) REGARDLESS of the train-side
    # knob above - the live client (realtime/mesh_viewer_segmented.py) only ever generates
    # scan-sweep-footprint patches, never whole-tooth ones, so val_miou needs to keep measuring
    # performance on that same real-deployment distribution for it to stay comparable across runs
    # (e.g. against the color_baseline/color_tuned checkpoints already in checkpoints/, both pure
    # scan-sweep). Mixing whole-tooth patches into val would answer a different question ("does the
    # model also get better at whole-tooth patches") instead of the one that matters here (does
    # training WITH them improve real scan-sweep performance).
    val = PatchTeeth3DSDatasetWithWholeTooth(
        root, processed_folder=processed_folder, num_classes=num_classes,
        is_train=False, train_test_split=train_test_split, early_bias_power=early_bias_power,
        transform=make_transform(), whole_tooth_patch_prob=0.0,
        whole_tooth_margin_mm=whole_tooth_margin_mm)
    return train, val


def get_dataloaders(train_dataset, val_dataset, model, num_workers):
    kwargs = dict(persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None)
    train_collator = PatchCollator.for_network(model.model)  # seed=None: fresh randomness every draw
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=train_collator, **kwargs)
    val_collator = PatchCollator.for_network(model.model, seed=0)  # fixed seed: reproducible across runs
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collator, **kwargs)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run patch-based training with synthetic per-face color input')
    parser.add_argument('--root', type=str, default='data/3dteethseg')
    parser.add_argument('--processed_folder', type=str, default='processed_w5',
                         help='The UNCAPPED, true-density cache - not processed_capped16k (see '
                              'dataset/patch_dataset.py)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=17, choices=[5, 17],
                         help='Experiment flag from claude_code_prompt.md Component 5')
    parser.add_argument('--dilation_gating', type=str, default='on', choices=['on', 'off'],
                         help='Experiment flag from claude_code_prompt.md Component 5')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                         help='Gradient accumulation over N batch_size=1 steps for a larger '
                              'effective batch - see the Component 4 batching decision')
    parser.add_argument('--num_workers', type=int, default=8,
                         help='CPU workers for the neighbor-graph precompute (models/patch_collate.py)')
    parser.add_argument('--no_class_weights', action='store_true',
                         help='Disable automatic inverse-frequency FocalLoss weighting (on by default)')
    parser.add_argument('--class_alpha_samples', type=int, default=200,
                         help='How many patches to sample when measuring class frequency for FocalLoss alpha')
    parser.add_argument('--tb_save_dir', type=str, default='logs/tensorboard')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                         help='Where model checkpoints (.ckpt) are saved - kept separate from '
                              '--tb_save_dir so logs and (large) checkpoint files do not mix')
    parser.add_argument('--experiment_name', type=str, default='patch_dilated_tooth_seg_net')
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--devices', nargs='+', default=[0])
    parser.add_argument('--n_bit_precision', type=str, default='32-true')
    parser.add_argument('--train_test_split', type=int, default=1)
    parser.add_argument('--ckpt', type=str, required=False, default=None,
                         help='Checkpoint path to resume training')
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--area_thresholds', type=float, nargs=3, default=[40.0, 180.0, 360.0],
                         metavar=('GATE1_MM2', 'GATE2_MM2', 'GATE3_MM2'),
                         help='mm^2 area a patch must reach for dilated blocks 1/2/3 to gate on - '
                              'default matches PatchDilatedToothSegmentationNetwork\'s own default, '
                              'so omitting this flag reproduces current behavior exactly')
    parser.add_argument('--dilation_ks', type=int, nargs=3, default=[200, 900, 1800],
                         metavar=('K1', 'K2', 'K3'),
                         help='Candidate-neighborhood size (pre-FPS-downsample) for dilated blocks '
                              '1/2/3 - default matches PatchDilatedToothSegmentationNetwork\'s own '
                              'default. Larger values let a block see further across the patch '
                              'before subsampling, at the cost of a bigger cdist in '
                              'models/patch_collate.py\'s neighbor precompute - see --max_faces\'s '
                              'own help on the GPU driver watchdog risk that already bit the '
                              '2026-08-19 tuned-area_thresholds run (crashed at epoch 56/99 with a '
                              'CUDA launch timeout, logs/experiments/17class_gateOn_color_tuned.log)')
    parser.add_argument('--architecture', choices=['dilated', 'dilated_transformer'], default='dilated',
                         help="'dilated' (default, unchanged): PatchDilatedToothSegmentationNetwork - "
                              "local + dilated edge-conv blocks only. 'dilated_transformer': "
                              "PatchDilatedToothSegTransformerNetwork (models/patch_dilated_tooth_seg_"
                              "transformer_network.py) - same backbone plus a GlobalTokenTransformerBlock "
                              "(models/patch_transformer_layer.py) run in parallel with the dilated "
                              "blocks, giving every point genuine all-pairs global context that no "
                              "existing block provides (FPS-downsample to a token budget, self-attend "
                              "among tokens, cross-attend tokens back onto every point - see that "
                              "file's own module docstring for the full reasoning and the O(N^2) "
                              "cost argument against attending over every face directly). Added "
                              "2026-08-22, motivated by the tooth-4-vs-5 premolar confusion this "
                              "project's own dilation_ks widening experiment only partly addressed "
                              "through wider LOCAL context alone (Docs/REALTIME.md).")
    parser.add_argument('--transformer_tokens', type=int, default=256,
                         help="Only used with --architecture dilated_transformer. FPS token budget "
                              "for the global attention block - clamped to the patch's own face count "
                              "for small patches.")
    parser.add_argument('--transformer_layers', type=int, default=2,
                         help="Only used with --architecture dilated_transformer. Self-attention "
                              "layers among the FPS-sampled tokens.")
    parser.add_argument('--transformer_heads', type=int, default=4,
                         help="Only used with --architecture dilated_transformer. Attention heads, "
                              "both for token self-attention and the point<-tokens cross-attention "
                              "broadcast.")
    parser.add_argument('--add_late_global', action='store_true',
                         help="Only used with --architecture dilated_transformer. Adds a SECOND "
                              "GlobalTokenTransformerBlock (models/patch_dilated_tooth_seg_"
                              "transformer_deep_network.py) operating on global_hidden_layer's "
                              "fused 1024-dim local+dilated+early-global output, instead of just "
                              "the first block's raw 60-dim local-only input - richer per-token "
                              "context for a second, later global attention pass, on top of (not "
                              "instead of) the early one. Joins after PointFeatureImportance, "
                              "widening res_block1's input - does NOT replace res_block1/res_block2/ "
                              "the output head (see that file's own module docstring for why).")
    parser.add_argument('--late_transformer_embed_dim', type=int, default=128,
                         help="Only used with --add_late_global. Embed dim for the late block - "
                              "compressed down from its 1024-dim input by default (unlike the "
                              "early block, which matches its 60-dim input exactly), since "
                              "attention cost scales with embed_dim too and a full 1024-dim pass "
                              "would partially undo FPS-downsampling's point.")
    parser.add_argument('--whole_tooth_patch_prob', type=float, default=0.3,
                         help='Fraction of TRAIN draws that use the whole-tooth curriculum '
                              '(dataset/patch_generator_whole_tooth.py: ground-truth-label-bounded, '
                              'margin-dilated, grown one whole tooth at a time) instead of the '
                              'existing scan-sweep local-footprint growth - mixed in probabilistically '
                              'per Docs/MAA_ToDo.md step 4, NOT a replacement. val always uses 0.0 '
                              '(pure scan-sweep) regardless of this flag - see get_datasets()\'s own '
                              'comment on why.')
    parser.add_argument('--whole_tooth_margin_mm', type=float, default=DEFAULT_MARGIN_MM,
                         help='Spatial dilation margin (mm) applied to each whole-tooth mask before '
                              'use - user-decided 2026-08-19 default of 1.5mm, see '
                              'dataset/patch_generator_whole_tooth.py\'s own comment on why a pure '
                              'label-boundary mask (margin=0) is unlike anything a real scan produces')
    parser.add_argument('--early_bias_power', type=float, default=2.5,
                         help='How hard training patch-stage sampling skews toward small/early '
                              'stages (rng.random()**power) - default matches '
                              'PatchTeeth3DSDataset\'s own default. Lower = less skew toward the '
                              'smallest patches, more weight on the partially-gated mid-size regime')
    parser.add_argument('--max_faces', type=int, default=DEFAULT_MAX_FACES,
                         help='Caps patch face count during training (dataset/patch_face_cap.py) - '
                              'prevents the uncapped O(N^2) local edge-conv cdist from triggering '
                              'a GPU driver watchdog kill on very large patches. Matches live '
                              "inference's own max_model_faces default (20000).")
    parser.add_argument('--color_style', choices=['none', 'flat', 'mosaic'], default='flat',
                         help="'flat' (default, unchanged): SyntheticColorPaint - one shared RGB "
                              "draw per class-group per patch, from hand-picked TOOTH_RGB_LOW/HIGH "
                              "and GUM_RGB_LOW/HIGH ranges. 'mosaic': RealisticColorPaintHSV - every "
                              "face gets its own draw (correlated HSV, not independent RGB - see "
                              "dataset/patch_color_augmentation.py) in a cream/white/yellowish "
                              "(teeth) or red/pink (gum) hue family, calibrated to the MEASURED real "
                              "scanner color distribution (analyze_snapshot_colors.py). Added "
                              "2026-08-19 after confirming empirically that flat's zero intra-class "
                              "variance, not just its color values, was itself hurting live hardware "
                              "accuracy. 'none': plain PatchPreTransform, no color channel at all "
                              "(feature_dim=24, not 27) - the RGB-effectiveness ablation baseline "
                              "(added 2026-08-22, run_experiment_matrix.sh) - --color_dropout_prob "
                              "must be 0 with this choice, there's no color to drop out of.")
    parser.add_argument('--color_dropout_prob', type=float, default=0.0,
                         help='Fraction of faces per patch whose color is overwritten with the '
                              'live no-color sentinel (dataset/patch_preprocessing_color_dropout.py) '
                              'instead of the color_style-painted value. 0.0 (default): unchanged, '
                              'no dropout. Added 2026-08-21 after live hardware confirmed a '
                              'color-trained checkpoint reads real live color-lessness (65-88%% of '
                              'faces in a real scan, per analyze_snapshot_colors.py) as strongly '
                              'gum-like, since training never once produced a color-less face for '
                              'it to learn to discount. Applied to BOTH train and val (this flag '
                              'changes what make_transform() returns identically for both datasets '
                              'in get_datasets() below) - deliberately, not an oversight: a val set '
                              'with 100%% valid color would keep measuring a distribution real '
                              'inference never sees, the same disconnect that made val_miou blind '
                              'to the live gum over-prediction in the first place. This means '
                              'val_miou from a --color_dropout_prob > 0 run is NOT directly '
                              'comparable to earlier runs (0.0) - expect a lower number that means '
                              'something closer to real deployment, not a regression.')

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    train_dataset, val_dataset = get_datasets(args.root, args.processed_folder, args.train_test_split,
                                               args.num_classes, args.early_bias_power, args.max_faces,
                                               args.color_style, args.whole_tooth_patch_prob,
                                               args.whole_tooth_margin_mm, args.color_dropout_prob)

    class_alpha = None
    if not args.no_class_weights:
        print(f'Measuring class frequency from {args.class_alpha_samples} sampled patches for FocalLoss alpha...')
        class_alpha = compute_class_alpha(train_dataset, num_classes=args.num_classes,
                                           n_samples=args.class_alpha_samples)
        print(f'class_alpha: {class_alpha.tolist()}')

    if args.add_late_global and args.architecture != 'dilated_transformer':
        print("[!] --add_late_global has no effect without --architecture dilated_transformer - ignoring it.")

    # 24 base geometry dims + 3 color dims, unless --color_style none (24 only) - see
    # get_datasets()'s color_transform_cls dict above.
    feature_dim = 24 if args.color_style == 'none' else 27

    if args.architecture == 'dilated_transformer' and args.add_late_global:
        model = PatchLitDilatedToothSegTransformerDeepNetwork(
            num_classes=args.num_classes, feature_dim=feature_dim, dilation_gating=(args.dilation_gating == 'on'),
            area_thresholds=tuple(args.area_thresholds), dilation_ks=tuple(args.dilation_ks),
            transformer_tokens=args.transformer_tokens, transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads, late_transformer_embed_dim=args.late_transformer_embed_dim,
            class_alpha=class_alpha, lr=args.lr)
    elif args.architecture == 'dilated_transformer':
        model = PatchLitDilatedToothSegTransformerNetwork(
            num_classes=args.num_classes, feature_dim=feature_dim, dilation_gating=(args.dilation_gating == 'on'),
            area_thresholds=tuple(args.area_thresholds), dilation_ks=tuple(args.dilation_ks),
            transformer_tokens=args.transformer_tokens, transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads, class_alpha=class_alpha, lr=args.lr)
    else:
        model = PatchLitDilatedToothSegmentationNetwork(
            num_classes=args.num_classes, feature_dim=feature_dim, dilation_gating=(args.dilation_gating == 'on'),
            area_thresholds=tuple(args.area_thresholds), dilation_ks=tuple(args.dilation_ks),
            class_alpha=class_alpha, lr=args.lr)

    train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, model, args.num_workers)

    if args.experiment_version is None:
        arch_tag = '' if args.architecture == 'dilated' else f'_{args.architecture}'
        if args.architecture == 'dilated_transformer' and args.add_late_global:
            arch_tag += '_lateglobal'
        experiment_version = f'nc{args.num_classes}_dg{args.dilation_gating}_color_{args.color_style}{arch_tag}'
    else:
        experiment_version = args.experiment_version

    logger = TensorBoardLogger(args.tb_save_dir, name=args.experiment_name, version=experiment_version)

    # same {experiment_name}/{experiment_version} structure as the TensorBoard logs, but under
    # its own top-level directory - keeps large .ckpt files out of the log tree, and makes it
    # obvious where to look for either one
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name, experiment_version)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='best-{epoch}-{val_miou:.4f}',
                                           save_top_k=1, monitor="val_miou", mode='max',
                                           save_last=True)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                          devices=[int(d) for d in args.devices], enable_progress_bar=True, logger=logger,
                          precision=args.n_bit_precision, callbacks=[checkpoint_callback], deterministic=False,
                          accumulate_grad_batches=args.accumulate_grad_batches,
                          limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=args.ckpt)
