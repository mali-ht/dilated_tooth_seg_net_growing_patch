import argparse
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_losses import compute_class_alpha
from models.patch_collate import PatchCollator
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork

# Component 5 (claude_code_prompt.md) training entry point, adapted from train_network.py
# (untouched) - reproducibility setup, TensorBoardLogger/ModelCheckpoint pattern, and CLI
# structure all mirrored from there. What's different:
#   - PatchTeeth3DSDataset/PatchCollator/PatchLitDilatedToothSegmentationNetwork throughout,
#     not the original full-arch path
#   - batch_size is hard-coded to 1, not a CLI choice - PatchCollator enforces this (see the
#     Component 4 batching decision in TRAINING_CONCERNS.md); --accumulate_grad_batches is the
#     knob for a larger EFFECTIVE batch instead
#   - --num_workers defaults to 8: measured optimal for the CPU-side neighbor-graph precompute on
#     a 24-core machine this session (collate cost drops ~40x, from ~0.3s to ~0.007s median, once
#     it overlaps with GPU compute instead of running serially)
#   - class_alpha (FocalLoss's inverse-frequency weighting, TRAINING_CONCERNS.md #5) is measured
#     automatically from the real training distribution before the run starts, unless disabled
#   - ModelCheckpoint monitors val_miou instead of val_acc: raw accuracy can look good on gum-
#     majority patches alone, per TRAINING_CONCERNS.md's 60-70%-gum figure
#   - precision defaults to '32-true', not the original's 16-bit default - fp16 hasn't been
#     validated against this architecture's area-gating/small-N-clamp logic yet

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_datasets(root, processed_folder, train_test_split, num_classes):
    train = PatchTeeth3DSDataset(root, processed_folder=processed_folder, num_classes=num_classes,
                                  is_train=True, train_test_split=train_test_split)
    val = PatchTeeth3DSDataset(root, processed_folder=processed_folder, num_classes=num_classes,
                                is_train=False, train_test_split=train_test_split)
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
    parser = argparse.ArgumentParser(description='Run patch-based training (Components 1-5)')
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
    parser.add_argument('--tb_save_dir', type=str, default='tensorboard_logs')
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

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    train_dataset, val_dataset = get_datasets(args.root, args.processed_folder, args.train_test_split,
                                               args.num_classes)

    class_alpha = None
    if not args.no_class_weights:
        print(f'Measuring class frequency from {args.class_alpha_samples} sampled patches for FocalLoss alpha...')
        class_alpha = compute_class_alpha(train_dataset, num_classes=args.num_classes,
                                           n_samples=args.class_alpha_samples)
        print(f'class_alpha: {class_alpha.tolist()}')

    model = PatchLitDilatedToothSegmentationNetwork(
        num_classes=args.num_classes, dilation_gating=(args.dilation_gating == 'on'),
        class_alpha=class_alpha, lr=args.lr)

    train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, model, args.num_workers)

    if args.experiment_version is None:
        experiment_version = f'nc{args.num_classes}_dg{args.dilation_gating}'
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
