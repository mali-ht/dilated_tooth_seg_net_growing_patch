"""Four controlled runs. No implicit launch, resume or background execution."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
from pathlib import Path
import subprocess
import sys
import random
from collections import Counter
import numpy as np

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from experiments.controlled.data import ControlledDataset, EpochSampler, collate, digest
from experiments.controlled.metrics import StratifiedMetrics
from experiments.controlled.prepare import file_hash, validate_manifest
from models.patch_lightning_module_transformer import PatchLitDilatedToothSegTransformerNetwork

RECIPES = {'A_inverse_sweep': ('inverse', False), 'B_sqrt_sweep': ('sqrt', False),
           'C_inverse_targeted': ('inverse', True), 'D_sqrt_targeted': ('sqrt', True)}


class RandomState(Callback):
    def on_save_checkpoint(self, trainer, module, checkpoint):
        numpy_state = np.random.get_state()
        checkpoint['controlled_rng'] = {'torch': torch.get_rng_state(),
                                        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                                        'python': random.getstate()}
        if module.device.type == 'cuda':
            checkpoint['controlled_rng']['cuda'] = torch.cuda.get_rng_state(module.device)

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.pending = checkpoint['controlled_rng']

    def on_train_start(self, trainer, module):
        if not hasattr(self, 'pending'):
            return
        state = self.pending
        torch.set_rng_state(state['torch'].cpu())
        numpy_state = state['numpy']
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        random.setstate(state['python'])
        if 'cuda' in state:
            torch.cuda.set_rng_state(state['cuda'].cpu(), module.device)
        del self.pending


def checkpoint_callbacks(directory):
    # Independent unmonitored save writes the CURRENT state each epoch even when
    # val_miou does not improve. Never copy a stale best file into last.ckpt.
    best = ModelCheckpoint(dirpath=directory, monitor='val_miou', mode='max',
                           filename='best-{epoch:03d}-{val_miou:.4f}', save_top_k=1,
                           save_last=False, save_on_train_epoch_end=False)
    last = ModelCheckpoint(dirpath=directory, monitor=None, filename='last',
                           save_top_k=1, save_last=False, every_n_epochs=1,
                           save_on_train_epoch_end=True, enable_version_counter=False)
    return [RandomState(), best, last]


class ControlledModel(PatchLitDilatedToothSegTransformerNetwork):
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return {**batch, 'patch': batch['patch'].to(device)}

    def on_train_epoch_start(self):
        self.exposure_faces = np.zeros(17, dtype=np.int64)
        self.exposure_patches = np.zeros(17, dtype=np.int64)
        self.exposure_modes = Counter()

    def training_step(self, batch, batch_idx):
        counts = np.bincount(batch['patch'].labels[0].detach().cpu().numpy(), minlength=17)
        self.exposure_faces += counts
        self.exposure_patches += counts > 0
        self.exposure_modes[(batch['metadata']['requested_mode'], batch['metadata']['mode'])] += 1
        return super().training_step(batch['patch'], batch_idx)

    def on_train_epoch_end(self):
        out = Path(self.logger.log_dir) / 'exposure'
        out.mkdir(exist_ok=True)
        data = {'faces_per_class': self.exposure_faces.tolist(),
                'patches_per_class': self.exposure_patches.tolist(),
                'requested_to_actual_mode': {f'{a}->{b}': n for (a, b), n in self.exposure_modes.items()}}
        (out / f'epoch_{self.current_epoch:03d}.json').write_text(json.dumps(data, indent=2))

    def on_validation_epoch_start(self):
        self.strata = StratifiedMetrics()

    def validation_step(self, batch, batch_idx):
        patch = batch['patch']
        logits, loss = self._forward_and_loss(patch)
        self.val_acc(logits, patch.labels)
        self.val_miou(logits, patch.labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=1)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log('val_miou', self.val_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.strata.add(patch.labels[0].cpu().numpy(), logits.argmax(1)[0].cpu().numpy(),
                        batch['face_area'].cpu().numpy(), batch['metadata'])

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        report = self.strata.result()
        overall = report['all']
        for name in ('false_terminal_area_fraction', 'false_identities_per_patch', 'pair_merge_proxy'):
            self.log('failure/' + name, overall[name])
        out = Path(self.logger.log_dir) / 'failure_metrics'
        out.mkdir(exist_ok=True)
        (out / f'epoch_{self.current_epoch:03d}.json').write_text(json.dumps(report, indent=2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Fixed optimizer-update schedule; identical in all four recipes.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [72000, 144000, 216000], gamma=.5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


def loader(dataset, workers, sampler=None):
    return DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False,
                      num_workers=workers, persistent_workers=workers > 0,
                      collate_fn=collate, **({'prefetch_factor': 2} if workers else {}))


def code_fingerprint():
    paths = sorted(set(Path('experiments/controlled').glob('*.py')) |
                   set(Path('dataset').glob('*.py')) | set(Path('models').glob('*.py')))
    return digest({str(p): p.read_text() for p in paths})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--manifest', default='experiments/controlled/manifests/v1/manifest.json')
    p.add_argument('--root', default='data/3dteethseg/processed_w5')
    p.add_argument('--output', default='experiment_artifacts/runs')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--steps-per-epoch', type=int, default=1200)
    p.add_argument('--validate-every', type=int, default=5)
    p.add_argument('--resume', action='store_true', help='Explicitly resume this exact run from last.ckpt')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.steps_per_epoch < 1 or a.validate_every < 1 or a.workers < 0:
        p.error('Positive epoch/step/validation counts and nonnegative workers required')
    if a.epochs % a.validate_every:
        p.error('epochs must be divisible by validate-every so the final epoch is evaluated')
    weighting, targeted = RECIPES[a.recipe]
    print(json.dumps({'recipe': a.recipe, 'weighting': weighting, 'targeted': targeted,
                      'optimizer_updates': a.epochs * a.steps_per_epoch, 'seed': a.seed,
                      'device': a.device, 'manifest': a.manifest, 'resume': a.resume}, indent=2))
    if a.dry_run:
        return
    torch.set_num_threads(1)
    pl.seed_everything(a.seed, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for experiment training; no silent CPU fallback')
    if not 0 <= a.device < torch.cuda.device_count():
        raise ValueError(f'GPU {a.device} is not visible; available count: {torch.cuda.device_count()}')
    manifest = json.loads(Path(a.manifest).read_text())
    validate_manifest(manifest)
    for f, size in manifest['cache_sizes'].items():
        path = Path(a.root) / f
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'Missing/different cache: {path}. Run prepare --verify after transfer.')
        if file_hash(path) != manifest['cache_sha256'][f]:
            raise ValueError(f'Cache content differs from the shared manifest: {path}')
    run = Path(a.output) / f'{a.recipe}_seed{a.seed}'
    config = dict(recipe=a.recipe, seed=a.seed, manifest_id=manifest['id'],
                  epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
                  validate_every=a.validate_every, alpha=manifest['alpha'][weighting],
                  code_sha256=code_fingerprint())
    ckpt = run / 'checkpoints' / 'last.ckpt'
    if a.resume:
        if not ckpt.exists() or json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Resume requires matching saved configuration, code and last.ckpt')
    elif run.exists():
        raise FileExistsError(f'{run} exists. Use --resume explicitly or a different output/seed.')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    provenance = {'command': sys.argv, 'python': sys.version, 'torch': torch.__version__,
                  'lightning': pl.__version__, 'gpu': torch.cuda.get_device_name(a.device),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (run / ('resume_environment.json' if a.resume else 'environment.json')).write_text(json.dumps(provenance, indent=2))
    train = ControlledDataset(a.root, manifest['splits']['train'], seed=a.seed,
                               length=a.steps_per_epoch, targeted=targeted)
    val = ControlledDataset(a.root, manifest['splits']['val'], records=manifest['val'], augment=False)
    model = ControlledModel(num_classes=17, feature_dim=27, area_thresholds=(20, 90, 180),
                            class_alpha=torch.tensor(manifest['alpha'][weighting]))
    callbacks = checkpoint_callbacks(run / 'checkpoints')
    trainer = pl.Trainer(max_epochs=a.epochs, accelerator='gpu', devices=[a.device],
                         precision='32-true', callbacks=callbacks, deterministic=True,
                         check_val_every_n_epoch=a.validate_every,
                         logger=TensorBoardLogger(str(run), name='tensorboard', version='metrics'))
    start_epoch = torch.load(ckpt, map_location='cpu', weights_only=False)['epoch'] + 1 if a.resume else 0
    if start_epoch >= a.epochs:
        raise ValueError('This run is already complete')
    trainer.fit(model, loader(train, a.workers, EpochSampler(len(train), start_epoch)), loader(val, a.workers),
                ckpt_path=str(ckpt) if a.resume else None)
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if saved['epoch'] != a.epochs - 1 or saved['global_step'] != a.epochs * a.steps_per_epoch:
        raise RuntimeError('Final checkpoint epoch/step verification failed')
    print(f'Completed and verified last checkpoint: {ckpt}')


if __name__ == '__main__':
    main()
