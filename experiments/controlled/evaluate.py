"""Evaluate saved checkpoints on identical fixed crops; never train a model."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from functools import partial
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from experiments.controlled.data import ControlledDataset, collate
from experiments.controlled.metrics import StratifiedMetrics
from experiments.controlled.prepare import file_hash, validate_manifest
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork
from models.patch_lightning_module_transformer import PatchLitDilatedToothSegTransformerNetwork
from models.patch_lightning_module_transformer_deep import PatchLitDilatedToothSegTransformerDeepNetwork


def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hp = saved['hyper_parameters']
    if hp.get('num_classes', 17) != 17:
        raise ValueError('Controlled evaluation requires a 17-class model')
    if 'late_transformer_embed_dim' in hp:
        cls = PatchLitDilatedToothSegTransformerDeepNetwork
    elif 'transformer_tokens' in hp:
        cls = PatchLitDilatedToothSegTransformerNetwork
    else:
        cls = PatchLitDilatedToothSegmentationNetwork
    model = cls(**hp)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, saved['epoch'], hp.get('feature_dim', 24)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--manifest', default='experiments/controlled/manifests/v1/manifest.json')
    p.add_argument('--root', default='data/3dteethseg/processed_w5')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--color', choices=('mosaic', 'flat'), default='mosaic')
    p.add_argument('--output', required=True, help='New result directory, never overwritten')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--limit', type=int, help='Debug only; results marked partial')
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('limit must be positive')
    torch.set_num_threads(1)
    manifest = json.loads(Path(a.manifest).read_text())
    validate_manifest(manifest)
    records = manifest[a.split][:a.limit]
    for f in manifest['splits'][a.split]:
        if file_hash(Path(a.root) / f) != manifest['cache_sha256'][f]:
            raise ValueError(f'Evaluation data hash mismatch: {f}')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(f'cuda:{a.device}')
    for index, path in enumerate(a.checkpoints):
        model, epoch, dims = load_model(path)
        model.to(device).eval()
        ds = ControlledDataset(a.root, manifest['splits'][a.split], records=records,
                                augment=False, color='none' if dims == 24 else a.color)
        dl = DataLoader(ds, batch_size=1, num_workers=a.workers,
                         collate_fn=partial(collate, k=model.model.k, dilation_ks=model.model.dilation_ks))
        metrics = StratifiedMetrics()
        with torch.inference_mode():
            for i, batch in enumerate(dl):
                b = batch['patch'].to(device)
                logits = model.model(b.x, b.pos, area_mm2=b.area_mm2,
                                     local_idx=b.local_idx, dilated_idx=b.dilated_idx)
                metrics.add(b.labels[0].cpu().numpy(), logits.argmax(-1)[0].cpu().numpy(),
                            batch['face_area'].numpy(), batch['metadata'])
                if (i + 1) % 100 == 0:
                    print(f'{path}: {i+1}/{len(ds)}', flush=True)
        report = {'checkpoint': str(Path(path).resolve()), 'checkpoint_sha256': file_hash(path),
                  'epoch': epoch, 'manifest_id': manifest['id'], 'split': a.split,
                  'partial': a.limit is not None, 'color': 'none' if dims == 24 else a.color,
                  'caveat': 'A new split is not necessarily unseen by historical checkpoints. '
                            'No real scanner color, instance matching or robot completion metric is inferred.',
                  'metrics': metrics.result()}
        (out / f'{index:02d}_{Path(path).parent.name}.json').write_text(json.dumps(report, indent=2))
        print(path, report['metrics']['all']['macro_iou_present_union'])
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
