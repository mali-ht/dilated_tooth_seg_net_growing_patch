import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -
# same pattern as testing/test_patch_model.py and the other testing/*.py scripts
from dataset.patch_dataset import PatchTeeth3DSDataset  # noqa: E402
from dataset.patch_preprocessing import PatchPreTransform  # noqa: E402
from dataset.patch_preprocessing_color import PatchPreTransformWithColor  # noqa: E402
from models.patch_collate import PatchCollator  # noqa: E402
from models.patch_lightning_module import PatchLitDilatedToothSegmentationNetwork  # noqa: E402

# New, additive-only script (nothing under dataset/ or models/ is modified) - a fast, no-retrain
# way to check whether different area_thresholds gate values look promising BEFORE committing
# hours to a full retrain with them. area_thresholds only controls a forward-time GATE
# (PatchDilatedToothSegmentationNetwork._resolve_gates) - the dilated blocks themselves are
# already-trained weights that exist regardless of the gate, so this can be swept on an EXISTING
# checkpoint by just reassigning model.model.area_thresholds before each forward pass, no
# retraining needed.
#
# CAVEAT (real, not hand-waved): the loaded checkpoint's weights were trained with WHATEVER
# thresholds it was actually trained under (see its hparams.yaml) - forcing a different threshold
# at inference time tells you how the EXISTING weights behave under that gate, not how a model
# actually trained with that gate from scratch would behave (the dilated blocks may be poorly
# calibrated in an area regime they rarely saw active during training). Treat results as a
# directional signal for picking what to try in the next real training run, not as the final word -
# a full retrain is still the real test.
#
# Uses PatchTeeth3DSDataset.get_full_sequence(index) (dataset/patch_dataset.py, untouched) rather
# than __getitem__, since we want every growth stage of each val arch (the exact
# small-to-large-patch progression the mesh_viewer_segmented.py hardware finding was about), not
# one random stage per arch. Ground truth is read from PatchCollator's OWN output
# (batch.labels), not the pre-collate stage tensors - PatchCollator can subsample the face count
# (verified empirically: a stage with 5205 faces pre-collate had batch.pos.shape[1]==3946 in an
# earlier check this session), so pred_labels and gt must both come from the same post-collate
# face set or they silently misalign.

AREA_BUCKETS = [(0.0, 100.0), (100.0, 400.0), (400.0, float('inf'))]


def bucket_for(area_mm2: float) -> str:
    for lo, hi in AREA_BUCKETS:
        if lo <= area_mm2 < hi:
            return f'{lo:g}-{hi:g}' if hi != float('inf') else f'{lo:g}+'
    return 'unknown'


def accumulate_confusion(cm: np.ndarray, num_classes: int, gt: np.ndarray, pred: np.ndarray) -> None:
    idx = num_classes * gt + pred
    cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray):
    """acc, macro_iou - macro_iou excludes classes with zero union (matches torchmetrics'
    JaccardIndex(average='macro') behavior, confirmed empirically this session - see
    models/patch_lightning_module.py's own comment on val_miou)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    valid = union > 0
    iou = np.zeros_like(tp)
    iou[valid] = tp[valid] / union[valid]
    macro_iou = float(iou[valid].mean()) if valid.any() else float('nan')
    total = cm.sum()
    acc = float(tp.sum() / total) if total > 0 else float('nan')
    return acc, macro_iou


def main():
    parser = argparse.ArgumentParser(
        description='Sweep area_thresholds (no retraining) against the val set on an existing checkpoint')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--root', type=str, default='data/3dteethseg')
    parser.add_argument('--processed_folder', type=str, default='processed_w5')
    parser.add_argument('--train_test_split', type=int, default=1)
    parser.add_argument('--max_archs', type=int, default=10,
                         help='How many val archs to sweep over (each contributes its full growth '
                              'sequence, ~dozens of stages) - kept modest by default since every '
                              'stage is re-run once per threshold config')
    parser.add_argument('--thresholds', type=float, nargs=3, action='append', required=True,
                         metavar=('GATE1_MM2', 'GATE2_MM2', 'GATE3_MM2'),
                         help='Repeat this flag once per candidate triple to compare, e.g. '
                              '--thresholds 40 180 360 --thresholds 20 90 180')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--csv_out', type=str, default=None)
    args = parser.parse_args()

    model = PatchLitDilatedToothSegmentationNetwork.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval()
    model.to(args.device)

    if not model.model.dilation_gating:
        print('WARNING: this checkpoint was trained with dilation_gating=False - _resolve_gates '
              'ignores area_thresholds entirely in that mode, so every config below will be a no-op.')

    feature_dim = getattr(model.hparams, 'feature_dim', 24)
    num_classes = model.hparams.num_classes
    transform = PatchPreTransformWithColor(classes=num_classes) if feature_dim == 27 \
        else PatchPreTransform(classes=num_classes)
    print(f'Checkpoint: {args.ckpt}\n  feature_dim={feature_dim} num_classes={num_classes} '
          f'dilation_gating={model.model.dilation_gating} dilation_ks={model.model.dilation_ks} '
          f'trained area_thresholds={model.model.area_thresholds}')

    val_dataset = PatchTeeth3DSDataset(args.root, processed_folder=args.processed_folder, num_classes=num_classes,
                                        is_train=False, train_test_split=args.train_test_split, transform=transform)
    n_archs = min(args.max_archs, len(val_dataset))
    print(f'Loading full growth sequences for {n_archs}/{len(val_dataset)} val archs...')
    all_sequences = []
    for idx in range(n_archs):
        sequence, _stages = val_dataset.get_full_sequence(idx)
        all_sequences.append(sequence)
        print(f'  arch {idx}: {len(sequence)} stages')
    total_stages = sum(len(s) for s in all_sequences)

    # seed=0: same deterministic face-subsample/neighbor-graph draw reused for every threshold
    # config below - so the ONLY thing that changes between configs is area_thresholds itself.
    collator = PatchCollator.for_network(model.model, seed=0)

    rows = []
    for triple in args.thresholds:
        triple = tuple(float(t) for t in triple)
        model.model.area_thresholds = triple

        overall_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        bucket_cms = {}
        done = 0
        for sequence in all_sequences:
            for item in sequence:
                area_mm2 = item[-1]
                batch = collator([item])
                x_b = batch.x.to(args.device)
                pos_b = batch.pos.to(args.device)
                local_idx_b = batch.local_idx.to(args.device)
                dilated_idx_b = tuple(d.to(args.device) for d in batch.dilated_idx)
                with torch.no_grad():
                    pred = model.model(x_b, pos_b, area_mm2=batch.area_mm2, local_idx=local_idx_b,
                                        dilated_idx=dilated_idx_b)
                    pred_labels = pred.argmax(dim=-1)[0].cpu().numpy()
                gt = batch.labels[0].cpu().numpy()

                accumulate_confusion(overall_cm, num_classes, gt, pred_labels)
                b = bucket_for(area_mm2)
                if b not in bucket_cms:
                    bucket_cms[b] = np.zeros((num_classes, num_classes), dtype=np.int64)
                accumulate_confusion(bucket_cms[b], num_classes, gt, pred_labels)
                done += 1
                if done % 25 == 0:
                    print(f'  [thresholds={triple}] {done}/{total_stages} stages done')

        acc, miou = metrics_from_confusion(overall_cm)
        row = {'thresholds': triple, 'n_stages': total_stages, 'overall_acc': acc, 'overall_miou': miou}
        print(f'thresholds={triple}: overall acc={acc:.4f} miou={miou:.4f} (n_stages={total_stages})')
        for b in sorted(bucket_cms):
            b_acc, b_miou = metrics_from_confusion(bucket_cms[b])
            n_faces = int(bucket_cms[b].sum())
            row[f'{b}mm2_acc'] = b_acc
            row[f'{b}mm2_miou'] = b_miou
            row[f'{b}mm2_n_faces'] = n_faces
            print(f'    area {b} mm2: acc={b_acc:.4f} miou={b_miou:.4f} n_faces={n_faces}')
        rows.append(row)

    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fieldnames = sorted({k for row in rows for k in row})
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {args.csv_out}')


if __name__ == '__main__':
    main()
