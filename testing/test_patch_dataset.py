import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset, collate_single_sequence


def check_transformed_item(item, num_classes):
    pos, x, labels, area_mm2 = item
    n = pos.shape[0]
    assert pos.shape == (n, 3)
    assert x.shape == (n, 24)
    assert torch.equal(pos, x[:, 9:12])
    assert labels.shape == (n,)
    assert labels.dtype == torch.long
    max_label = 4 if num_classes == 5 else 16
    assert torch.all((labels >= 0) & (labels <= max_label)), f"label out of range for {num_classes}-class"
    assert isinstance(area_mm2, float) and area_mm2 > 0, "area_mm2 should be a positive real mm^2 value"
    return n


def main(root):
    print("=== default mode (single random patch, PatchPreTransform applied): train, 17-class ===")
    train_ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1)
    print(f"  {len(train_ds)} arches in the training split")
    assert len(train_ds) > 0, "no files found - check the data wiring"
    assert train_ds.transform is not None

    t0 = time.time()
    item = train_ds[0]
    elapsed = time.time() - t0
    n = check_transformed_item(item, num_classes=17)
    print(f"  arch 0, one draw: {n} faces ({elapsed:.2f}s)")

    print("  drawing 8 more from the same arch (train mode - should vary in size):")
    sizes = []
    for _ in range(8):
        pos, x, labels, area_mm2 = train_ds[0]
        sizes.append(pos.shape[0])
    print(f"    sizes: {sizes}")
    assert len(set(sizes)) > 1, "train-mode draws from the same arch came back identical - randomness broken"
    assert min(sizes) < 2000, "early_bias_power should make small draws common"

    print()
    print("=== min_occlusality randomization ===")
    print(f"  default fixed value: {train_ds.min_occlusality} (expect 0.6)")
    assert train_ds.min_occlusality == 0.6
    print(f"  default randomization range: {train_ds.min_occlusality_range} (expect (0.2, 0.8))")
    assert train_ds.min_occlusality_range == (0.2, 0.8)
    rng = np.random.default_rng(0)
    draws = [train_ds._sample_min_occlusality(rng) for _ in range(200)]
    print(f"  200 draws: min={min(draws):.3f}  max={max(draws):.3f}  mean={sum(draws)/len(draws):.3f}")
    assert len(set(draws)) > 100, "draws should vary, not repeat the same handful of values"
    assert all(0.2 <= d <= 0.8 for d in draws), "draws should stay within the configured range"

    fixed_ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1,
                                     min_occlusality_range=None)
    assert fixed_ds._sample_min_occlusality(rng) == 0.6, "min_occlusality_range=None should disable randomization"
    print("  ok - min_occlusality_range=None correctly disables randomization (always 0.6)")

    print()
    print("=== transform=None (raw passthrough, backward compatible) ===")
    raw_ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=False, train_test_split=1, val_seed=0,
                                   transform=None)
    assert raw_ds.transform is None
    mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels, area_mm2 = raw_ds[0]
    n = mesh_faces.shape[0]
    assert mesh_triangles.shape == (n, 3, 3)
    assert mesh_vertices_normals.shape == (n, 3, 3)
    assert mesh_face_normals.shape == (n, 3)
    assert isinstance(area_mm2, float) and area_mm2 > 0
    print(f"  ok - raw 5-tuple + area_mm2 ({area_mm2:.1f} mm^2), {n} faces")

    print()
    print("=== eval mode, 17-class: reproducibility check (default transform) ===")
    val_ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=False, train_test_split=1, val_seed=0)
    print(f"  {len(val_ds)} arches in the testing split")
    item_a = val_ds[0]
    item_b = val_ds[0]
    for a, b in zip(item_a, item_b):
        # area_mm2 is a plain float (not a tensor) - torch.equal only accepts tensors
        assert (a == b) if isinstance(a, float) else torch.equal(a, b)
    print(f"  ok - identical patch ({item_a[0].shape[0]} faces, area {item_a[3]:.1f}mm^2) "
          f"across repeated calls to index 0")

    print()
    print("=== 5-class coarse labels ===")
    coarse_ds = PatchTeeth3DSDataset(root, num_classes=5, is_train=False, train_test_split=1, val_seed=0)
    item = coarse_ds[0]
    n = check_transformed_item(item, num_classes=5)
    print(f"  ok - {n} faces, all labels in [0,4]")

    print()
    print("=== get_full_sequence: still available, not the __getitem__ default ===")
    sequence, stages = val_ds.get_full_sequence(0)
    assert len(sequence) == len(stages)
    phases = [p for p, t in stages]
    assert phases == sorted(phases, key=["occlusal", "facial", "lingual"].index)
    areas = [item[3] for item in sequence]
    assert all(b >= a for a, b in zip(areas, areas[1:])), "area_mm2 should grow monotonically across the sequence"
    print(f"  ok - {len(sequence)}-stage full sequence, phases in order "
          f"{[p for p in ['occlusal', 'facial', 'lingual'] if p in phases]}, "
          f"area {areas[0]:.1f} -> {areas[-1]:.1f} mm^2")

    print()
    print("=== DataLoader integration (batch_size=1, default collate - fixed shape per call now) ===")
    loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    pos_b, x_b, labels_b, area_b = next(iter(loader))
    print(f"  batch pos shape: {tuple(pos_b.shape)} (batch dim + N + 3), area_mm2 batch: {area_b}")

    print()
    print("=== DataLoader shuffle=True: consecutive steps interleave different arches/sizes ===")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    it = iter(loader)
    seen = [next(it)[0].shape[1] for _ in range(6)]
    print(f"  6 consecutive shuffled steps, face counts: {seen}")

    print()
    print("=== rough per-sample timing (train mode, 10 draws) ===")
    t0 = time.time()
    for _ in range(10):
        train_ds[0]
    elapsed = time.time() - t0
    print(f"  {elapsed / 10:.3f}s/sample average")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check PatchTeeth3DSDataset end to end")
    parser.add_argument("--root", default="data/3dteethseg")
    args = parser.parse_args()
    main(args.root)
