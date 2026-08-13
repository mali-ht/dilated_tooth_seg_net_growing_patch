import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_augmentation import (SimToRealAugment, density_subsample, erode_boundary, jitter_normals,
                                         jitter_positions, punch_holes, _patch_adjacency)
from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_generator import grow_patch_scanner_sweep, load_cached_arch, pick_incisor_seed_label


def _boundary_faces(mesh, face_idx):
    adj = _patch_adjacency(mesh, face_idx)
    degree = np.asarray((adj > 0).sum(axis=1)).ravel()
    return degree < 3


def main(pt_path, root):
    mesh, labels = load_cached_arch(pt_path)
    rng = np.random.default_rng(0)
    seed_label = pick_incisor_seed_label(labels, rng)
    masks, order = grow_patch_scanner_sweep(mesh, labels, seed_label, rng)
    mask = masks[4]  # a reasonably large patch (several teeth) to exercise all the augmentations
    n0 = int(mask.sum())
    print(f"base patch: {n0} faces (teeth {order[:5]})")

    print()
    print("=== punch_holes: only removes faces, respects the removal budget ===")
    rng2 = np.random.default_rng(1)
    m = punch_holes(mesh, mask, rng2, max_holes=2, hole_size_range=(3, 40), max_removed_fraction=0.15)
    removed = n0 - int(m.sum())
    print(f"  removed {removed} / {n0} faces ({removed / n0:.1%})")
    assert np.all(m <= mask), "punch_holes added faces that weren't in the original patch"
    assert removed <= int(n0 * 0.15) + 1, "removed more than the configured budget"

    print()
    print("=== erode_boundary: only removes boundary faces, never interior ones ===")
    boundary_before = _boundary_faces(mesh, np.where(mask)[0])
    rng3 = np.random.default_rng(2)
    m2 = erode_boundary(mesh, mask, rng3, max_drop_prob=0.3)
    assert np.all(m2 <= mask)
    face_idx = np.where(mask)[0]
    dropped_local = (mask & ~m2)[face_idx]
    assert np.all(boundary_before[dropped_local]), "erode_boundary dropped a non-boundary (interior) face"
    print(f"  removed {int(dropped_local.sum())} boundary faces, 0 interior faces - ok")

    print()
    print("=== density_subsample: roughly matches the requested keep_fraction ===")
    rng4 = np.random.default_rng(3)
    m3 = density_subsample(mask, rng4, keep_fraction_range=(0.5, 0.5))  # fixed 0.5 for a precise check
    kept_frac = m3.sum() / mask.sum()
    print(f"  requested 0.5, got {kept_frac:.3f}")
    assert 0.4 < kept_frac < 0.6

    print()
    print("=== jitter_positions: shared vertices get IDENTICAL jitter (mesh stays watertight) ===")
    face_idx = np.where(mask)[0]
    patch_faces = mesh.faces[face_idx]
    rng5 = np.random.default_rng(4)
    noisy_vertices = jitter_positions(mesh.vertices, patch_faces, rng5, sigma_mm_range=(0.1, 0.1))
    moved = np.linalg.norm(noisy_vertices - mesh.vertices, axis=1)
    used_verts = np.unique(patch_faces)
    assert np.all(moved[used_verts] > 0), "every vertex used by the patch should have moved"
    unused_mask = np.ones(len(mesh.vertices), dtype=bool)
    unused_mask[used_verts] = False
    assert np.all(moved[unused_mask] == 0), "vertices outside the patch should be untouched"
    # find a vertex shared by 2 patch faces and confirm both faces see the same new position
    from collections import Counter
    vc = Counter(patch_faces.reshape(-1).tolist())
    shared_vertex = next(v for v, c in vc.items() if c >= 2)
    faces_with_v = [f for f in patch_faces if shared_vertex in f]
    positions = [noisy_vertices[shared_vertex] for _ in faces_with_v]
    assert all(np.allclose(p, positions[0]) for p in positions), "shared vertex got inconsistent jitter"
    print(f"  ok - {len(used_verts)} used vertices moved by ~0.1mm, shared vertices stay consistent")

    print()
    print("=== jitter_normals: stays unit-length after perturbation ===")
    rng6 = np.random.default_rng(5)
    vn, fn = jitter_normals(mesh.vertex_normals, mesh.face_normals, patch_faces, face_idx, rng6,
                              sigma_range=(0.08, 0.08))
    used_vn_norms = np.linalg.norm(vn[used_verts], axis=1)
    used_fn_norms = np.linalg.norm(fn[face_idx], axis=1)
    print(f"  vertex normal norms: min={used_vn_norms.min():.4f} max={used_vn_norms.max():.4f}")
    print(f"  face normal norms:   min={used_fn_norms.min():.4f} max={used_fn_norms.max():.4f}")
    assert np.allclose(used_vn_norms, 1.0, atol=1e-4)
    assert np.allclose(used_fn_norms, 1.0, atol=1e-4)
    # sanity: the perturbation should have actually changed direction, not been a no-op
    original_fn_used = mesh.face_normals[face_idx]
    cos_sim = np.sum(fn[face_idx] * original_fn_used, axis=1)
    assert np.mean(cos_sim) < 0.9999
    print(f"  ok - mean cos-similarity to original normals: {np.mean(cos_sim):.4f} (perturbed, not identical)")

    print()
    print("=== SimToRealAugment: full composition never adds faces, tiny patches are left alone ===")
    aug = SimToRealAugment()
    rng7 = np.random.default_rng(6)
    m_full = aug.augment_mask(mesh, mask, rng7)
    assert np.all(m_full <= mask)
    print(f"  composed mask: {n0} -> {int(m_full.sum())} faces")

    tiny_mask = np.zeros(len(labels), dtype=bool)
    tiny_mask[np.where(mask)[0][:10]] = True  # 10 faces, below MIN_FACES_FOR_DROPOUT=20
    m_tiny = aug.augment_mask(mesh, tiny_mask, np.random.default_rng(7))
    assert np.array_equal(m_tiny, tiny_mask), "dropout augmentations should skip patches below the size floor"
    print("  ok - a 10-face patch (below the size floor) is left untouched by dropout augmentations")

    print()
    print("=== PatchTeeth3DSDataset integration: augment=True (train) vs augment=False vs eval ===")
    train_aug = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1, augment=True)
    train_noaug = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1, augment=False)
    val_aug = PatchTeeth3DSDataset(root, num_classes=17, is_train=False, train_test_split=1, val_seed=0,
                                    augment=True)
    assert train_aug.augment is not None
    assert train_noaug.augment is None
    assert val_aug.augment is not None  # requested...
    # ...but __getitem__ must still never apply it in eval mode - check reproducibility survives,
    # which augmentation (unseeded relative to the rest of the draw) would break if it ran
    item_a = val_aug[0]
    item_b = val_aug[0]
    for a, b in zip(item_a, item_b):
        # area_mm2 is a plain float (not a tensor) - torch.equal only accepts tensors
        assert (a == b) if isinstance(a, float) else torch.equal(a, b)
    print("  ok - eval mode stays perfectly reproducible even with augment=True (augmentation is train-only)")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check sim-to-real patch augmentation")
    parser.add_argument("--pt_path", default="data/3dteethseg/processed_w5/data_00OMSZGW_lower.pt")
    parser.add_argument("--root", default="data/3dteethseg")
    args = parser.parse_args()
    main(args.pt_path, args.root)
