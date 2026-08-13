import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_generator import grow_patch_full_sweep, load_cached_arch, pick_incisor_seed_label
from dataset.patch_preprocessing import PatchPreTransform
from dataset.preprocessing import PreTransform


def local_positions(pos, face_idx, common_idx):
    """Row-index common_idx (global face ids) within a patch's own local (0..N-1) ordering."""
    local = np.searchsorted(face_idx, common_idx)
    assert np.array_equal(face_idx[local], common_idx), "common_idx wasn't actually a subset of face_idx"
    return pos[local]


def main(root):
    # transform=None: this test applies (and compares) transforms manually, so it needs the raw
    # 5-tuples from _stage_tensors, not PatchTeeth3DSDataset's now-default PatchPreTransform output
    ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=False, train_test_split=1, val_seed=0,
                               transform=None)
    pt_path = os.path.join(ds.root, ds.processed_folder, ds.file_names[0])
    mesh, labels = load_cached_arch(pt_path)
    rng = np.random.default_rng(ds.val_seed + 0)
    seed_label = pick_incisor_seed_label(labels, rng)
    masks, stages = grow_patch_full_sweep(mesh, labels, seed_label, rng, ds.width_mm, ds.height_mm,
                                           ds.slab_mm, ds.min_occlusality)
    print(f"arch 0: {len(masks)} stages, seed tooth {seed_label}")

    small_i, large_i = 0, len(masks) - 1
    small_mask, large_mask = masks[small_i], masks[large_i]
    common_mask = small_mask & large_mask
    small_face_idx = np.where(small_mask)[0]
    large_face_idx = np.where(large_mask)[0]
    common_idx = np.where(common_mask)[0]
    print(f"stage {small_i} {stages[small_i]}: {small_mask.sum()} faces")
    print(f"stage {large_i} {stages[large_i]}: {large_mask.sum()} faces")
    print(f"faces present in BOTH (the ones we can fairly compare): {len(common_idx)}")
    assert len(common_idx) > 50, "need a reasonable number of shared faces to test with"

    # _stage_tensors returns the raw 5-tuple + area_mm2 appended (Component 4's area-gating
    # input) - split it back apart since PatchPreTransform/PreTransform below still expect
    # exactly the 5-tuple
    small_stage_full = ds._stage_tensors(mesh, labels, small_mask)
    large_stage_full = ds._stage_tensors(mesh, labels, large_mask)
    small_stage, small_area_mm2 = small_stage_full[:5], small_stage_full[5]
    large_stage, large_area_mm2 = large_stage_full[:5], large_stage_full[5]
    print(f"area_mm2: small stage={small_area_mm2:.1f}  large stage={large_area_mm2:.1f}")
    assert large_area_mm2 > small_area_mm2, "the larger (later) stage should have more real area"

    new_transform = PatchPreTransform(scale_mm=15.0)
    pos_small, x_small, labels_small = new_transform(small_stage)
    pos_large, x_large, labels_large = new_transform(large_stage)

    assert torch.equal(pos_small, x_small[:, 9:12])
    assert x_small.shape[1] == 24
    print(f"\nshapes ok: x_small={tuple(x_small.shape)}  x_large={tuple(x_large.shape)}")

    mesh_vertices_normals_small = small_stage[2]
    assert torch.equal(x_small[:, 12:15], mesh_vertices_normals_small[:, 0])
    mesh_face_normals_small = small_stage[3]
    assert torch.equal(x_small[:, 21:], mesh_face_normals_small)
    print("normals confirmed raw (unmodified from input)")

    print("\n=== PatchPreTransform: pairwise distances among the SAME (shared) faces ===")
    common_pos_small = local_positions(pos_small, small_face_idx, common_idx)
    common_pos_large = local_positions(pos_large, large_face_idx, common_idx)
    d_small = torch.cdist(common_pos_small, common_pos_small)
    d_large = torch.cdist(common_pos_large, common_pos_large)
    max_diff = (d_small - d_large).abs().max().item()
    print(f"  max |pairwise distance difference| between small-patch and large-patch versions: {max_diff:.6f}")
    # threshold is in normalized (/15mm) units - 1e-2 here is ~0.15mm in real space, far below a
    # single mesh triangle's size; anything under that is float32 rounding, not a real difference
    assert max_diff < 1e-2, "pairwise distances among the SAME faces should be patch-size-invariant"
    print("  ok - essentially identical (a different centroid is just a translation, doesn't "
          "affect pairwise distance; fixed SCALE_MM means scale doesn't shift either; the tiny "
          "residual is float32 rounding from subtracting differently-sized centroids)")

    print("\n=== contrast: dataset/preprocessing.py's PreTransform (untouched, whole-PATCH stats here) ===")
    old_transform = PreTransform(classes=17)
    pos_small_old, _, _ = old_transform(small_stage)
    pos_large_old, _, _ = old_transform(large_stage)
    common_pos_small_old = local_positions(pos_small_old, small_face_idx, common_idx)
    common_pos_large_old = local_positions(pos_large_old, large_face_idx, common_idx)
    d_small_old = torch.cdist(common_pos_small_old, common_pos_small_old)
    d_large_old = torch.cdist(common_pos_large_old, common_pos_large_old)
    max_diff_old = (d_small_old - d_large_old).abs().max().item()
    print(f"  max |pairwise distance difference| with the OLD per-patch-stats scheme: {max_diff_old:.6f}")
    print(f"  ({max_diff_old / max(max_diff, 1e-12):.0f}x larger than PatchPreTransform's - same "
          f"faces, different apparent scale depending on how much else is in the patch)")
    assert max_diff_old > max_diff, "expected the old whole-mesh-style scheme to be far less consistent"

    print("\n=== scale sanity: a real 15mm offset should normalize to ~1.0 ===")
    fake_data = (
        small_stage[0][:2],
        torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                      [[15., 0., 0.], [16., 0., 0.], [15., 1., 0.]]]),
        small_stage[2][:2],
        small_stage[3][:2],
        small_stage[4][:2],
    )
    pos_fake, x_fake, _ = new_transform(fake_data)
    offset = (pos_fake[1] - pos_fake[0]).norm().item()
    print(f"  two face-centers 15mm apart in real space -> {offset:.3f} in normalized space (expect ~1.0)")
    assert abs(offset - 1.0) < 0.2

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check PatchPreTransform's patch-size invariance")
    parser.add_argument("--root", default="data/3dteethseg")
    args = parser.parse_args()
    main(args.root)
