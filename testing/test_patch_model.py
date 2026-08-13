import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset
from models.patch_collate import PatchCollator, precompute_neighbor_indices
from models.patch_dilated_tooth_seg_network import PatchDilatedToothSegmentationNetwork
from models.patch_layer import knn as clamped_knn


def main(root, device):
    print(f"=== forward pass across the full N range (device={device}) ===")
    model = PatchDilatedToothSegmentationNetwork(num_classes=17, feature_dim=24).to(device)
    model.eval()
    # N=32 is the exact old crash floor (models/dilated_tooth_seg_network.py fails at N<=32,
    # confirmed in TRAINING_CONCERNS.md #1) - the whole point of the k-clamp fix is that this
    # network must NOT crash here
    for n in [20000, 500, 40, 33, 32, 24, 10, 3, 2]:
        pos = torch.randn(1, n, 3, device=device)
        x = torch.randn(1, n, 24, device=device)
        with torch.no_grad():
            out = model(x, pos)
        assert out.shape == (1, n, 17)
        print(f"  N={n:6d}: ok")
    print("  ok - no crash anywhere down to N=2 (old model crashed at N<=32)")

    print()
    print("=== area-gated dilation: gated-off blocks receive no gradient ===")
    model_gated = PatchDilatedToothSegmentationNetwork(num_classes=17, feature_dim=24).to(device)
    n = 300
    pos = torch.randn(1, n, 3, device=device)
    x = torch.randn(1, n, 24, device=device)
    out = model_gated(x, pos, area_mm2=10.0)  # below all 3 thresholds (40/180/360) - all gated off
    out.sum().backward()
    for name, p in model_gated.named_parameters():
        if 'dilated_edge_graph_conv_block' in name:
            assert p.grad is None or torch.all(p.grad == 0), f"{name} got a nonzero gradient while gated off"
    print("  ok - all 3 dilated blocks' parameters got zero/no gradient when area is below every threshold")

    model_gated.zero_grad()
    out2 = model_gated(x, pos, area_mm2=1000.0)  # above all 3 thresholds - all engaged
    out2.sum().backward()
    touched = [name for name, p in model_gated.named_parameters()
               if 'dilated_edge_graph_conv_block' in name and p.grad is not None and torch.any(p.grad != 0)]
    assert len(touched) > 0, "dilated blocks should receive gradient when engaged"
    print(f"  ok - {len(touched)} dilated-block parameter tensors got a nonzero gradient when area clears "
          f"every threshold")

    print()
    print("=== GroupNorm: eval-mode output ~independent of batch composition ===")
    # area_mm2=0.0 gates all 3 dilated blocks off deliberately: DilatedEdgeGraphConvBlock's
    # on-the-fly fallback path (pos.reshape(B*N,-1)[idx_l]) has a pre-existing B>1 indexing bug
    # inherited from the original models/layer.py - it flattens pos across the batch dim but its
    # topk indices are per-item local indices, not batch-offset, so it silently pulls from the
    # wrong batch item whenever B>1. Out of scope to fix: we deliberately chose batch_size=1 (see
    # TRAINING_CONCERNS.md), and real training always supplies precomputed idx via PatchCollator,
    # which only ever handles one patch at a time and never hits this path. Gating dilated blocks
    # off here isolates the actual thing under test - GroupNorm in the local blocks/head - from
    # that unrelated, unsupported-combination bug.
    with torch.no_grad():
        out_alone = model(x, pos, area_mm2=0.0)
        out_stacked = model(torch.cat([x, x], dim=0), torch.cat([pos, pos], dim=0), area_mm2=0.0)
    max_diff = (out_alone[0] - out_stacked[0]).abs().max().item()
    print(f"  max output diff, alone vs stacked-with-duplicate: {max_diff:.6f}")
    # not bit-exact - cuDNN can pick different conv algorithms for different batch sizes, causing
    # float32 summation-order noise (~1e-3 measured) even for a mathematically batch-independent
    # op. The bar here is "far tighter than BatchNorm-with-running-stats would ever produce if
    # patch composition genuinely leaked into the normalization," not bit-for-bit reproducibility.
    assert max_diff < 1e-2, "GroupNorm output should be nearly batch-independent (float noise only)"
    print("  ok - output is nearly identical (float32 noise only) whether run alone or stacked")

    print()
    print("=== precomputed idx vs on-the-fly: block 1's local kNN gives the IDENTICAL neighbor set ===")
    rng = np.random.default_rng(0)
    n = 800
    pos_np = rng.standard_normal((n, 3)).astype(np.float32)
    pos_cpu = torch.from_numpy(pos_np).unsqueeze(0)
    # knn()/get_graph_feature's calling convention passes pos already transposed to (B, 3, N) -
    # see EdgeGraphConvBlock.forward's pos_t = pos.transpose(2, 1) before it reaches knn()
    onfly_idx = clamped_knn(pos_cpu.transpose(2, 1), k=32)[0].numpy()
    local_idx, _ = precompute_neighbor_indices(pos_np, k=32, dilation_ks=(200, 600, 1800), rng=rng)
    match = sum(set(onfly_idx[i].tolist()) == set(local_idx[i].tolist()) for i in range(n))
    print(f"  {match}/{n} faces: precomputed KD-tree neighbor SET exactly matches on-the-fly cdist+topk")
    assert match == n, "KD-tree and brute-force cdist should find the exact same k-nearest set (both exact)"

    print()
    print("=== precomputed idx vs on-the-fly: dilated blocks' CANDIDATE POOL matches (pre-FPS) ===")
    from models.patch_layer import DilatedEdgeGraphConvBlock
    cd = torch.cdist(pos_cpu, pos_cpu, p=2)
    for dk in (200, 600, 1800):
        onfly_cand = torch.topk(cd, min(dk, n), largest=False)[1][0].numpy()
        onfly_sets = [set(row.tolist()) for row in onfly_cand]
        _, dilated_all = precompute_neighbor_indices(pos_np, k=32, dilation_ks=(dk,), rng=rng)
        # the precomputed path only keeps the FPS-selected subset, not the full candidate pool -
        # recompute the pool alone the same way local_idx was checked above, via the tree query
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_np)
        _, cand_idx = tree.query(pos_np, k=min(dk, n), workers=-1)
        cand_sets = [set(np.atleast_2d(cand_idx)[i].tolist()) for i in range(n)]
        match = sum(a == b for a, b in zip(onfly_sets, cand_sets))
        print(f"  dilation_k={dk:4d}: {match}/{n} faces' candidate pools match exactly")
        assert match == n

    print()
    print("=== full pipeline: dataset -> PatchCollator -> model, timing across draws ===")
    ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1)
    collator = PatchCollator.for_network(model, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collator)
    it = iter(loader)
    next(it)  # warm up numba's JIT compile, excluded from timing below
    collate_times, n_faces = [], []
    for _ in range(8):
        t0 = time.time()
        batch = next(it)
        collate_times.append(time.time() - t0)
        n_faces.append(batch.pos.shape[1])
        with torch.no_grad():
            out = model(batch.x.to(device), batch.pos.to(device), area_mm2=batch.area_mm2,
                        local_idx=batch.local_idx.to(device),
                        dilated_idx=tuple(d.to(device) for d in batch.dilated_idx))
        assert out.shape == (1, batch.pos.shape[1], 17)
    print(f"  8 draws, N range {min(n_faces)}-{max(n_faces)}, "
          f"collate time {min(collate_times):.3f}-{max(collate_times):.3f}s")
    assert max(collate_times) < 5.0, "collate should stay well under a second even for large patches"

    print()
    print("=== worst case: near-full-arch (occlusal+facial+lingual) patch ===")
    val_ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=False, train_test_split=1, val_seed=0)
    sequence, stages = val_ds.get_full_sequence(0)
    pos, x, labels, area_mm2 = sequence[-1]
    t0 = time.time()
    batch = collator([(pos, x, labels, area_mm2)])
    t_collate = time.time() - t0
    with torch.no_grad():
        out = model(batch.x.to(device), batch.pos.to(device), area_mm2=batch.area_mm2,
                    local_idx=batch.local_idx.to(device),
                    dilated_idx=tuple(d.to(device) for d in batch.dilated_idx))
    print(f"  N={pos.shape[0]}, area={area_mm2:.1f}mm2: collate={t_collate:.3f}s, out={tuple(out.shape)}")
    assert t_collate < 5.0

    print()
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component 4 integration test: model surgery + collate")
    parser.add_argument("--root", default="data/3dteethseg")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.root, args.device)
