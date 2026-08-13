import argparse
import time

import numpy as np
import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_losses import FocalLoss
from models.patch_collate import PatchCollator
from models.patch_dilated_tooth_seg_network import PatchDilatedToothSegmentationNetwork

# Estimates a real training iteration's wall-clock cost (collate + forward + loss + backward +
# optimizer step) on ACTUAL patches drawn from the real training distribution - not a synthetic
# guess. No Lightning training loop exists yet (Component 5), so this is a standalone benchmark
# using the same pieces one would go into, run directly.


def main(root, n_iters, device, num_workers):
    ds = PatchTeeth3DSDataset(root, num_classes=17, is_train=True, train_test_split=1)
    print(f"{len(ds)} arches in the training split -> {len(ds)} iterations/epoch under the "
          f"current __getitem__ (one random patch per arch per call)")

    model = PatchDilatedToothSegmentationNetwork(num_classes=17, feature_dim=24).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters, num_workers={num_workers}")

    collator = PatchCollator.for_network(model)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers,
                                          collate_fn=collator,
                                          persistent_workers=(num_workers > 0),
                                          prefetch_factor=4 if num_workers > 0 else None)

    loss_fn = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)

    it = iter(loader)
    n_warmup = 5 if num_workers == 0 else 2 * num_workers  # let worker processes spin up and fill the prefetch queue
    print(f"\nwarming up ({n_warmup} iters: JIT compile, cuDNN autotune, worker startup) - excluded from timing...")
    for _ in range(n_warmup):
        batch = next(it)
        x, pos, labels = batch.x.to(device), batch.pos.to(device), batch.labels.to(device)
        local_idx = batch.local_idx.to(device)
        dilated_idx = tuple(d.to(device) for d in batch.dilated_idx)
        out = model(x, pos, area_mm2=batch.area_mm2, local_idx=local_idx, dilated_idx=dilated_idx)
        loss = loss_fn(out.transpose(2, 1), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"timing {n_iters} real iterations...\n")
    records = []
    for i in range(n_iters):
        t0 = time.time()
        batch = next(it)
        if device == "cuda":
            torch.cuda.synchronize()
        t_collate = time.time() - t0

        t0 = time.time()
        x, pos, labels = batch.x.to(device), batch.pos.to(device), batch.labels.to(device)
        local_idx = batch.local_idx.to(device)
        dilated_idx = tuple(d.to(device) for d in batch.dilated_idx)
        out = model(x, pos, area_mm2=batch.area_mm2, local_idx=local_idx, dilated_idx=dilated_idx)
        loss = loss_fn(out.transpose(2, 1), labels)
        if device == "cuda":
            torch.cuda.synchronize()
        t_forward = time.time() - t0

        t0 = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
        t_backward = time.time() - t0

        total = t_collate + t_forward + t_backward
        records.append((pos.shape[1], t_collate, t_forward, t_backward, total))
        print(f"  [{i:3d}] N={pos.shape[1]:6d}  collate={t_collate:.3f}s  forward+loss={t_forward:.3f}s  "
              f"backward+step={t_backward:.3f}s  total={total:.3f}s")

    ns = np.array([r[0] for r in records])
    totals = np.array([r[4] for r in records])
    collates = np.array([r[1] for r in records])
    fwds = np.array([r[2] for r in records])
    bwds = np.array([r[3] for r in records])

    print(f"\n=== summary over {n_iters} real iterations (N range {ns.min()}-{ns.max()}) ===")
    print(f"  collate:      mean={collates.mean():.3f}s  median={np.median(collates):.3f}s  max={collates.max():.3f}s")
    print(f"  forward+loss: mean={fwds.mean():.3f}s  median={np.median(fwds):.3f}s  max={fwds.max():.3f}s")
    print(f"  backward+step:mean={bwds.mean():.3f}s  median={np.median(bwds):.3f}s  max={bwds.max():.3f}s")
    print(f"  TOTAL/iter:   mean={totals.mean():.3f}s  median={np.median(totals):.3f}s  max={totals.max():.3f}s")

    epoch_len = len(ds)
    mean_epoch = totals.mean() * epoch_len
    median_epoch = np.median(totals) * epoch_len
    print(f"\n=== extrapolated to 1 epoch ({epoch_len} iterations) ===")
    print(f"  using mean/iter:   {mean_epoch:.0f}s  (~{mean_epoch / 60:.1f} min)")
    print(f"  using median/iter: {median_epoch:.0f}s  (~{median_epoch / 60:.1f} min)")
    print(f"  (mean is pulled up by the rare huge full-arch patches; median is more representative "
          f"of a typical step, given early_bias_power biases sampling toward small/early patches)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark real training-iteration wall-clock time")
    parser.add_argument("--root", default="data/3dteethseg")
    parser.add_argument("--n_iters", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args.root, args.n_iters, args.device, args.num_workers)
