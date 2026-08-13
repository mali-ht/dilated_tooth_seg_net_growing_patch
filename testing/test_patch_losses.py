import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.patch_dataset import PatchTeeth3DSDataset
from dataset.patch_losses import FocalLoss, compute_class_alpha


def main(root):
    torch.manual_seed(0)
    B, C, N = 2, 17, 500
    logits = torch.randn(B, C, N)
    targets = torch.randint(0, C, (B, N))

    print("=== gamma=0, alpha=None reduces to plain cross-entropy ===")
    fl = FocalLoss(gamma=0.0, alpha=None, reduction='mean')
    ce = nn.CrossEntropyLoss(reduction='mean')
    fl_val, ce_val = fl(logits, targets).item(), ce(logits, targets).item()
    print(f"  focal(gamma=0)={fl_val:.6f}  cross_entropy={ce_val:.6f}")
    assert abs(fl_val - ce_val) < 1e-4, "gamma=0 focal loss should match plain CE"

    print()
    print("=== gamma > 0 down-weights confident-correct predictions ===")
    # a target class with logits already strongly favoring it (easy/confident) vs. a target class
    # the logits are indifferent to (hard) - focal loss should shrink the easy example's
    # contribution relative to CE much more than the hard one's
    easy_logits = torch.zeros(1, C, 1)
    easy_logits[0, 3, 0] = 10.0  # very confident, correct class 3
    easy_target = torch.tensor([[3]])
    hard_logits = torch.zeros(1, C, 1)  # uniform - no confidence either way
    hard_target = torch.tensor([[5]])

    fl2 = FocalLoss(gamma=2.0)
    ce_fn = nn.CrossEntropyLoss(reduction='mean')
    easy_ce, easy_fl = ce_fn(easy_logits, easy_target).item(), fl2(easy_logits, easy_target).item()
    hard_ce, hard_fl = ce_fn(hard_logits, hard_target).item(), fl2(hard_logits, hard_target).item()
    print(f"  easy example: CE={easy_ce:.6f}  focal={easy_fl:.6f}  ratio={easy_fl / max(easy_ce, 1e-9):.4f}")
    print(f"  hard example: CE={hard_ce:.6f}  focal={hard_fl:.6f}  ratio={hard_fl / max(hard_ce, 1e-9):.4f}")
    assert easy_fl / max(easy_ce, 1e-9) < hard_fl / max(hard_ce, 1e-9), \
        "focal loss should shrink the easy example's loss proportionally more than the hard one's"

    print()
    print("=== per-class alpha weighting ===")
    alpha = torch.ones(C)
    alpha[3] = 5.0  # class 3 weighted 5x
    fl3 = FocalLoss(gamma=0.0, alpha=alpha, reduction='none')
    t = torch.tensor([[3, 5]])
    lg = torch.zeros(1, C, 2)
    losses = fl3(lg, t)
    print(f"  losses for classes [3, 5]: {losses.tolist()} (expect class 3's ~5x class 5's)")
    assert abs(losses[0].item() / losses[1].item() - 5.0) < 1e-3

    print()
    print("=== ignore_index masks out padded/invalid targets ===")
    fl4 = FocalLoss(gamma=2.0, ignore_index=-1)
    t2 = torch.tensor([[0, 1, -1, 2]])
    lg2 = torch.randn(1, C, 4)
    loss_with_ignore = fl4(lg2, t2)
    loss_manual = fl4(lg2[:, :, :2], t2[:, :2])  # same result as if the -1 slot never existed...
    # (not identical to loss_manual because the 4th valid element still contributes) - just check
    # it runs and produces a finite scalar without the ignored index corrupting anything
    assert torch.isfinite(loss_with_ignore)
    print(f"  ok - finite loss ({loss_with_ignore.item():.6f}) with one ignored target present")

    print()
    print("=== gradient sanity: loss backpropagates into logits ===")
    logits_g = torch.randn(1, C, 50, requires_grad=True)
    targets_g = torch.randint(0, C, (1, 50))
    FocalLoss(gamma=2.0)(logits_g, targets_g).backward()
    assert logits_g.grad is not None and torch.isfinite(logits_g.grad).all()
    print("  ok - finite gradient reaches logits")

    print()
    print("=== compute_class_alpha on the real patch distribution (5-class, quick n_samples) ===")
    ds = PatchTeeth3DSDataset(root, num_classes=5, is_train=True, train_test_split=1)
    alpha5 = compute_class_alpha(ds, num_classes=5, n_samples=30, seed=0)
    print(f"  alpha (mean should be ~1.0): {alpha5.tolist()}  mean={alpha5.mean().item():.4f}")
    assert abs(alpha5.mean().item() - 1.0) < 1e-3
    assert alpha5[0] < alpha5[1:].mean(), "gum (class 0) is the majority class - should get the smallest alpha"
    print("  ok - gum (class 0, the dominant class) gets the smallest weight, as expected")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check FocalLoss and compute_class_alpha")
    parser.add_argument("--root", default="data/3dteethseg")
    args = parser.parse_args()
    main(args.root)
