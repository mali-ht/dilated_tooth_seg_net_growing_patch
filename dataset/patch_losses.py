import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al. 2017), generalizing nn.CrossEntropyLoss with a
    (1 - p_t)^gamma focusing term that down-weights already-easy, confidently-correct
    predictions - TRAINING_CONCERNS.md #5: gum was 60-70% of many patches in earlier testing, so
    without this an easy majority-class shortcut ("predict gum, get a good loss") could
    substitute for actually learning tooth-type/boundary distinctions. gamma=0 with alpha=None
    reduces exactly to plain cross-entropy (see testing/test_patch_losses.py) - gamma=2.0 is the
    standard default from the paper and a reasonable starting point here.

    Same calling convention as nn.CrossEntropyLoss for per-point classification: logits
    (B, C, N, ...) and targets (B, N, ...) of dtype long - a drop-in replacement for
    `self.loss = nn.CrossEntropyLoss()` in models/dilated_tooth_seg_network.py's
    LitDilatedToothSegmentationNetwork (that file is untouched; this is for whenever a
    patch-training-pipeline-specific Lightning module is built on top of it).
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean', ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        targets = targets.long()
        valid = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid] = 0  # dummy in-range index; masked out of the loss below

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets_safe]
            loss = loss * alpha_t

        loss = loss[valid]
        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        if self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none' - flattened to just the valid elements, not the original shape


def compute_class_alpha(dataset, num_classes, n_samples=200, seed=0, normalize=True):
    """
    Inverse-class-frequency alpha weights, measured empirically from the REAL patch-sampling
    distribution `dataset` (a train-mode PatchTeeth3DSDataset) actually produces - not whole-arch
    frequency, which would misrepresent how skewed a typical PATCH is (see TRAINING_CONCERNS.md
    #5's 60-70% gum figure). `seed` fixes WHICH arch indices get sampled (via this function's own
    local rng) - it does NOT make the run bit-reproducible end to end: dataset[i] on a train-mode
    PatchTeeth3DSDataset draws its own fresh, unseeded randomness every call (seed tooth, sweep
    side, min_occlusality, augmentation - see PatchTeeth3DSDataset._rng_for), by design, so the
    SAME index can return a very differently-sized/composed patch on two different calls. Two
    runs with the same seed will therefore NOT produce identical alpha - only a large enough
    n_samples makes the AGGREGATE class-frequency ratios stable run-to-run (confirmed: n_samples=20
    gave wildly different alpha across two runs in testing; n_samples=200, the default, is the
    intended, more statistically stable setting). normalize=True rescales so mean(alpha) == 1,
    keeping the overall loss magnitude comparable to unweighted cross-entropy instead of just
    shrinking or inflating it.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    py_rng = np.random.default_rng(seed)
    indices = py_rng.integers(0, len(dataset), size=n_samples)
    for i in indices:
        _, _, labels, _ = dataset[int(i)]
        binc = torch.bincount(labels.view(-1), minlength=num_classes).numpy()
        counts += binc[:num_classes]

    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for classes absent from the sample
    alpha = 1.0 / counts
    if normalize:
        alpha = alpha * (num_classes / alpha.sum())
    return torch.as_tensor(alpha, dtype=torch.float32)
