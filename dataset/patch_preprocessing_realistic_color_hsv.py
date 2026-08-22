import hashlib

import numpy as np
import torch

from dataset.patch_color_augmentation import RealisticColorPaintHSV
from dataset.patch_preprocessing import PatchPreTransform

# New, additive-only file - dataset/patch_preprocessing.py, dataset/patch_preprocessing_color.py
# and dataset/patch_preprocessing_realistic_color.py are all untouched. Sibling to that last file's
# PatchPreTransformWithRealisticColor, swapping in RealisticColorPaintHSV (correlated HSV-space
# per-face color, dataset/patch_color_augmentation.py) instead of RealisticColorPaint (independent
# per-channel RGB noise - confirmed 2026-08-22 to occasionally produce implausible saturated
# colors like purple/magenta "enamel"). Same composition pattern: calls the existing
# PatchPreTransform as-is for the 24-dim geometry layout, appends a 3-dim color channel for 27
# total - identical structure to PatchPreTransformWithRealisticColor, only the paint class differs.


class PatchPreTransformWithRealisticColorHSV:
    def __init__(self, scale_mm: float = 15.0, classes: int = 17):
        self._base = PatchPreTransform(scale_mm=scale_mm, classes=classes)
        self._paint = RealisticColorPaintHSV()

    def __call__(self, data):
        pos, x24, labels = self._base(data)

        # Same content-derived-seed reasoning as PatchPreTransformWithColor/
        # PatchPreTransformWithRealisticColor - reproducible for the SAME patch (eval mode's fixed
        # validation set), even though every face gets an independent draw.
        seed = int(hashlib.sha256(labels.numpy().tobytes()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        face_colors = self._paint(labels.numpy(), rng)

        s = x24.shape[0]
        x = torch.zeros(s, 27, dtype=x24.dtype)
        x[:, :24] = x24
        x[:, 24:27] = torch.from_numpy(face_colors).to(x24.dtype) / 127.5 - 1.0

        return pos, x, labels
