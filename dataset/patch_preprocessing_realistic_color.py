import hashlib

import numpy as np
import torch

from dataset.patch_color_augmentation import RealisticColorPaint
from dataset.patch_preprocessing import PatchPreTransform

# New, additive-only file - dataset/patch_preprocessing.py and dataset/patch_preprocessing_color.py
# are both untouched. Sibling to patch_preprocessing_color.py's PatchPreTransformWithColor, using
# RealisticColorPaint (mosaic-style, per-face draws from the MEASURED real scanner color
# distribution - dataset/patch_color_augmentation.py) instead of SyntheticColorPaint (flat,
# one-shared-draw-per-class-group, hand-picked ranges). Same composition pattern: calls the
# existing PatchPreTransform as-is for the 24-dim geometry layout, appends a 3-dim color channel
# for 27 total.


class PatchPreTransformWithRealisticColor:
    def __init__(self, scale_mm: float = 15.0, classes: int = 17):
        self._base = PatchPreTransform(scale_mm=scale_mm, classes=classes)
        self._paint = RealisticColorPaint()

    def __call__(self, data):
        pos, x24, labels = self._base(data)

        # Same content-derived-seed reasoning as PatchPreTransformWithColor - see that file's own
        # comment. Mosaic-style per-face color still needs to be reproducible for the SAME patch
        # (eval mode's fixed validation set), even though every face gets an independent draw.
        seed = int(hashlib.sha256(labels.numpy().tobytes()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        face_colors = self._paint(labels.numpy(), rng)

        s = x24.shape[0]
        x = torch.zeros(s, 27, dtype=x24.dtype)
        x[:, :24] = x24
        x[:, 24:27] = torch.from_numpy(face_colors).to(x24.dtype) / 127.5 - 1.0

        return pos, x, labels
