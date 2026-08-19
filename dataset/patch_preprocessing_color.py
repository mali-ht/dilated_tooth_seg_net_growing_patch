import hashlib

import numpy as np
import torch

from dataset.patch_color_augmentation import SyntheticColorPaint
from dataset.patch_preprocessing import PatchPreTransform

# New, additive-only file - dataset/patch_preprocessing.py is untouched. Composes with the
# existing PatchPreTransform (calls it as-is for the original 24-dim geometry layout) rather than
# reimplementing or editing it, then appends a synthetic 3-dim per-face RGB channel
# (SyntheticColorPaint, dataset/patch_color_augmentation.py) for a 27-dim feature vector.
#
# Same (mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels) 5-tuple
# call signature as the original PatchPreTransform, so this drops straight into
# PatchTeeth3DSDataset's existing `transform=` constructor argument
# (dataset/patch_dataset.py:_DEFAULT_TRANSFORM handling) with no changes needed there either - no
# new dataset subclass is required for this feature at all.


class PatchPreTransformWithColor:
    def __init__(self, scale_mm: float = 15.0, classes: int = 17):
        self._base = PatchPreTransform(scale_mm=scale_mm, classes=classes)
        self._paint = SyntheticColorPaint()

    def __call__(self, data):
        pos, x24, labels = self._base(data)

        # Seed derived from this specific patch's own label content, NOT from
        # PatchTeeth3DSDataset's per-index RNG (dataset/patch_dataset.py's _rng_for) - threading
        # that RNG in here would require changing that file's call site
        # (self.transform(raw)), which this file deliberately never touches. This still gives
        # both properties that RNG would have given: the SAME synthetic color for the SAME patch
        # every time (train or eval), which matches eval mode's fixed/reproducible validation set
        # since a validation patch's content never changes between epochs; and naturally varied
        # color across different patches/epochs in train mode, since two different-content
        # patches essentially never hash to the same seed.
        seed = int(hashlib.sha256(labels.numpy().tobytes()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        face_colors = self._paint(labels.numpy(), rng)

        s = x24.shape[0]
        x = torch.zeros(s, 27, dtype=x24.dtype)
        x[:, :24] = x24
        # 0-255 uint8 -> [-1, 1], roughly matching the scale of the rest of the feature vector
        # (unit normals, positions divided by a fixed mm scale) instead of leaving color at a
        # much larger 0-255 magnitude that would dominate the first layer's weights.
        x[:, 24:27] = torch.from_numpy(face_colors).to(x24.dtype) / 127.5 - 1.0

        return pos, x, labels
