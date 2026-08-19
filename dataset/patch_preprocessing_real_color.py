import torch

from dataset.patch_preprocessing import PatchPreTransform

# New, additive-only file - dataset/patch_preprocessing.py is untouched. Sibling to
# dataset/patch_preprocessing_color.py's PatchPreTransformWithColor, for the LIVE INFERENCE path
# instead of training: that class synthesizes its own per-class-group RGB from ground-truth
# labels (needed at train time since Teeth3DS carries no real color - see
# dataset/patch_color_augmentation.py's own docstring). At inference time there IS real captured
# color from the scanner and no ground truth to synthesize from anyway, so this takes real
# per-face RGB as an explicit argument instead. Same 24-base-dims + 3-color-dims layout and the
# same 0-255 -> [-1,1] normalization as the training-time transform, so a model trained with
# PatchPreTransformWithColor sees a matching input distribution at inference time regardless of
# which of the two color sources (synthetic vs real) produced it.


class PatchPreTransformWithRealColor:
    def __init__(self, scale_mm: float = 15.0, classes: int = 17):
        self._base = PatchPreTransform(scale_mm=scale_mm, classes=classes)

    def __call__(self, data, face_colors):
        """face_colors: (F, 3) uint8 real per-face RGB, F matching data's face count."""
        pos, x24, labels = self._base(data)

        s = x24.shape[0]
        x = torch.zeros(s, 27, dtype=x24.dtype)
        x[:, :24] = x24
        x[:, 24:27] = torch.from_numpy(face_colors).to(x24.dtype) / 127.5 - 1.0

        return pos, x, labels
