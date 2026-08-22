import hashlib

import numpy as np
import torch

from dataset.patch_color_augmentation import NO_COLOR_SENTINEL

# New, additive-only file - dataset/patch_preprocessing_color.py and
# dataset/patch_preprocessing_realistic_color.py are both untouched. Neither training color
# transform (flat SyntheticColorPaint or mosaic RealisticColorPaint) ever produces a color-less
# face - every training patch has valid, in-range color on 100% of its faces, confirmed directly
# by reading both transforms. Real live scans do not: analyze_snapshot_colors.py measured 65-88%
# of faces in real runs (2026-08-21) carrying NO real captured color at all (the scanner's own
# wire protocol just doesn't have it yet for that publish - see
# realtime/Windows_server/mesh_intercept.js). A color-trained checkpoint had never once seen that
# during training, and live testing showed it reads NO_COLOR_SENTINEL as far darker than either
# real class range, defaulting to gum almost unconditionally wherever it appears (99.9% gum rate
# on sentinel faces in one live run) - training never gave the model a reason not to trust color
# completely.
#
# This wraps EITHER existing color transform (composition, not a new duplicate transform) and,
# after it runs, overwrites a random fraction of faces' color feature with the exact same sentinel
# value live inference uses - so the model sees color-less patches during training too, and has to
# learn to fall back on geometry (the other 24 feature dims, untouched by this) for those faces
# instead of a shortcut that live reality can't always provide.
#
# Per-face independent dropout (like RealisticColorPaint's per-face color draws), NOT
# spatially-contiguous blobs matching how live color-lessness actually clusters (whole
# not-yet-processed regions, not scattered faces) - a real simplification. If per-face dropout
# doesn't close the gap on the next real-hardware test, a spatially-contiguous version (grown from
# a random seed face similarly to dataset/patch_augmentation.py's punch_holes) is the documented
# next step, not implemented here yet.


class ColorDropoutWrapper:
    """Wraps an inner color transform (PatchPreTransformWithColor or
    PatchPreTransformWithRealisticColor - anything producing (pos, x, labels) with x's last 3
    columns holding [-1, 1]-scaled RGB, matching both of those classes' and the live-inference
    PatchPreTransformWithRealColor's shared normalization) and overwrites dropout_prob's fraction
    of faces' color columns with NO_COLOR_SENTINEL, scaled the same way.
    """

    def __init__(self, inner_transform, dropout_prob: float = 0.6):
        self._inner = inner_transform
        self.dropout_prob = dropout_prob
        # /127.5 - 1.0: same 0-255 -> [-1,1] mapping every color transform in this codebase uses
        # (dataset/patch_preprocessing_color.py, dataset/patch_preprocessing_realistic_color.py,
        # dataset/patch_preprocessing_real_color.py) - computed once here since it's a constant.
        self._sentinel_scaled = torch.from_numpy(NO_COLOR_SENTINEL.astype(np.float32)) / 127.5 - 1.0

    def __call__(self, data):
        pos, x, labels = self._inner(data)

        # Independent draw from a DIFFERENTLY-salted seed than the inner transform's own
        # label-content hash, so which faces get dropped isn't coupled to which color each face
        # happened to be painted - same reproducibility property (fixed for a given patch's label
        # content, so eval mode stays stable across epochs) via the same content-derived-seed
        # pattern the inner transforms already use.
        seed = int(hashlib.sha256(labels.numpy().tobytes() + b'color_dropout').hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        drop_mask = rng.random(x.shape[0]) < self.dropout_prob
        x[drop_mask, 24:27] = self._sentinel_scaled.to(x.dtype)

        return pos, x, labels
