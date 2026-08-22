import numpy as np

# New, additive-only file - dataset/patch_dataset.py, dataset/patch_preprocessing.py and
# dataset/patch_augmentation.py are all untouched. This is the "artificial color" input
# experiment: Teeth3DS's raw .obj files carry no real per-face color at all (verified directly -
# every face of a real sample file loads as identical placeholder gray RGB(128,128,128); ground
# truth always comes from a separate .json sidecar, never mesh color - see Docs/MAA_ToDo.md's
# "Per-face color as a model input" section), so this paints a plausible synthetic color onto each
# training patch FROM its own ground-truth labels, giving the model a color signal to learn from
# that doesn't exist in the source data but does exist on the real, live scanner.
#
# Deliberately ONE color draw per class-GROUP within a patch - every tooth face (label != 0)
# shares one draw, every gum face (label == 0) shares another - not per-face and not per-
# individual-tooth. Independent per-face random draws would look like meaningless mosaic noise no
# real scan ever has; a real patient's teeth are all much closer in shade to each other than to
# gum, so treating "all teeth in this patch" as one color group still gives the model a clean,
# realistic teeth-vs-gum color signal without this augmentation needing to track individual tooth
# identity.

# Same value as realtime/live_preprocessing.py's NO_COLOR_SENTINEL (and
# realtime/analyze_snapshot_colors.py's own inlined copy of it) - duplicated here rather than
# imported for the same reason analyze_snapshot_colors.py gives: dataset/ shouldn't depend on
# realtime/'s own sys.path setup. What the live scanner's wire protocol sends for a face with NO
# real captured color at all (realtime/Windows_server/mesh_intercept.js: colorsBytes=0 whenever
# the vendor DLL's own color pointer is null for that publish) - confirmed empirically
# (analyze_snapshot_colors.py, 2026-08-21) that 65-88% of faces in a real live scan carry this
# value, and that a color-trained checkpoint reads it as far darker than either real class range
# below, causing it to default to gum almost unconditionally on this sentinel value (99.9% gum
# rate in one live run). See dataset/patch_preprocessing_color_dropout.py.
NO_COLOR_SENTINEL = np.array([102, 102, 102], dtype=np.uint8)

# RGB ranges (0-255). Confirmed values (user-approved), not placeholders.
TOOTH_RGB_LOW = np.array([210, 200, 175], dtype=np.float32)   # white/cream
TOOTH_RGB_HIGH = np.array([245, 235, 215], dtype=np.float32)
GUM_RGB_LOW = np.array([190, 100, 115], dtype=np.float32)     # red/pink/salmon
GUM_RGB_HIGH = np.array([225, 150, 155], dtype=np.float32)


class SyntheticColorPaint:
    """Callable: paint(labels, rng) -> (F, 3) uint8 RGB array.

    label 0 = gum in BOTH the 17-class and 5-class schemes (the 5-class remap keeps 0 -> 0), so
    this works unchanged regardless of which scheme `labels` is already in.
    """

    def __call__(self, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        colors = np.zeros((len(labels), 3), dtype=np.uint8)
        tooth_mask = labels != 0
        gum_mask = ~tooth_mask
        if tooth_mask.any():
            colors[tooth_mask] = rng.uniform(TOOTH_RGB_LOW, TOOTH_RGB_HIGH).astype(np.uint8)
        if gum_mask.any():
            colors[gum_mask] = rng.uniform(GUM_RGB_LOW, GUM_RGB_HIGH).astype(np.uint8)
        return colors


# Measured directly from real scanner captures (analyze_snapshot_colors.py, 2 independent live
# hardware sessions 2026-08-19, ~650K genuinely-captured faces combined after excluding sentinel/
# no-real-color faces - see that script's own output) - NOT hand-picked like TOOTH_RGB_LOW/HIGH
# above (those were chosen before any real data existed). Confirmed empirically that
# SyntheticColorPaint's flat, zero-variance color was itself a real, measurable cause of the
# color-trained model's live hardware degradation (a controlled test: identical geometry/labels,
# only adding +-25/channel noise on top of otherwise-flat color dropped mIoU by 0.085) - these
# stats exist to retrain against a distribution that actually resembles what the scanner produces.
MEASURED_TOOTH_RGB_MEAN = np.array([198.8, 194.0, 198.4], dtype=np.float32)
MEASURED_TOOTH_RGB_STD = np.array([13.4, 16.2, 17.9], dtype=np.float32)
MEASURED_GUM_RGB_MEAN = np.array([175.6, 140.0, 136.5], dtype=np.float32)
MEASURED_GUM_RGB_STD = np.array([22.7, 25.5, 25.5], dtype=np.float32)


class RealisticColorPaint:
    """Callable: paint(labels, rng) -> (F, 3) uint8 RGB array.

    The deliberate reversal of SyntheticColorPaint's design, requested once real scanner data
    confirmed the flat/zero-variance color itself was the problem: every face gets its OWN
    independent draw from a per-class Gaussian (mosaic-style), not one shared value per class-
    group. SyntheticColorPaint's own docstring called per-face variation "meaningless mosaic
    noise no real scan ever has" - that was true when it was written (no real color data existed
    yet), but analyze_snapshot_colors.py's stats (measured std ~13-27/channel, not 0) show real
    scanner color DOES vary per-face, continuously, this much - so matching that variance during
    training is the realistic choice now, not the naive one. SyntheticColorPaint itself is left
    completely untouched (still there to fall back to) - this is a new, separate class, not a
    replacement.

    Centered on MEASURED_TOOTH/GUM_RGB_MEAN/STD by default (not the hand-picked TOOTH_RGB_LOW/HIGH
    range), but takes its own mean/std so it can be recalibrated later from a fresh
    analyze_snapshot_colors.py run without editing code - just pass new values in.

    label 0 = gum in BOTH the 17-class and 5-class schemes (the 5-class remap keeps 0 -> 0), same
    as SyntheticColorPaint.
    """

    def __init__(self, tooth_mean=MEASURED_TOOTH_RGB_MEAN, tooth_std=MEASURED_TOOTH_RGB_STD,
                 gum_mean=MEASURED_GUM_RGB_MEAN, gum_std=MEASURED_GUM_RGB_STD):
        self.tooth_mean = tooth_mean
        self.tooth_std = tooth_std
        self.gum_mean = gum_mean
        self.gum_std = gum_std

    def __call__(self, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        colors = np.zeros((len(labels), 3), dtype=np.float64)
        tooth_mask = labels != 0
        gum_mask = ~tooth_mask
        if tooth_mask.any():
            colors[tooth_mask] = rng.normal(self.tooth_mean, self.tooth_std, size=(tooth_mask.sum(), 3))
        if gum_mask.any():
            colors[gum_mask] = rng.normal(self.gum_mean, self.gum_std, size=(gum_mask.sum(), 3))
        return np.clip(colors, 0, 255).astype(np.uint8)
