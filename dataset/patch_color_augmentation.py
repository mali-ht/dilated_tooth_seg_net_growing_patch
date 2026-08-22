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


class RealisticColorPaintHSV:
    """Callable: paint(labels, rng) -> (F, 3) uint8 RGB array.

    Fixes a real problem found in RealisticColorPaint (confirmed 2026-08-22 via
    testing/visualize_training_input.py, the first time this project actually looked at painted
    training color rather than just its summary statistics): drawing R, G, B as three
    INDEPENDENT per-channel Gaussians can - and regularly does - land far enough from the mean on
    one channel while another goes the opposite way to produce a strongly saturated, wrong-family
    color for a whole face (a sampled tooth face came out RGB(212,149,231), a visible purple/
    magenta - no real tooth enamel looks like that). Real tooth/gum color doesn't vary that way:
    channels move together (a slightly more yellow tooth shifts R and G together, not one up and
    one down independently), which is exactly what sampling in HSV and converting to RGB
    enforces - the 3 output channels are correlated by construction, not independent draws.

    Hue is drawn from a narrow, class-appropriate range instead of matching MEASURED_TOOTH/
    GUM_RGB_MEAN/STD's raw RGB directly - user-directed 2026-08-22: teeth in a warm cream/white/
    yellowish family, gum in red/pink. (The raw measured tooth mean's OWN computed hue is ~305
    degrees/magenta-leaning at saturation ~0.02 - meaningless at near-zero saturation, an artifact
    of which channel is fractionally highest in an essentially-gray color, not a real "true hue"
    worth reproducing; the gum mean's hue, ~5 degrees/red, IS meaningful at its higher ~0.22
    saturation and matches the red/pink family directly.) Saturation and value are calibrated so
    the resulting RGB mean/std land close to MEASURED_TOOTH/GUM_RGB_MEAN/STD's own magnitude
    (verified directly, not just asserted) - same overall amount of per-face variation as before,
    just correlated instead of independent.

    label 0 = gum in BOTH the 17-class and 5-class schemes (the 5-class remap keeps 0 -> 0), same
    as SyntheticColorPaint/RealisticColorPaint.
    """

    # (hue_mean_deg, hue_std_deg, hue_range_deg, sat_mean, sat_std, sat_range, val_mean, val_std,
    # val_range) - hue_range is allowed to go negative (e.g. gum's -10) and wraps via mod 360, for
    # a family straddling 0/360 (red) without needing two disjoint ranges.
    TOOTH_HSV = dict(hue_mean=48.0, hue_std=8.0, hue_range=(25.0, 65.0),
                      sat_mean=0.12, sat_std=0.05, sat_range=(0.02, 0.28),
                      val_mean=0.80, val_std=0.06, val_range=(0.55, 1.0))
    GUM_HSV = dict(hue_mean=5.0, hue_std=6.0, hue_range=(-10.0, 20.0),
                   sat_mean=0.30, sat_std=0.10, sat_range=(0.10, 0.60),
                   val_mean=0.70, val_std=0.08, val_range=(0.45, 0.92))

    def __init__(self, tooth_hsv=TOOTH_HSV, gum_hsv=GUM_HSV):
        self.tooth_hsv = tooth_hsv
        self.gum_hsv = gum_hsv

    @staticmethod
    def _sample_rgb(n, hsv_params, rng):
        from matplotlib.colors import hsv_to_rgb
        p = hsv_params
        h_deg = np.clip(rng.normal(p['hue_mean'], p['hue_std'], n), *p['hue_range'])
        h = np.mod(h_deg, 360.0) / 360.0
        s = np.clip(rng.normal(p['sat_mean'], p['sat_std'], n), *p['sat_range'])
        v = np.clip(rng.normal(p['val_mean'], p['val_std'], n), *p['val_range'])
        return hsv_to_rgb(np.stack([h, s, v], axis=1)) * 255.0

    def __call__(self, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        colors = np.zeros((len(labels), 3), dtype=np.float64)
        tooth_mask = labels != 0
        gum_mask = ~tooth_mask
        if tooth_mask.any():
            colors[tooth_mask] = self._sample_rgb(int(tooth_mask.sum()), self.tooth_hsv, rng)
        if gum_mask.any():
            colors[gum_mask] = self._sample_rgb(int(gum_mask.sum()), self.gum_hsv, rng)
        return np.clip(colors, 0, 255).astype(np.uint8)
