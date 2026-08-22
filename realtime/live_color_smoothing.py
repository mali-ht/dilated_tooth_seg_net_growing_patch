import numpy as np
from scipy.spatial import cKDTree

# New, additive-only file - realtime/live_preprocessing.py is untouched except for one
# backward-compatible extension point (see below). live_preprocessing.py's
# classify_and_flatten_colors (--color_mode flat's redness>threshold heuristic) is stateless -
# every process_once() cycle classifies from scratch with nothing carried over from the last one.
# Live testing 2026-08-21 showed this is a real source of instability: gum% swung 25%->56%->25%->
# 62% across a handful of consecutive cycles on real hardware, tracked almost exactly by the raw
# per-cycle heuristic (96-100% agreement with the model's own predictions) - i.e. a momentary
# lighting/exposure blip in one cycle's real captured color flips the call for that whole cycle,
# with nothing smoothing it out.
#
# Separately confirmed (same session, pooling 2.35M real captured faces across 6 live runs): the
# redness>20 threshold ITSELF is already close to data-optimal (Otsu search over the pooled
# distribution found 19.92) - the redness distribution is a single skewed lump with a long tail,
# not two separated bumps, so no threshold fixes the underlying overlap. Temporal smoothing is the
# remaining real lever: even an imperfect per-cycle classification becomes far more usable once
# it's not allowed to flip on a single noisy cycle.
#
# TemporalGumClassifier is a drop-in replacement for classify_and_flatten_colors (same call
# signature and return shape - an (F,3) uint8 painted-color array), passed to
# mesh_to_model_inputs_with_color's new classify_fn= parameter. It duplicates
# classify_and_flatten_colors's own redness-heuristic and sentinel-propagation logic rather than
# calling it directly - same reasoning realtime/analyze_snapshot_colors.py already gives for
# inlining its own copy of NO_COLOR_SENTINEL: keeps this file's only dependency on
# live_preprocessing.py to the one constant, not a deeper call chain. If
# classify_and_flatten_colors's redness_threshold or sentinel-propagation logic changes, this
# file's copy needs the same update - flagged here so that's not missed.
#
# NOT YET VERIFIED ON REAL HARDWARE - written 2026-08-21 on the Linux side, where there's no live
# scanner to test against. Opt-in only (mesh_viewer_segmented.py's --gum_smoothing_alpha, default
# None = old stateless behavior, byte-identical to before this file existed) specifically so it
# can be tried without risking the already-working --color_mode flat path.


class TemporalGumClassifier:
    """Stateful, drop-in replacement for live_preprocessing.classify_and_flatten_colors.

    Each call: computes this cycle's RAW is-gum score (redness = R - mean(G,B), same formula and
    default threshold as the original) with the same nearest-real-color sentinel propagation, then
    blends it with the nearest spatially-matching face from the PREVIOUS cycle's cache via an EMA
    (alpha = weight on the new, current-cycle evidence; 1-alpha = weight on accumulated history) -
    higher alpha tracks real change faster but smooths less, lower alpha is stabler but slower to
    respond to an actual, real newly-scanned gum/tooth boundary.

    Matching across cycles is by NEAREST 3D CENTROID (max_match_dist_mm cap), not face index - the
    accumulated mesh's own face indices aren't stable across cycles (voxel dedup/remeshing), but a
    given real-world spot's position is. A face with no cached match within max_match_dist_mm
    (freshly-scanned geometry, or the very first cycle) falls back to its own raw score for that
    cycle alone - nothing to blend with yet, same effective behavior as the original stateless
    function for brand-new surface.
    """

    def __init__(self, alpha: float = 0.3, redness_threshold: float = 20.0, max_match_dist_mm: float = 1.0):
        self.alpha = alpha
        self.redness_threshold = redness_threshold
        self.max_match_dist_mm = max_match_dist_mm
        self._prev_centroids = None  # (N, 3) float, previous cycle's face_centroids
        self._prev_score = None      # (N,) float in [0, 1], previous cycle's blended gum score

    def _raw_gum_score(self, face_colors: np.ndarray, face_centroids: np.ndarray) -> np.ndarray:
        """Same classification as classify_and_flatten_colors, as a continuous [0, 1] score
        (1.0 = clearly gum, 0.0 = clearly tooth) instead of a boolean, so EMA blending has
        something to average - redness itself, min-max squashed by the threshold rather than
        hard-cut, preserves "how confidently gum" instead of collapsing it before blending."""
        from live_preprocessing import NO_COLOR_SENTINEL  # local import: avoid a module-level
        # dependency cycle risk (this file gets imported by mesh_viewer_segmented.py, which also
        # imports live_preprocessing directly) - NO_COLOR_SENTINEL itself never changes at runtime
        # so importing it lazily here costs nothing.
        face_colors = face_colors.astype(np.float64)
        is_sentinel = np.all(face_colors == NO_COLOR_SENTINEL, axis=1)
        redness = face_colors[:, 0] - face_colors[:, 1:3].mean(axis=1)
        # Same soft mapping around the threshold on both sides - 20 redness above threshold is as
        # confidently gum as 20 below is confidently tooth. Clipped to [0,1] for blending.
        score = np.clip(0.5 + (redness - self.redness_threshold) / 40.0, 0.0, 1.0)

        if is_sentinel.any():
            if is_sentinel.all():
                score[:] = 0.0  # last-resort default - same as classify_and_flatten_colors
            else:
                real_idx = np.where(~is_sentinel)[0]
                nearest = cKDTree(face_centroids[real_idx]).query(face_centroids[is_sentinel], k=1)[1]
                score[is_sentinel] = score[real_idx[nearest]]
        return score

    def __call__(self, face_colors: np.ndarray, face_centroids: np.ndarray,
                 tooth_color: np.ndarray, gum_color: np.ndarray) -> np.ndarray:
        raw_score = self._raw_gum_score(face_colors, face_centroids)

        if self._prev_centroids is not None and len(self._prev_centroids) > 0:
            dists, nearest = cKDTree(self._prev_centroids).query(face_centroids, k=1)
            has_match = dists <= self.max_match_dist_mm
            blended = raw_score.copy()
            blended[has_match] = (self.alpha * raw_score[has_match]
                                   + (1 - self.alpha) * self._prev_score[nearest[has_match]])
        else:
            blended = raw_score

        self._prev_centroids = face_centroids.copy()
        self._prev_score = blended.copy()

        is_gum = blended > 0.5
        out = np.empty_like(face_colors, dtype=np.uint8)
        out[~is_gum] = tooth_color
        out[is_gum] = gum_color
        return out
