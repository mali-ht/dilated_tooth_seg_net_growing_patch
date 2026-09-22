"""Small, model-free state machines used by live scan guidance.

Detection and motion are deliberately separate here.  A segmentation label can
be visible in the accumulated model input before the scanner has physically
visited it; treating ``confirmed`` as ``reached`` made the live controller skip
ahead and reverse at the #5 -> #4 transition.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


@dataclass
class ForwardSweepProgress:
    """Track physically reached waypoints for one outbound arch sweep.

    At most one waypoint advances per inference cycle.  A waypoint must first
    be semantically confirmed, then the scanner must be near it or have passed
    it along the sweep axis.  After the first waypoint, real scanner movement
    is also required so overlapping model context cannot consume several teeth
    while the probe is stationary.
    """

    sequence: Sequence[int]
    waypoint_mm: float = 8.0
    min_progress_mm: float = 4.0
    reached: set = field(default_factory=set)
    _axis: Optional[np.ndarray] = None
    _last_reached_position: Optional[np.ndarray] = None

    def _establish_axis(self, centroids: Dict[int, np.ndarray]) -> None:
        if self._axis is not None:
            return
        for a, b in zip(self.sequence, self.sequence[1:]):
            if a not in centroids or b not in centroids:
                continue
            direction = np.asarray(centroids[b], dtype=float) - np.asarray(
                centroids[a], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                self._axis = direction / norm
                return

    @property
    def axis(self):
        return None if self._axis is None else self._axis.copy()

    def next_target(self) -> Optional[int]:
        return next((label for label in self.sequence if label not in self.reached), None)

    def update(self, confirmed: Iterable[int], centroids: Dict[int, np.ndarray],
               current_position) -> Optional[int]:
        """Advance no more than one physically reached tooth; return next target."""
        target = self.next_target()
        if target is None or current_position is None:
            return target

        confirmed = set(confirmed)
        if target not in confirmed or target not in centroids:
            return target

        current = np.asarray(current_position, dtype=float).reshape(-1)
        tooth = np.asarray(centroids[target], dtype=float).reshape(-1)
        if current.shape != tooth.shape or not np.isfinite(current).all() \
                or not np.isfinite(tooth).all():
            return target

        self._establish_axis(centroids)
        near = float(np.linalg.norm(current - tooth)) <= self.waypoint_mm
        passed = (self._axis is not None
                  and float(np.dot(current - tooth, self._axis)) >= 0.0)
        progressed = (self._last_reached_position is None
                      or float(np.linalg.norm(current - self._last_reached_position))
                      >= self.min_progress_mm)
        if progressed and (near or passed):
            self.reached.add(target)
            self._last_reached_position = current.copy()
        return self.next_target()
