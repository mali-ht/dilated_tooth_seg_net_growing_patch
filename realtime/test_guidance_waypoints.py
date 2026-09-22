#!/usr/bin/env python3
"""Headless regression tests for physical outbound-waypoint progression."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guidance_waypoints import ForwardSweepProgress  # noqa: E402


def test_detection_is_not_arrival():
    state = ForwardSweepProgress((8, 7, 6), waypoint_mm=3.0, min_progress_mm=4.0)
    centers = {8: np.array([0.0, 0.0, 0.0]),
               7: np.array([10.0, 0.0, 0.0]),
               6: np.array([20.0, 0.0, 0.0])}
    # All three labels are visible in the accumulated mesh, but the probe is at
    # tooth 8.  Only tooth 8 may complete.
    assert state.update({8, 7, 6}, centers, np.array([0.5, 0.0, 0.0])) == 7
    assert state.reached == {8}
    # Repeated inference while stationary cannot consume tooth 7.
    for _ in range(5):
        assert state.update({8, 7, 6}, centers, np.array([0.5, 0.0, 0.0])) == 7
    assert state.reached == {8}


def test_near_or_passed_after_real_progress_advances_once():
    state = ForwardSweepProgress((8, 7, 6), waypoint_mm=2.0, min_progress_mm=4.0)
    centers = {8: np.array([0.0, 0.0, 0.0]),
               7: np.array([10.0, 0.0, 0.0]),
               6: np.array([20.0, 0.0, 0.0])}
    state.update({8, 7, 6}, centers, np.array([0.0, 0.0, 0.0]))
    assert state.update({8, 7, 6}, centers, np.array([10.5, 0.0, 0.0])) == 6
    # Even though this sample has passed both remaining teeth, one call can
    # complete only one waypoint.
    assert state.update({8, 7, 6}, centers, np.array([25.0, 0.0, 0.0])) is None
    assert state.reached == {8, 7, 6}


def test_unconfirmed_waypoint_never_completes():
    state = ForwardSweepProgress((8,), waypoint_mm=8.0, min_progress_mm=0.0)
    centers = {8: np.array([0.0, 0.0, 0.0])}
    assert state.update(set(), centers, np.array([0.0, 0.0, 0.0])) == 8
    assert not state.reached


if __name__ == "__main__":
    test_detection_is_not_arrival()
    test_near_or_passed_after_real_progress_advances_once()
    test_unconfirmed_waypoint_never_completes()
    print("ALL PASS")
