"""Unit tests for tabvision.video.fretboard.tracker."""

import numpy as np

from tabvision.types import Homography
from tabvision.video.fretboard.tracker import smooth_homography_track


def _h(scale=1.0, conf=0.8, method="geometric") -> Homography:
    """Construct a simple identity-ish homography scaled by `scale`."""
    H = np.eye(3) * scale
    H[2, 2] = 1.0  # keep last element 1
    return Homography(H=H, confidence=conf, method=method)


def test_smooth_track_returns_one_per_frame():
    out = smooth_homography_track([_h(1.0), _h(2.0), _h(3.0)])
    assert len(out) == 3


def test_smooth_track_blends_high_confidence_frames():
    out = smooth_homography_track([_h(1.0), _h(3.0)], alpha=0.5)
    # Second frame: blend of det (3.0 * I_proj) with last (1.0 * I_proj),
    # alpha=0.5 → (1+3)/2 on diagonal entries [0,0] and [1,1].
    assert out[1].H[0, 0] == 2.0


def test_low_confidence_extrapolates_last():
    out = smooth_homography_track(
        [_h(1.0, conf=0.9), _h(99.0, conf=0.05), _h(2.0, conf=0.9)],
        alpha=0.5,
        min_confidence_for_update=0.2,
    )
    # Second frame is below threshold → reuses last.
    assert out[1].method == "tracker_extrapolated"
    assert out[1].H[0, 0] == 1.0
    assert out[1].confidence < 0.9
    # Third frame gets the regular blend.
    assert out[2].method == "geometric"
