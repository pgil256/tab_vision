"""Unit tests for ``tabvision.video.hand.fingertip_to_fret``.

The projection + posterior logic is pure-Python; these exercise it
directly with synthetic ``HandSample`` inputs and a hand-built
homography, no MediaPipe / opencv / video dataset required.
"""

# ruff: noqa: N806 — H is the math-convention name for the homography matrix

from __future__ import annotations

import numpy as np
import pytest

from tabvision.types import FrameFingering, GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import (
    FRETTING_FINGERS,
    FingerSample,
    HandSample,
    PosteriorConfig,
    compute_fingering,
    marginal_string_fret,
)

# ----- helpers -----


def _make_homography(
    nut_x: float = 100.0,
    body_x: float = 500.0,
    top_y: float = 200.0,
    bottom_y: float = 320.0,
) -> Homography:
    """A horizontal-fretboard homography in pixel space.

    Canonical [0,1]² maps to:
        (0,0) -> (nut_x, top_y)        # nut, high-E side
        (1,0) -> (body_x, top_y)       # body, high-E side
        (1,1) -> (body_x, bottom_y)    # body, low-E side
        (0,1) -> (nut_x, bottom_y)     # nut, low-E side
    """
    import cv2

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array(
        [
            [nut_x, top_y],
            [body_x, top_y],
            [body_x, bottom_y],
            [nut_x, bottom_y],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)  # noqa: N806
    return Homography(H=H.astype(np.float64), confidence=0.9, method="keypoint")


def _hand_sample_with_fingers(positions: dict[str, tuple[float, float]]) -> HandSample:
    """Build a HandSample with the given pixel-coord tip positions.

    Defaults: wrist far off the fretboard, all fingers fully extended,
    tip_z ≈ wrist_z (so the press-z bonus fires).
    """
    fingers: dict[str, FingerSample] = {}
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        xy = positions.get(name, (1000.0, 1000.0))  # default: off-frame
        fingers[name] = FingerSample(
            name=name,
            tip_xy=xy,
            tip_z=0.0,
            curl_ratio=1.0,  # fully extended
        )
    return HandSample(
        wrist_xy=(50.0, 250.0),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.9,
        fingers=fingers,
    )


# ----- compute_fingering: shape & basic invariants -----


def test_compute_fingering_returns_correct_shape():
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()
    hand = _hand_sample_with_fingers({})
    out = compute_fingering(hand, H, cfg)
    assert isinstance(out, FrameFingering)
    assert out.finger_pos_logits.shape == (
        len(FRETTING_FINGERS),
        cfg.n_strings,
        cfg.max_fret + 1,
    )
    assert out.homography_confidence == pytest.approx(0.9)


def test_compute_fingering_zero_homography_confidence_returns_uniform_floor():
    cfg = GuitarConfig(max_fret=12)
    H = Homography(H=np.eye(3, dtype=np.float64), confidence=0.0, method="keypoint")
    hand = _hand_sample_with_fingers({})
    out = compute_fingering(hand, H, cfg)
    # All cells should be at the floor logit.
    assert (out.finger_pos_logits == PosteriorConfig().floor_logit).all()
    assert out.homography_confidence == 0.0


# ----- compute_fingering: peak location -----


def test_index_finger_at_fret_5_string_3_peaks_at_correct_cell():
    """A finger placed at the 5th-fret centre of the 3rd string from the top
    (tuning idx 5-1-2 = 2 since cfg.tuning_midi is low-to-high; canonical y
    for tuning idx 2 is (n-1-2+0.5)/n = (6-1-2+0.5)/6 = 3.5/6 ≈ 0.583)
    should produce its peak at exactly (string=2, fret=5)."""
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()
    # canonical (x, y) for fret-cell-5 centre × tuning-idx-2 row:
    #   x = (5 + 0.5) / 13 = 0.4231
    #   y = (6 - 1 - 2 + 0.5) / 6 = 0.5833
    canon_x = (5 + 0.5) / 13.0
    canon_y = (cfg.n_strings - 1 - 2 + 0.5) / cfg.n_strings
    # Project canonical → pixels via H to get the synthetic finger position.
    proj = H.H @ np.array([canon_x, canon_y, 1.0])
    px = float(proj[0] / proj[2])
    py = float(proj[1] / proj[2])

    hand = _hand_sample_with_fingers({"index": (px, py)})
    out = compute_fingering(hand, H, cfg)
    # finger 0 = index. Its argmax cell should be (string=2, fret=5).
    finger_logits = out.finger_pos_logits[0]
    s_arg, f_arg = np.unravel_index(finger_logits.argmax(), finger_logits.shape)
    assert (int(s_arg), int(f_arg)) == (2, 5)


def test_curled_finger_logits_uniformly_lower_than_extended():
    """Setting curl_ratio low (curled) should subtract the curl penalty
    everywhere — extended-finger logits beat curled ones at every cell."""
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()

    # Synthetic finger position at fret 5 / string 2 again.
    proj = H.H @ np.array([(5 + 0.5) / 13.0, (cfg.n_strings - 1 - 2 + 0.5) / cfg.n_strings, 1.0])
    px, py = float(proj[0] / proj[2]), float(proj[1] / proj[2])

    fingers_extended = {"index": (px, py)}
    hand_ext = _hand_sample_with_fingers(fingers_extended)
    # Same finger but curled.
    base = _hand_sample_with_fingers(fingers_extended)
    curled_index = FingerSample(
        name="index", tip_xy=base.fingers["index"].tip_xy,
        tip_z=base.fingers["index"].tip_z, curl_ratio=0.6,
    )
    new_fingers = dict(base.fingers)
    new_fingers["index"] = curled_index
    hand_curled = HandSample(
        wrist_xy=base.wrist_xy,
        wrist_z=base.wrist_z,
        is_left_hand=base.is_left_hand,
        confidence=base.confidence,
        fingers=new_fingers,
    )

    ext_logits = compute_fingering(hand_ext, H, cfg).finger_pos_logits[0]
    cur_logits = compute_fingering(hand_curled, H, cfg).finger_pos_logits[0]
    diff = ext_logits - cur_logits
    # Extended logits >= curled logits everywhere; mean diff is positive.
    assert (diff >= -1e-9).all()
    assert diff.mean() > 0


def test_pressing_z_window_adds_bonus_to_logits():
    """When tip_z is within the press window, a positive bonus is added."""
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()
    proj = H.H @ np.array([(5 + 0.5) / 13.0, (cfg.n_strings - 1 - 2 + 0.5) / cfg.n_strings, 1.0])
    px, py = float(proj[0] / proj[2]), float(proj[1] / proj[2])

    pcfg = PosteriorConfig()
    base = _hand_sample_with_fingers({"index": (px, py)})

    # Pressing: tip_z near wrist_z (within window).
    pressing = base.fingers["index"]
    pressing_hand = HandSample(
        wrist_xy=base.wrist_xy, wrist_z=0.0, is_left_hand=base.is_left_hand,
        confidence=base.confidence,
        fingers={**base.fingers,
                 "index": FingerSample("index", pressing.tip_xy, 0.0, 1.0)},
    )

    # Not pressing: tip_z way past the press window.
    not_pressing_hand = HandSample(
        wrist_xy=base.wrist_xy, wrist_z=0.0, is_left_hand=base.is_left_hand,
        confidence=base.confidence,
        fingers={**base.fingers,
                 "index": FingerSample("index", pressing.tip_xy, 1.0, 1.0)},
    )

    press_logits = compute_fingering(pressing_hand, H, cfg, pcfg).finger_pos_logits[0]
    no_press_logits = compute_fingering(not_pressing_hand, H, cfg, pcfg).finger_pos_logits[0]
    diff = press_logits - no_press_logits
    # Diff should equal the press bonus (modulo floor clamping at far-away cells).
    assert diff.max() == pytest.approx(pcfg.pressing_log_bonus, abs=1e-9)


# ----- marginal_string_fret -----


def test_marginal_is_a_proper_distribution():
    """Output sums to 1 and is non-negative."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 6, 13))
    m = marginal_string_fret(logits)
    assert m.shape == (6, 13)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_marginal_combines_finger_evidence():
    """Two fingers concentrated at distinct cells → marginal has mass at both."""
    n_fingers, n_strings, n_frets = 4, 6, 13
    logits = np.full((n_fingers, n_strings, n_frets), -10.0)
    logits[0, 0, 0] = 0.0      # finger 0 sharp at (string 0, fret 0)
    logits[1, 5, 12] = 0.0     # finger 1 sharp at (string 5, fret 12)
    m = marginal_string_fret(logits)
    # The two designated cells should be the largest two entries.
    flat = m.reshape(-1)
    top2 = np.argsort(flat)[-2:]
    assert (n_strings * n_frets - 1) in top2  # last cell
    assert 0 in top2                            # first cell


def test_marginal_string_fret_via_frame_fingering_method():
    """The §8 dataclass method delegates to the same function."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 6, 13))
    ff = FrameFingering(t=0.5, finger_pos_logits=logits, homography_confidence=0.7)
    m = ff.marginal_string_fret()
    assert m.shape == (6, 13)
    assert m.sum() == pytest.approx(1.0)


def test_marginal_string_fret_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected"):
        marginal_string_fret(np.zeros((6, 13)))  # missing finger dim


# ----- string axis convention -----


def test_string_index_zero_is_low_e_at_bottom_of_canonical_y():
    """A finger projected to canonical y ≈ 1 (low-E side) should have its
    peak on string idx 0 (low-E in cfg.tuning_midi)."""
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()
    # canonical (x, y) = (0.4, 0.95): near low-E side (y ≈ 1).
    proj = H.H @ np.array([0.4, 0.95, 1.0])
    px, py = float(proj[0] / proj[2]), float(proj[1] / proj[2])
    hand = _hand_sample_with_fingers({"index": (px, py)})
    out = compute_fingering(hand, H, cfg)
    s_arg, _ = np.unravel_index(out.finger_pos_logits[0].argmax(),
                                out.finger_pos_logits[0].shape)
    assert int(s_arg) == 0


def test_string_index_max_is_high_e_at_top_of_canonical_y():
    cfg = GuitarConfig(max_fret=12)
    H = _make_homography()
    proj = H.H @ np.array([0.4, 0.05, 1.0])  # near high-E side (y ≈ 0)
    px, py = float(proj[0] / proj[2]), float(proj[1] / proj[2])
    hand = _hand_sample_with_fingers({"index": (px, py)})
    out = compute_fingering(hand, H, cfg)
    s_arg, _ = np.unravel_index(out.finger_pos_logits[0].argmax(),
                                out.finger_pos_logits[0].shape)
    assert int(s_arg) == cfg.n_strings - 1
