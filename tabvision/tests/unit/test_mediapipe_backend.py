"""Unit tests for ``tabvision.video.hand.mediapipe_backend``.

The MediaPipe inference path is gated to integration tests (real model
weights + opencv); here we exercise the pure-Python helpers
(`_select_fretting_hand`, `_build_hand_sample`, `_empty_fingering`) by
injecting fake landmark / handedness objects and stub out
``MediaPipeHandBackend._extract_fretting_hand`` for the end-to-end path.
"""

# ruff: noqa: N806 — H is the math-convention name for the homography matrix

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tabvision.errors import BackendError
from tabvision.types import GuitarConfig, Homography
from tabvision.video.hand.fingertip_to_fret import HandSample
from tabvision.video.hand.mediapipe_backend import (
    DEFAULT_MODEL_ENV,
    FRETTING_FINGERS,
    HandBackendConfig,
    MediaPipeHandBackend,
    _build_hand_sample,
    _default_model_path,
    _empty_fingering,
    _select_fretting_hand,
)

# ----- model path / construction -----


def test_default_model_path_uses_env_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, str(tmp_path / "custom.task"))
    assert _default_model_path() == tmp_path / "custom.task"


def test_default_model_path_falls_back_to_mediapipe_dir(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)
    p = _default_model_path()
    assert p.name == "hand_landmarker.task"
    # ~/.mediapipe/models/... is the conventional location.
    assert ".mediapipe" in p.as_posix()


def test_constructor_does_not_load_model(tmp_path):
    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")
    assert backend._landmarker is None


def test_missing_model_raises_helpful_backend_error(tmp_path):
    backend = MediaPipeHandBackend(model_path=tmp_path / "missing.task")
    with pytest.raises(BackendError, match="hand_landmarker"):
        backend._load()


# ----- _select_fretting_hand -----


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _FakeCategory:
    category_name: str
    score: float = 0.9


def _hand_with_finger_xs(xs: list[float]) -> list[_FakeLandmark]:
    """Build a 21-landmark list whose fingertip x-coords are ``xs`` (5 items)."""
    assert len(xs) == 5
    lms = [_FakeLandmark(0.0, 0.0) for _ in range(21)]
    # FINGERTIP indices: 4 (thumb), 8, 12, 16, 20.
    for tip_idx, x in zip([4, 8, 12, 16, 20], xs, strict=True):
        lms[tip_idx] = _FakeLandmark(x, 0.5)
    return lms


def test_select_fretting_hand_single_hand_returns_zero():
    lms = [_hand_with_finger_xs([0.1, 0.2, 0.3, 0.4, 0.5])]
    handedness = [[_FakeCategory("Left")]]
    assert _select_fretting_hand(lms, handedness) == 0


def test_select_fretting_hand_picks_mediapipe_right_label():
    lms = [
        _hand_with_finger_xs([0.1, 0.2, 0.3, 0.4, 0.5]),
        _hand_with_finger_xs([0.6, 0.7, 0.8, 0.9, 1.0]),
    ]
    # MediaPipe handedness is mirrored: "Right" = player's left = fretting.
    handedness = [[_FakeCategory("Left")], [_FakeCategory("Right")]]
    assert _select_fretting_hand(lms, handedness) == 1


def test_select_fretting_hand_falls_back_to_widest_spread():
    """When neither hand is labelled "Right", widest fingertip spread wins."""
    narrow = _hand_with_finger_xs([0.40, 0.42, 0.44, 0.46, 0.48])  # spread = 0.08
    wide = _hand_with_finger_xs([0.10, 0.30, 0.50, 0.70, 0.90])    # spread = 0.80
    lms = [narrow, wide]
    handedness = [[_FakeCategory("Left")], [_FakeCategory("Left")]]
    assert _select_fretting_hand(lms, handedness) == 1


# ----- _build_hand_sample -----


def _full_landmark_list(
    *,
    wrist_xy: tuple[float, float] = (0.10, 0.50),
    tip_offsets: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50),
    pip_offsets: tuple[float, ...] = (0.05, 0.12, 0.18, 0.24, 0.30),
    mcp_offsets: tuple[float, ...] = (0.02, 0.05, 0.08, 0.10, 0.12),
) -> list[_FakeLandmark]:
    """Build a believable 21-landmark hand pose, image-normalised coords."""
    lms = [_FakeLandmark(0.0, 0.0) for _ in range(21)]
    lms[0] = _FakeLandmark(*wrist_xy)
    # Layout: thumb (1..4), index (5..8), middle (9..12), ring (13..16), pinky (17..20).
    finger_groups = [
        (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20),
    ]
    for fi, (cmc, mcp, pip, tip) in enumerate(finger_groups):
        lms[cmc] = _FakeLandmark(wrist_xy[0] + 0.01, wrist_xy[1])
        lms[mcp] = _FakeLandmark(wrist_xy[0] + mcp_offsets[fi], wrist_xy[1])
        lms[pip] = _FakeLandmark(wrist_xy[0] + pip_offsets[fi], wrist_xy[1])
        lms[tip] = _FakeLandmark(wrist_xy[0] + tip_offsets[fi], wrist_xy[1])
    return lms


def test_build_hand_sample_extended_fingers_have_high_curl_ratio():
    """Straight-line finger pose -> curl_ratio close to 1.0."""
    lms = _full_landmark_list()
    handedness = [_FakeCategory("Right", score=0.8)]
    sample = _build_hand_sample(lms, handedness, frame_width=640, frame_height=360)
    for name in FRETTING_FINGERS:
        assert sample.fingers[name].curl_ratio > 0.95


def test_build_hand_sample_curled_fingers_have_low_curl_ratio():
    """A pose where tip is closer to MCP than the joints -> curl_ratio < 1.0."""
    # tip at offset 0.05, pip at 0.12, mcp at 0.10 — tip "tucked back".
    lms = _full_landmark_list(
        tip_offsets=(0.05,) * 5,
        pip_offsets=(0.12,) * 5,
        mcp_offsets=(0.10,) * 5,
    )
    handedness = [_FakeCategory("Right")]
    sample = _build_hand_sample(lms, handedness, frame_width=640, frame_height=360)
    for name in FRETTING_FINGERS:
        assert sample.fingers[name].curl_ratio < 0.95


def test_build_hand_sample_scales_to_pixel_coords():
    lms = _full_landmark_list(wrist_xy=(0.25, 0.5))
    handedness = [_FakeCategory("Right")]
    sample = _build_hand_sample(lms, handedness, frame_width=640, frame_height=360)
    assert sample.wrist_xy == (0.25 * 640, 0.5 * 360)


def test_build_hand_sample_records_handedness_correctly():
    """is_left_hand corresponds to MediaPipe's "Right" label (mirrored)."""
    lms = _full_landmark_list()
    h_right = [_FakeCategory("Right", score=0.9)]
    h_left = [_FakeCategory("Left", score=0.85)]
    sample_right = _build_hand_sample(lms, h_right, frame_width=320, frame_height=240)
    sample_left = _build_hand_sample(lms, h_left, frame_width=320, frame_height=240)
    assert sample_right.is_left_hand is True   # MediaPipe "Right" -> player's left
    assert sample_left.is_left_hand is False
    assert sample_right.confidence == pytest.approx(0.9)


# ----- _empty_fingering -----


def test_empty_fingering_has_zero_confidence_and_correct_shape():
    cfg = GuitarConfig(max_fret=12)
    ff = _empty_fingering(cfg)
    assert ff.homography_confidence == 0.0
    assert ff.finger_pos_logits.shape == (
        len(FRETTING_FINGERS), cfg.n_strings, cfg.max_fret + 1,
    )


# ----- detect() end-to-end with a stubbed extractor -----


def test_detect_returns_empty_when_no_hand_extracted(monkeypatch, tmp_path):
    """When the underlying extractor returns None, detect() emits the
    empty-fingering sentinel rather than raising."""
    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")
    monkeypatch.setattr(backend, "_extract_fretting_hand", lambda _frame: None)
    cfg = GuitarConfig(max_fret=12)
    H = Homography(H=np.eye(3, dtype=np.float64), confidence=0.0, method="keypoint")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = backend.detect(frame, H, cfg)
    assert out.homography_confidence == 0.0
    assert out.finger_pos_logits.shape == (
        len(FRETTING_FINGERS), cfg.n_strings, cfg.max_fret + 1,
    )


def test_detect_runs_compute_fingering_when_hand_extracted(monkeypatch, tmp_path):
    """When the extractor yields a HandSample, detect() routes through
    fingertip_to_fret and returns a non-degenerate FrameFingering."""
    import cv2  # noqa: F401 — used to construct the homography

    from tabvision.video.hand.fingertip_to_fret import FingerSample

    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")
    cfg = GuitarConfig(max_fret=12)

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array([[100, 200], [500, 200], [500, 320], [100, 320]], dtype=np.float32)
    H_arr = cv2.getPerspectiveTransform(src, dst)  # noqa: N806
    H = Homography(H=H_arr.astype(np.float64), confidence=0.85, method="keypoint")

    fingers = {
        name: FingerSample(name=name, tip_xy=(300.0, 260.0), tip_z=0.0, curl_ratio=1.0)
        for name in ("thumb", "index", "middle", "ring", "pinky")
    }
    fake_sample = HandSample(
        wrist_xy=(80.0, 250.0), wrist_z=0.0, is_left_hand=True, confidence=0.9,
        fingers=fingers,
    )
    monkeypatch.setattr(backend, "_extract_fretting_hand", lambda _f: fake_sample)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = backend.detect(frame, H, cfg)
    assert out.homography_confidence == pytest.approx(0.85)
    # Some logit cell should be > the floor (i.e. peak from the Gaussian kernel).
    assert out.finger_pos_logits.max() > -10.0


def test_detect_raises_for_non_bgr_frame(tmp_path):
    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")
    cfg = GuitarConfig()
    H = Homography(H=np.eye(3, dtype=np.float64), confidence=1.0, method="keypoint")
    bad = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(BackendError, match="BGR"):
        backend.detect(bad, H, cfg)


def test_close_is_safe_when_landmarker_never_loaded(tmp_path):
    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")
    backend.close()  # must not raise
    backend.close()  # idempotent


def test_context_manager_calls_close(tmp_path):
    backend = MediaPipeHandBackend(model_path=tmp_path / "x.task")

    closed = []

    def _close():
        closed.append(True)

    backend.close = _close  # type: ignore[assignment]
    with backend:
        pass
    assert closed == [True]


def test_handbackendconfig_defaults_are_reasonable():
    cfg = HandBackendConfig()
    assert 0 < cfg.min_detection_confidence <= 1
    assert 0 < cfg.min_tracking_confidence <= 1
    assert cfg.max_num_hands >= 1
