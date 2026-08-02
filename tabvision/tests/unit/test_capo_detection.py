"""Tests for preflight capo detection.

The physical upper bound is the only part of this module with a guarantee
behind it — a capo at C makes everything below open+C unplayable — so it is
tested directly. The estimators themselves are heuristics whose accuracy is
an empirical question answered in q7_capo_detect_2026-07-23.md, not something
a unit test can assert; what is tested here is that they stay inside the
bound and degrade safely when there is nothing to measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.fusion.inharmonicity import StringStiffnessModel
from tabvision.preflight.capo import (
    detect_capo_from_inharmonicity,
    detect_capo_from_pitches,
    detect_capo_from_video,
)
from tabvision.types import AudioEvent, GuitarConfig

CFG = GuitarConfig()


def _event(pitch: int, onset: float = 0.0) -> AudioEvent:
    return AudioEvent(
        onset_s=onset, offset_s=onset + 0.4, pitch_midi=pitch, velocity=0.8, confidence=0.9
    )


def test_low_note_forces_a_low_upper_bound() -> None:
    # An open low E (40) is only playable with no capo.
    events = [_event(40, i * 0.5) for i in range(20)]
    estimate = detect_capo_from_pitches(events, CFG)
    assert estimate.upper_bound == 0
    assert estimate.capo == 0


def test_high_register_permits_a_high_bound() -> None:
    # Nothing below 52 means a capo up to 12 is physically possible; the bound
    # is capped at the max_capo the caller allows.
    events = [_event(60 + (i % 5), i * 0.5) for i in range(20)]
    estimate = detect_capo_from_pitches(events, CFG, max_capo=7)
    assert estimate.upper_bound == 7


def test_estimate_never_exceeds_the_physical_bound() -> None:
    events = [_event(42, i * 0.5) for i in range(10)] + [_event(55, 9.0)]
    estimate = detect_capo_from_pitches(events, CFG)
    assert estimate.capo <= estimate.upper_bound


def test_empty_input_is_safe() -> None:
    estimate = detect_capo_from_pitches([], CFG)
    assert estimate.capo == 0
    assert estimate.confidence == 0.0


def test_inharmonicity_declines_without_enough_measurable_notes() -> None:
    # Silence yields no usable stiffness fits, so the detector must abstain
    # rather than invent a capo from noise.
    events = [_event(60, i * 0.6) for i in range(4)]
    model = StringStiffnessModel(log_b0={s: -9.0 - 0.3 * s for s in range(6)})
    estimate = detect_capo_from_inharmonicity(events, np.zeros(44100 * 4), 44100, model, CFG)
    assert estimate.capo == 0
    assert estimate.confidence == 0.0
    assert "insufficient" in estimate.method


# --- video estimator (FretCam) -------------------------------------------


def _low_events(lowest_midi: int, count: int = 40) -> list[AudioEvent]:
    """Events whose lowest pitch fixes the physical capo upper bound."""
    return [
        AudioEvent(
            onset_s=0.1 * index,
            offset_s=0.1 * index + 0.05,
            pitch_midi=lowest_midi + (index % 12),
            velocity=0.8,
            confidence=0.9,
        )
        for index in range(count)
    ]


def test_video_estimate_is_reported_when_the_bound_allows_it():
    cfg = GuitarConfig()
    # Lowest note well above open low E, so a capo at 3 is physically possible.
    events = _low_events(min(cfg.tuning_midi) + 5)
    estimate = detect_capo_from_video(3, 0.8, events, cfg)
    assert estimate.capo == 3
    assert estimate.method == "video"
    assert estimate.confidence == pytest.approx(0.8)


def test_video_estimate_above_the_physical_bound_is_refuted():
    """The bound cannot locate a capo but it can refute one; it must win."""
    cfg = GuitarConfig()
    # Open low E is played, so no capo is possible at all.
    events = _low_events(min(cfg.tuning_midi))
    estimate = detect_capo_from_video(5, 0.95, events, cfg)
    assert estimate.capo == 0
    assert estimate.method == "video-refuted-by-bound"
    assert estimate.confidence == 0.0


def test_video_abstention_yields_no_capo():
    cfg = GuitarConfig()
    estimate = detect_capo_from_video(None, 0.0, _low_events(60), cfg)
    assert estimate.capo == 0
    assert estimate.confidence == 0.0
    assert estimate.method == "video"


@pytest.mark.parametrize("bad", [-1, 99])
def test_video_estimate_out_of_range_is_rejected(bad):
    cfg = GuitarConfig()
    estimate = detect_capo_from_video(bad, 0.9, _low_events(60), cfg)
    assert estimate.capo == 0
    assert estimate.method == "video-out-of-range"


def test_video_confidence_is_clamped():
    cfg = GuitarConfig()
    events = _low_events(min(cfg.tuning_midi) + 5)
    assert detect_capo_from_video(2, 5.0, events, cfg).confidence == 1.0
    assert detect_capo_from_video(2, -3.0, events, cfg).confidence == 0.0
