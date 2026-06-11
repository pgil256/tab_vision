"""Unit tests for chunk-3 video-evidence robustness helpers."""

from __future__ import annotations

import numpy as np

from tabvision.fusion.vision_evidence import (
    ORIENTATION_BY_NAME,
    candidate_support,
    choose_orientation,
    combine_fingerings,
    gate_fingering_to_audio,
)
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig


def _event(pitch: int = 69, t: float = 1.0) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=pitch,
        velocity=1.0,
        confidence=1.0,
    )


def _peak(t: float, string_idx: int, fret: int, cfg: GuitarConfig) -> FrameFingering:
    logits = np.full((4, cfg.n_strings, cfg.max_fret + 1), -12.0, dtype=np.float64)
    logits[0, string_idx, fret] = 12.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


def test_choose_orientation_uses_audio_pitch_candidates() -> None:
    cfg = GuitarConfig()
    event = _event(69)
    # A4's audio default is high-E fret 5. This raw posterior is inverted on
    # both canonical axes, so only flip-both places mass on a pitch-compatible
    # string/fret cell.
    raw = _peak(event.onset_s, string_idx=0, fret=19, cfg=cfg)

    orientation, scores = choose_orientation([[raw]], [event], cfg)

    assert orientation == ORIENTATION_BY_NAME["flip-both"]
    assert scores["flip-both"] > scores["none"]


def test_combine_fingerings_votes_nearby_frame_posteriors() -> None:
    cfg = GuitarConfig()
    event = _event(69)
    fingerings = [
        _peak(0.96, string_idx=5, fret=5, cfg=cfg),
        _peak(1.02, string_idx=5, fret=5, cfg=cfg),
    ]

    voted = combine_fingerings(fingerings, cfg, t=event.onset_s)

    assert voted.t == event.onset_s
    assert voted.homography_confidence == 0.9
    assert candidate_support(event, voted, cfg) > 0.5


def test_gate_drops_video_that_does_not_support_audio_pitch() -> None:
    cfg = GuitarConfig()
    event = _event(69)
    wrong_pitch = _peak(event.onset_s, string_idx=0, fret=0, cfg=cfg)

    gated = gate_fingering_to_audio(event, wrong_pitch, cfg)

    assert gated.homography_confidence == 0.0
    assert not gated.finger_pos_logits.any()


def test_gate_keeps_decisive_pitch_compatible_video() -> None:
    cfg = GuitarConfig()
    event = _event(69)
    non_default_a4 = _peak(event.onset_s, string_idx=3, fret=14, cfg=cfg)

    gated = gate_fingering_to_audio(event, non_default_a4, cfg)

    assert gated is non_default_a4
