"""Unit tests for attaching hand-neck anchors to audio events."""

from __future__ import annotations

import pytest

from tabvision.fusion.neck_prior import anchor_position_prior, apply_neck_anchor_priors
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.hand.neck_anchor import HandNeckAnchor


def _ev(midi: int, t: float) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=0.8,
    )


def test_apply_neck_anchor_priors_matches_nearest_anchor():
    cfg = GuitarConfig(max_fret=12)
    events = [_ev(69, 1.0)]
    anchors = [
        (0.0, HandNeckAnchor(2.0, 1.0, 3.0, 0.8)),
        (1.05, HandNeckAnchor(10.0, 9.0, 11.0, 0.8)),
    ]

    enriched = apply_neck_anchor_priors(events, anchors, cfg)

    assert enriched[0] is not events[0]
    assert enriched[0].fret_prior is not None
    assert enriched[0].fret_prior.shape == (cfg.n_strings, cfg.max_fret + 1)
    assert int(enriched[0].fret_prior.sum(axis=0).argmax()) == 10


def test_apply_neck_anchor_priors_leaves_event_when_anchor_too_far():
    cfg = GuitarConfig()
    events = [_ev(69, 1.0)]
    anchors = [(2.0, HandNeckAnchor(10.0, 9.0, 11.0, 0.8))]

    enriched = apply_neck_anchor_priors(events, anchors, cfg, max_time_distance_s=0.15)

    assert enriched[0] is events[0]
    assert enriched[0].fret_prior is None


def test_anchor_position_prior_is_normalized_and_peaks_near_center():
    cfg = GuitarConfig(max_fret=12)
    prior = anchor_position_prior(HandNeckAnchor(7.5, 6.0, 9.0, 0.9), cfg)

    assert prior.shape == (cfg.n_strings, cfg.max_fret + 1)
    assert float(prior.sum()) == pytest.approx(1.0)
    assert int(prior.sum(axis=0).argmax()) in {7, 8}
