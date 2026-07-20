"""Runtime assisted-review candidate ranking (2026-07-20 program)."""

from __future__ import annotations

from dataclasses import replace

from tabvision.assist import compute_note_candidates
from tabvision.fusion import fuse
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig


def _events() -> list[AudioEvent]:
    # Ambiguous mid-register pitches: each is playable on several strings.
    pitches = [64, 67, 69, 64]
    return [
        AudioEvent(
            onset_s=0.5 * i,
            offset_s=0.5 * i + 0.4,
            pitch_midi=pitch,
            velocity=0.8,
            confidence=0.9,
        )
        for i, pitch in enumerate(pitches)
    ]


def test_candidates_align_to_production_decode() -> None:
    cfg = GuitarConfig()
    events = _events()
    tab_events = fuse(events, [], cfg, SessionConfig(), lambda_vision=0.0)

    ranked = compute_note_candidates(
        tab_events,
        events,
        cfg=cfg,
        sequence_prior="none",
    )

    assert len(ranked) == len(tab_events)
    for event, candidates in zip(tab_events, ranked, strict=True):
        assert candidates is not None
        positions = [(cand.string_idx, cand.fret) for cand in candidates]
        # The emitted position is always present and pitch is preserved.
        assert (event.string_idx, event.fret) in positions
        tuning = cfg.tuning_midi
        for cand in candidates:
            assert tuning[cand.string_idx] + cand.fret == event.pitch_midi
        # Ambiguous pitches expose at least one alternative to cycle to.
        assert len(positions) >= 2
        # Ranking is by min-marginal cost delta, best first.
        deltas = [cand.cost_delta_from_best for cand in candidates]
        assert deltas == sorted(deltas)
        assert deltas[0] <= 1e-9


def test_unalignable_note_degrades_to_none() -> None:
    cfg = GuitarConfig()
    events = _events()
    tab_events = fuse(events, [], cfg, SessionConfig(), lambda_vision=0.0)
    # A note the analysis decode has never seen (fabricated onset).
    tampered = [replace(tab_events[0], onset_s=99.0), *tab_events[1:]]

    ranked = compute_note_candidates(tampered, events, cfg=cfg, sequence_prior="none")

    assert ranked[0] is None
    assert all(item is not None for item in ranked[1:])


def test_empty_input_is_empty() -> None:
    assert compute_note_candidates([], [], cfg=GuitarConfig(), sequence_prior="none") == []
