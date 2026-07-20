from __future__ import annotations

import numpy as np
import pytest

from tabvision.audio.checkpoint_ensemble import (
    align_checkpoints,
    emitted_pitch_probability,
    intersection_events,
    match_same_pitch,
    select_events,
    union_events,
)
from tabvision.types import AudioEvent


def _event(
    onset: float, pitch: int, *, offset: float | None = None, score: float = 0.8
) -> AudioEvent:
    logits = np.full(128, -8.0, dtype=np.float32)
    logits[pitch] = np.log(score / (1.0 - score))
    return AudioEvent(onset, onset + 0.3 if offset is None else offset, pitch, 0.7, 0.7, logits)


def test_same_pitch_matching_is_closest_deterministic_and_boundary_inclusive() -> None:
    gaps = [_event(0.00, 64), _event(0.08, 64), _event(1.00, 67)]
    fl = [_event(0.05, 64), _event(0.09, 64), _event(1.05, 67)]

    first = match_same_pitch(gaps, fl)
    second = match_same_pitch(gaps, fl)

    assert first == second
    assert [(match.gaps_index, match.fl_index) for match in first] == [(0, 0), (1, 1), (2, 2)]


def test_agreed_events_preserve_gaps_onset_duration_and_identity() -> None:
    agreed = _event(0.2, 64, offset=1.4)
    fl = _event(0.23, 64, offset=0.5)

    intersection = intersection_events([agreed], [fl])
    union = union_events([agreed], [fl])

    assert intersection == (agreed,)
    assert union == (agreed,)
    assert intersection[0] is agreed
    assert intersection[0].offset_s == 1.4


def test_chord_notes_and_overlapping_disagreements_are_one_to_one() -> None:
    gaps = [_event(2.0, 60), _event(2.0, 64), _event(2.0, 67)]
    fl = [_event(2.01, 60), _event(2.01, 63), _event(2.01, 67)]

    alignment = align_checkpoints(gaps, fl)

    assert [(m.gaps_index, m.fl_index) for m in alignment.agreements] == [(0, 0), (2, 2)]
    assert [(m.gaps_index, m.fl_index) for m in alignment.disagreements] == [(1, 1)]
    assert not alignment.gaps_only
    assert not alignment.fl_only


def test_union_suppresses_only_cross_checkpoint_same_pitch_duplicates() -> None:
    gaps = [_event(0.0, 60), _event(0.01, 64)]
    fl = [_event(0.02, 60), _event(0.01, 67), _event(0.5, 69)]

    merged = union_events(gaps, fl)

    assert [(event.onset_s, event.pitch_midi) for event in merged] == [
        (0.0, 60),
        (0.01, 64),
        (0.01, 67),
        (0.5, 69),
    ]
    assert merged[0] is gaps[0]


def test_selector_preserves_agreement_and_uses_calibrated_scores() -> None:
    agreed = _event(0.0, 60, score=0.1)
    gaps_wrong = _event(1.0, 64, score=0.6)
    fl_winner = _event(1.01, 65, score=0.9)
    gaps_low_isolated = _event(2.0, 67, score=0.4)
    fl_high_isolated = _event(3.0, 69, score=0.8)
    gaps = [agreed, gaps_wrong, gaps_low_isolated]
    fl = [_event(0.01, 60), fl_winner, fl_high_isolated]

    selected = select_events(
        gaps,
        fl,
        score=lambda _source, _index, event: emitted_pitch_probability(event),
    )

    assert selected[0] is agreed
    assert [(event.onset_s, event.pitch_midi) for event in selected] == [
        (0.0, 60),
        (1.01, 65),
        (3.0, 69),
    ]


def test_emitted_pitch_probability_requires_real_finite_logits() -> None:
    assert emitted_pitch_probability(_event(0.0, 64, score=0.8)) == pytest.approx(0.8)
    no_logits = AudioEvent(0.0, 0.3, 64, 0.7, 0.7)
    assert emitted_pitch_probability(no_logits) == 0.0
    bad = _event(0.0, 64)
    bad.pitch_logits[64] = np.nan
    assert emitted_pitch_probability(bad) == 0.0
