"""Corrected studio transcriptions as gold-session matches."""

from __future__ import annotations

import pytest

from tabvision.personal.corrections import correction_notes_to_matches
from tabvision.types import GuitarConfig

# Standard tuning: E2=40 A2=45 D3=50 G3=55 B3=59 E4=64.


def test_corrected_notes_become_full_confidence_matches() -> None:
    matches = correction_notes_to_matches(
        [
            {"timestamp": 1.5, "string": 6, "fret": 3},
            {"timestamp": 2.0, "string": 1, "fret": 0},
        ]
    )

    assert [(m.note.string_idx, m.note.fret, m.note.pitch_midi) for m in matches] == [
        (0, 3, 43),
        (5, 0, 64),
    ]
    assert [m.onset_s for m in matches] == [1.5, 2.0]
    assert all(m.confidence == 1.0 for m in matches)


def test_muted_notes_are_skipped_not_guessed() -> None:
    matches = correction_notes_to_matches(
        [
            {"timestamp": 1.0, "string": 6, "fret": "X"},
            {"timestamp": 2.0, "string": 5, "fret": 2},
        ]
    )

    assert len(matches) == 1
    assert matches[0].note.fret == 2


def test_structural_problems_raise() -> None:
    with pytest.raises(ValueError, match="string must be 1..6"):
        correction_notes_to_matches([{"timestamp": 1.0, "string": 7, "fret": 0}])
    with pytest.raises(ValueError, match="fret must be 0..24"):
        correction_notes_to_matches([{"timestamp": 1.0, "string": 6, "fret": 25}])
    with pytest.raises(ValueError, match="finite and non-negative"):
        correction_notes_to_matches([{"timestamp": -1.0, "string": 6, "fret": 0}])
    with pytest.raises(ValueError, match="needs a numeric"):
        correction_notes_to_matches([{"string": 6, "fret": 0}])
    with pytest.raises(ValueError, match="no playable notes"):
        correction_notes_to_matches([{"timestamp": 1.0, "string": 6, "fret": "X"}])


def test_capo_is_refused() -> None:
    with pytest.raises(ValueError, match="capo 0"):
        correction_notes_to_matches(
            [{"timestamp": 1.0, "string": 6, "fret": 3}],
            GuitarConfig(capo=2),
        )
