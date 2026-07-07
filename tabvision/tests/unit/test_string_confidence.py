"""B4 — string-assignment confidence from the Viterbi string-flip margin.

``fuse`` now writes ``TabEvent.confidence`` as a monotone transform of the
best-vs-next-best-string margin read off the trellis (not the old velocity
proxy). Higher margin (the chosen string is clearly cheaper than any other
string) → higher confidence; an unambiguous pitch → 1.0; a tie → 0.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabvision.fusion import fuse
from tabvision.fusion.playability import string_margin_to_confidence
from tabvision.types import AudioEvent, GuitarConfig


def _ev(midi: int, t: float, *, fret_prior: np.ndarray | None = None) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=0.8,
        fret_prior=fret_prior,
    )


# ---------- the mapping ----------


def test_margin_to_confidence_endpoints_and_monotonic() -> None:
    assert string_margin_to_confidence(math.inf) == 1.0
    assert string_margin_to_confidence(0.0) == 0.0
    assert string_margin_to_confidence(-1.0) == 0.0  # clamped
    lo = string_margin_to_confidence(0.5)
    mid = string_margin_to_confidence(1.0)
    hi = string_margin_to_confidence(3.0)
    assert 0.0 < lo < mid < hi < 1.0


# ---------- confidence is no longer the velocity proxy ----------


def test_unambiguous_pitch_is_full_confidence() -> None:
    """MIDI 40 (low E) is playable on exactly one string → confidence 1.0,
    not the 0.8 audio-event confidence."""
    (note,) = fuse([_ev(40, 0.0)], [], GuitarConfig())
    assert note.string_idx == 0 and note.fret == 0
    assert note.confidence == 1.0


def test_ambiguous_pitch_is_below_full_confidence() -> None:
    """E4 (MIDI 64) is playable on 6 strings; the open-string pick wins by a
    finite margin → confidence strictly between 0 and 1."""
    (note,) = fuse([_ev(64, 0.0)], [], GuitarConfig())
    assert note.string_idx == 5 and note.fret == 0  # open high E
    assert 0.0 < note.confidence < 1.0


def test_more_ambiguous_note_is_less_confident() -> None:
    """C4 (no open-string option, two low frets close in cost) is a tighter
    call than E4 (an open string cleanly wins) → lower confidence."""
    (e4,) = fuse([_ev(64, 0.0)], [], GuitarConfig())
    (c4,) = fuse([_ev(60, 0.0)], [], GuitarConfig())
    assert c4.confidence < e4.confidence


def test_strong_fret_prior_raises_confidence() -> None:
    """A prior sharply favouring one string for an ambiguous pitch widens the
    margin → higher string confidence than the flat-prior decode."""
    cfg = GuitarConfig()
    prior = np.full((cfg.n_strings, cfg.max_fret + 1), 1e-3)
    prior[3, 9] = 1.0  # G string, fret 9 → MIDI 64
    (peaked,) = fuse([_ev(64, 0.0, fret_prior=prior)], [], cfg)
    (flat,) = fuse([_ev(64, 0.0)], [], cfg)
    assert peaked.string_idx == 3 and peaked.fret == 9  # prior steered the pick
    assert peaked.confidence > flat.confidence


def test_confidence_temp_scales_but_preserves_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The temperature sets the absolute confidence, not the note ranking."""
    import tabvision.fusion.playability as play

    monkeypatch.setattr(play, "STRING_CONFIDENCE_TEMP", 0.5)
    (sharp_c4,) = fuse([_ev(60, 0.0)], [], GuitarConfig())
    (sharp_e4,) = fuse([_ev(64, 0.0)], [], GuitarConfig())
    monkeypatch.setattr(play, "STRING_CONFIDENCE_TEMP", 4.0)
    (soft_c4,) = fuse([_ev(60, 0.0)], [], GuitarConfig())
    (soft_e4,) = fuse([_ev(64, 0.0)], [], GuitarConfig())
    # Ranking (C4 < E4) preserved at both temperatures; hotter temp = lower conf.
    assert sharp_c4.confidence < sharp_e4.confidence
    assert soft_c4.confidence < soft_e4.confidence
    assert soft_e4.confidence < sharp_e4.confidence


def test_confidence_reflects_chord_string_certainty() -> None:
    """In a two-note cluster, an unambiguous member stays high-confidence."""
    # 40 (low E, one string) + 64 (six strings) struck together.
    notes = fuse([_ev(40, 0.0), _ev(64, 0.0)], [], GuitarConfig())
    by_pitch = {n.pitch_midi: n for n in notes}
    assert by_pitch[40].confidence == 1.0
    assert by_pitch[64].confidence < 1.0
