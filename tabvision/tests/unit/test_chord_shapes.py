"""A5 — chord-shape dictionary + per-cluster shape bonus.

Covers the port itself (voicing DB built in v1 string convention), the
overlap scorer, the ``chord_shape_cost`` emission term (no-op at the default
bonus, count-scaled reward, min-notes gating, live rebinding for the A3 sweep),
and two end-to-end ``fuse`` properties:

- **single-line invariance**: singleton clusters never match, so the bonus
  cannot change a single-line decode at any magnitude;
- **chord recovery**: a strong bonus pulls a chord cluster onto the matching
  canonical voicing.
"""

from __future__ import annotations

import pytest

import tabvision.fusion.chord_shapes as cs
from tabvision.fusion import fuse
from tabvision.types import AudioEvent, GuitarConfig

# E major open, v1 (string_idx, fret) convention — the reference for the
# port's string mapping (v0 {6:0,5:2,4:2,3:1,2:0,1:0}).
_E_MAJOR_OPEN = frozenset({(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)})
# The six MIDI pitches E major open produces (E2 B2 E3 G#3 B3 E4).
_E_MAJOR_OPEN_MIDI = [40, 47, 52, 56, 59, 64]


def _ev(midi: int, t: float, confidence: float = 0.8) -> AudioEvent:
    return AudioEvent(
        onset_s=t, offset_s=t + 0.25, pitch_midi=midi, velocity=0.8, confidence=confidence
    )


# ---------- the ported voicing dictionary ----------


def test_voicing_db_built_with_expected_size() -> None:
    # 22 open + 12 frets x 6 barre shapes (72) + 13 frets x 3 power (39) = 133.
    assert len(cs.VOICINGS) == 133


def test_every_voicing_is_well_formed() -> None:
    for v in cs.VOICINGS:
        assert v.positions, f"empty voicing {v.name}"
        for string_idx, fret in v.positions:
            assert 0 <= string_idx <= 5, f"{v.name}: string_idx {string_idx} out of range"
            assert 0 <= fret <= 24, f"{v.name}: fret {fret} out of range"
        # A voicing can't press two frets on one string.
        strings = [s for s, _ in v.positions]
        assert len(strings) == len(set(strings)), f"{v.name}: duplicate string"


def test_string_convention_matches_spec_low_e_zero() -> None:
    """E major open must land on string_idx 0=low E .. 5=high E (SPEC §8)."""
    e_major = next(v for v in cs.VOICINGS if v.name == "E major open")
    assert e_major.positions == _E_MAJOR_OPEN


def test_muted_strings_are_dropped() -> None:
    """A minor open mutes the low E (v0 string 6 -> None), so string_idx 0 absent."""
    a_minor = next(v for v in cs.VOICINGS if v.name == "A minor open")
    assert a_minor.positions == frozenset({(1, 0), (2, 2), (3, 2), (4, 1), (5, 0)})
    assert all(s != 0 for s, _ in a_minor.positions)


# ---------- best_shape_overlap ----------


def test_overlap_full_voicing_is_its_size() -> None:
    assert cs.best_shape_overlap(_E_MAJOR_OPEN) == 6


def test_overlap_partial_subset() -> None:
    subset = {(0, 0), (1, 2), (2, 2)}  # three of E major open
    assert cs.best_shape_overlap(subset) == 3


def test_overlap_zero_off_every_shape() -> None:
    # Frets >= 15 exceed every generated voicing (max is a barre at fret 14).
    assert cs.best_shape_overlap({(0, 15), (1, 16), (2, 17)}) == 0


def test_overlap_empty_is_zero() -> None:
    assert cs.best_shape_overlap([]) == 0


# ---------- chord_shape_cost ----------


def _state(positions: set[tuple[int, int]]) -> tuple:
    from tabvision.fusion.candidates import Candidate

    return tuple(Candidate(string_idx=s, fret=f) for s, f in positions)


def test_default_bonus_is_the_gated_value() -> None:
    # Shipped default is the A5-gated 0.1 -> a full-shape match is rewarded
    # (-0.1 * overlap 6).
    assert cs.CHORD_SHAPE_BONUS == pytest.approx(0.1)
    assert cs.chord_shape_cost(_state(set(_E_MAJOR_OPEN))) == pytest.approx(-0.6)


def test_zero_bonus_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # The off switch: bonus 0.0 -> exact no-op even for a full-shape match.
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.0)
    assert cs.chord_shape_cost(_state(set(_E_MAJOR_OPEN))) == 0.0


def test_cost_count_scaled_when_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.5)
    # Full E major open: overlap 6 -> -0.5 * 6.
    assert cs.chord_shape_cost(_state(set(_E_MAJOR_OPEN))) == pytest.approx(-3.0)
    # A 3-note subset: overlap 3 -> -0.5 * 3.
    assert cs.chord_shape_cost(_state({(0, 0), (1, 2), (2, 2)})) == pytest.approx(-1.5)


def test_cost_below_min_notes_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.5)
    # A dyad on a real shape still overlaps only 2 < default min 3 -> no reward.
    assert cs.chord_shape_cost(_state({(0, 0), (1, 2)})) == 0.0
    # A singleton -> overlap 1 -> no reward (single-line tier untouched).
    assert cs.chord_shape_cost(_state({(3, 1)})) == 0.0


def test_cost_min_notes_is_rebindable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.5)
    monkeypatch.setattr(cs, "CHORD_SHAPE_MIN_NOTES", 2)
    # With the threshold lowered to 2, the dyad now rewards -0.5 * 2.
    assert cs.chord_shape_cost(_state({(0, 0), (1, 2)})) == pytest.approx(-1.0)


def test_cost_zero_for_unmatched_triad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.5)
    assert cs.chord_shape_cost(_state({(0, 15), (1, 16), (2, 17)})) == 0.0


# ---------- end-to-end fuse() ----------


def test_single_line_decode_invariant_to_bonus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Well-separated notes are all singleton clusters; the bonus can never
    fire, so the decode is bit-identical at any magnitude."""
    cfg = GuitarConfig()
    events = [_ev(60, 0.0), _ev(62, 0.5), _ev(64, 1.0), _ev(65, 1.5)]
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.0)
    off = [(t.string_idx, t.fret) for t in fuse(events, [], cfg)]
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 100.0)
    boosted = [(t.string_idx, t.fret) for t in fuse(events, [], cfg)]
    assert off == boosted


def test_strong_bonus_recovers_chord_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Six simultaneous E-major notes, decoded under a dominating bonus, must
    land exactly on the E major open voicing."""
    cfg = GuitarConfig()
    events = [_ev(m, 0.0) for m in _E_MAJOR_OPEN_MIDI]
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 10.0)
    out = fuse(events, [], cfg)
    decoded = frozenset((t.string_idx, t.fret) for t in out)
    assert decoded == _E_MAJOR_OPEN


def test_bonus_does_not_reduce_shape_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strong bonus can only steer a chord toward more shape overlap, never
    less, than the no-bonus decode."""
    cfg = GuitarConfig()
    events = [_ev(m, 0.0) for m in _E_MAJOR_OPEN_MIDI]
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 0.0)
    off = frozenset((t.string_idx, t.fret) for t in fuse(events, [], cfg))
    monkeypatch.setattr(cs, "CHORD_SHAPE_BONUS", 10.0)
    boosted = frozenset((t.string_idx, t.fret) for t in fuse(events, [], cfg))
    assert cs.best_shape_overlap(boosted) >= cs.best_shape_overlap(off)
