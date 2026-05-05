"""Unit tests for tabvision.fusion.candidates."""

from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig


def test_open_low_e_yields_one_candidate_at_fret_zero():
    """MIDI 40 = E2 = open low E. Only the low-E string can play it."""
    cfg = GuitarConfig()
    cands = candidate_positions(40, cfg)
    assert len(cands) == 1
    assert cands[0].string_idx == 0
    assert cands[0].fret == 0


def test_a4_yields_multiple_strings():
    """MIDI 69 = A4. Playable on multiple strings."""
    cfg = GuitarConfig()
    cands = candidate_positions(69, cfg)
    # On high E (open=64), fret 5; B (59) fret 10; G (55) fret 14;
    # D (50) fret 19; A (45) fret 24. Low E would need fret 29 — out of range.
    assert {(c.string_idx, c.fret) for c in cands} == {
        (5, 5),
        (4, 10),
        (3, 14),
        (2, 19),
        (1, 24),
    }


def test_results_are_sorted_lowest_fret_first():
    cfg = GuitarConfig()
    cands = candidate_positions(60, cfg)  # C4
    frets = [c.fret for c in cands]
    assert frets == sorted(frets)


def test_capo_excludes_below_capo_frets():
    cfg = GuitarConfig(capo=3)
    # A4 on high-E would be fret 5 (>= capo). On B (59) it's fret 10.
    cands = candidate_positions(69, cfg)
    assert all(c.fret >= 3 for c in cands)


def test_out_of_range_pitch_returns_empty():
    cfg = GuitarConfig()
    # A pitch below the lowest open string.
    assert candidate_positions(20, cfg) == []
    # A pitch above max_fret on every string.
    assert candidate_positions(150, cfg) == []
