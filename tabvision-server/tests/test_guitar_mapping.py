"""Tests for guitar_mapping module."""
import pytest
from app.guitar_mapping import get_candidate_positions, pick_lowest_fret, Position


class TestGetCandidatePositions:
    """Tests for MIDI note to fret/string position mapping."""

    def test_midi_69_a4_returns_three_positions(self):
        """MIDI 69 (A4) should be playable at 3 positions in standard tuning."""
        # A4 = MIDI 69
        # String 1 (high E, open=64): fret 5 (64+5=69) ✓
        # String 2 (B, open=59): fret 10 (59+10=69) ✓
        # String 3 (G, open=55): fret 14 (55+14=69) ✓
        # String 4 (D, open=50): fret 19 (50+19=69) ✓
        # String 5 (A, open=45): fret 24 (45+24=69) ✓
        # String 6 (low E, open=40): fret 29 - beyond fret 24, excluded

        positions = get_candidate_positions(midi_note=69)

        # Should return positions sorted by fret ascending
        assert len(positions) == 5
        assert positions[0] == Position(string=1, fret=5)
        assert positions[1] == Position(string=2, fret=10)
        assert positions[2] == Position(string=3, fret=14)
        assert positions[3] == Position(string=4, fret=19)
        assert positions[4] == Position(string=5, fret=24)

    def test_midi_40_low_e_returns_open_string(self):
        """MIDI 40 (E2) should return open low E string."""
        # E2 = MIDI 40 = open string 6
        positions = get_candidate_positions(midi_note=40)

        assert len(positions) >= 1
        assert positions[0] == Position(string=6, fret=0)

    def test_capo_excludes_positions_below_capo(self):
        """With capo on fret 2, positions at frets 0-1 should be excluded."""
        # MIDI 64 (E4) = open string 1 (fret 0)
        # With capo at fret 2, fret 0 is not playable
        positions_no_capo = get_candidate_positions(midi_note=64, capo_fret=0)
        positions_capo_2 = get_candidate_positions(midi_note=64, capo_fret=2)

        # Without capo, should include fret 0
        assert Position(string=1, fret=0) in positions_no_capo

        # With capo at 2, fret 0 should be excluded
        assert Position(string=1, fret=0) not in positions_capo_2

    def test_out_of_range_low_returns_empty(self):
        """MIDI note below guitar range returns empty list."""
        # MIDI 39 is below low E (40)
        positions = get_candidate_positions(midi_note=39)
        assert positions == []

    def test_out_of_range_high_returns_empty(self):
        """MIDI note above guitar range returns empty list."""
        # MIDI 89 (F6) is above highest playable note (E4 + 24 = MIDI 88)
        positions = get_candidate_positions(midi_note=89)
        assert positions == []


class TestPickLowestFret:
    """Tests for the lowest-fret selection heuristic."""

    def test_picks_lowest_fret_from_candidates(self):
        """Should return the position with the lowest fret number."""
        candidates = [
            Position(string=3, fret=14),
            Position(string=1, fret=5),
            Position(string=2, fret=10),
        ]

        result = pick_lowest_fret(candidates)

        assert result == Position(string=1, fret=5)

    def test_empty_candidates_returns_none(self):
        """Empty candidate list returns None."""
        result = pick_lowest_fret([])
        assert result is None
