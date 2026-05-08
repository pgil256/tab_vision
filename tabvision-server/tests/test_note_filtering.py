"""Tests for note filtering in audio pipeline and fusion engine."""
import sys
sys.path.insert(0, '.')

import pytest
from app.audio_pipeline import (
    DetectedNote, AudioAnalysisConfig,
    _filter_sustain_redetections, _filter_harmonics,
)
from app.fusion_engine import (
    FusionConfig, _prefilter_notes, _limit_chord_sizes,
    fuse_audio_only,
)


def make_note(start, end, midi, amp=0.5, conf=0.7, bend=0.0):
    """Helper to create DetectedNote."""
    return DetectedNote(
        start_time=start, end_time=end, midi_note=midi,
        confidence=conf, amplitude=amp, pitch_bend=bend,
    )


class TestSustainRedetectionFilter:
    """Tests for _filter_sustain_redetections."""

    def test_removes_same_pitch_lower_amplitude(self):
        config = AudioAnalysisConfig(sustain_amplitude_ratio=0.95)
        notes = [
            make_note(0.0, 0.8, 60, amp=0.8),  # original
            make_note(0.4, 0.9, 60, amp=0.7),  # re-detection (amp < 0.8 * 0.95)
        ]
        result = _filter_sustain_redetections(notes, config)
        assert len(result) == 1
        assert result[0].start_time == 0.0

    def test_keeps_new_pluck_higher_amplitude(self):
        config = AudioAnalysisConfig(sustain_amplitude_ratio=0.95)
        notes = [
            make_note(0.0, 0.5, 60, amp=0.5),  # first
            make_note(0.4, 0.9, 60, amp=0.8),  # louder re-pluck (amp > 0.5 * 0.95)
        ]
        result = _filter_sustain_redetections(notes, config)
        assert len(result) == 2

    def test_keeps_notes_beyond_window(self):
        config = AudioAnalysisConfig(sustain_redetection_window=0.5)
        notes = [
            make_note(0.0, 0.3, 60, amp=0.8),
            make_note(0.8, 1.2, 60, amp=0.5),  # beyond 0.5s window AND note ended
        ]
        result = _filter_sustain_redetections(notes, config)
        assert len(result) == 2

    def test_keeps_different_pitches(self):
        config = AudioAnalysisConfig()
        notes = [
            make_note(0.0, 0.8, 60, amp=0.8),
            make_note(0.3, 0.9, 64, amp=0.5),  # different pitch
        ]
        result = _filter_sustain_redetections(notes, config)
        assert len(result) == 2

    def test_keeps_note_when_previous_ended(self):
        config = AudioAnalysisConfig(sustain_amplitude_ratio=0.95)
        notes = [
            make_note(0.0, 0.3, 60, amp=0.8),  # ends at 0.3
            make_note(0.5, 0.9, 60, amp=0.5),  # starts at 0.5, prev ended at 0.3
        ]
        result = _filter_sustain_redetections(notes, config)
        assert len(result) == 2  # prev end (0.3) < note start - 0.1 (0.4)


class TestImprovedHarmonicsFilter:
    """Tests for improved _filter_harmonics."""

    def test_removes_overtone_above(self):
        """Octaves use stricter threshold (0.35) since they occur naturally on guitar."""
        config = AudioAnalysisConfig(harmonic_amplitude_ratio=0.7)
        notes = [
            make_note(0.0, 0.5, 60, amp=0.8),   # fundamental
            make_note(0.0, 0.5, 72, amp=0.2),   # octave above, very low amp
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 1
        assert result[0].midi_note == 60

    def test_removes_sub_harmonic_below(self):
        """Sub-harmonics (octave below) with very low amp are filtered."""
        config = AudioAnalysisConfig(
            filter_sub_harmonics=True, harmonic_amplitude_ratio=0.7
        )
        notes = [
            make_note(0.0, 0.5, 48, amp=0.2),   # sub-harmonic (octave below)
            make_note(0.0, 0.5, 60, amp=0.8),   # louder actual note
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 1
        assert result[0].midi_note == 60

    def test_wider_time_tolerance(self):
        """Harmonics with slight time offset are still caught within tolerance.

        Interval 19 (octave+5th) is also a common musical voicing, so the
        filter applies a stricter amplitude ratio (musical_interval_amp_ratio).
        Only clearly quiet notes — well below the musical-voicing threshold —
        are filtered as harmonics.
        """
        config = AudioAnalysisConfig(
            harmonic_time_tolerance=0.15, harmonic_amplitude_ratio=0.7,
            musical_interval_amp_ratio=0.35, harmonic_protect_amplitude=0.45,
        )
        notes = [
            make_note(0.0, 0.5, 60, amp=0.8),
            make_note(0.10, 0.5, 79, amp=0.2),  # octave+5th (19 semitones), 100ms offset, very quiet
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 1
        assert result[0].midi_note == 60

    def test_keeps_musical_octave_plus_fifth(self):
        """Interval 19 at musically reasonable amplitude is a voicing, not a harmonic."""
        config = AudioAnalysisConfig(
            harmonic_time_tolerance=0.15, harmonic_amplitude_ratio=0.7,
            musical_interval_amp_ratio=0.35, harmonic_protect_amplitude=0.45,
        )
        notes = [
            make_note(0.0, 0.5, 60, amp=0.8),
            make_note(0.10, 0.5, 79, amp=0.30),  # 0.30 > 0.8 * 0.35 = 0.28 → kept
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 2

    def test_keeps_when_sub_harmonics_disabled(self):
        config = AudioAnalysisConfig(
            filter_sub_harmonics=False, harmonic_amplitude_ratio=0.7
        )
        notes = [
            make_note(0.0, 0.5, 48, amp=0.3),   # sub-harmonic
            make_note(0.0, 0.5, 60, amp=0.8),   # fundamental
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 2  # sub-harmonic not filtered

    def test_keeps_both_when_amplitudes_similar(self):
        config = AudioAnalysisConfig(harmonic_amplitude_ratio=0.7)
        notes = [
            make_note(0.0, 0.5, 60, amp=0.7),
            make_note(0.0, 0.5, 72, amp=0.6),  # amp ratio 0.6/0.7 = 0.86 > 0.7
        ]
        result = _filter_harmonics(notes, config)
        assert len(result) == 2


class TestGhostNoteFilter:
    """Tests for ghost note removal in _prefilter_notes."""

    def test_removes_low_amp_overlapping(self):
        config = FusionConfig(
            ghost_note_amplitude_threshold=0.40,
            ghost_note_ratio=0.6,
        )
        notes = [
            make_note(0.0, 1.0, 60, amp=0.8),   # loud note
            make_note(0.1, 0.9, 45, amp=0.3),   # quiet overlapping ghost
        ]
        result = _prefilter_notes(notes, config)
        assert len(result) == 1
        assert result[0].midi_note == 60

    def test_keeps_low_amp_isolated(self):
        config = FusionConfig(
            ghost_note_amplitude_threshold=0.40,
            ghost_note_ratio=0.6,
        )
        notes = [
            make_note(0.0, 0.3, 60, amp=0.8),
            make_note(0.5, 0.9, 45, amp=0.3),  # low amp but doesn't overlap
        ]
        result = _prefilter_notes(notes, config)
        assert len(result) == 2

    def test_keeps_high_amp_overlapping(self):
        config = FusionConfig(
            ghost_note_amplitude_threshold=0.40,
            ghost_note_ratio=0.6,
        )
        notes = [
            make_note(0.0, 1.0, 60, amp=0.8),
            make_note(0.1, 0.9, 64, amp=0.7),  # high amp, not a ghost
        ]
        result = _prefilter_notes(notes, config)
        assert len(result) == 2


class TestChordFragmentFilter:
    """Tests for chord fragment removal in _prefilter_notes."""

    def test_removes_sustaining_note_redetection(self):
        config = FusionConfig(
            chord_fragment_amplitude_ratio=0.9,
            ghost_note_amplitude_threshold=0.20,  # don't trigger ghost filter
        )
        notes = [
            make_note(0.0, 1.5, 60, amp=0.8),   # chord note, sustains to 1.5s
            make_note(0.0, 1.5, 64, amp=0.7),   # chord note
            make_note(0.5, 1.0, 60, amp=0.6),   # re-detection of MIDI 60 (amp < 0.8*0.9)
        ]
        result = _prefilter_notes(notes, config)
        assert len(result) == 2
        assert all(n.start_time == 0.0 for n in result)

    def test_keeps_new_pluck_of_same_pitch(self):
        config = FusionConfig(
            chord_fragment_amplitude_ratio=0.9,
            ghost_note_amplitude_threshold=0.20,
        )
        notes = [
            make_note(0.0, 0.5, 60, amp=0.5),   # first play
            make_note(0.0, 0.5, 64, amp=0.7),
            make_note(0.4, 0.9, 60, amp=0.8),   # louder re-pluck (amp > 0.5*0.9)
        ]
        result = _prefilter_notes(notes, config)
        assert len(result) == 3


class TestChordSizeLimiting:
    """Tests for _limit_chord_sizes."""

    def test_limits_to_max_size(self):
        config = FusionConfig(max_chord_size=3)
        chords = [[
            make_note(0.0, 0.5, 60, amp=0.8),  # keep (highest)
            make_note(0.0, 0.5, 64, amp=0.7),  # keep
            make_note(0.0, 0.5, 67, amp=0.6),  # keep
            make_note(0.0, 0.5, 72, amp=0.3),  # trim (lowest amp)
            make_note(0.0, 0.5, 48, amp=0.2),  # trim (lowest amp)
        ]]
        result = _limit_chord_sizes(chords, config)
        assert len(result) == 1
        assert len(result[0]) == 3
        # Should keep highest amplitude notes
        amps = [n.amplitude for n in result[0]]
        assert min(amps) >= 0.6

    def test_preserves_small_chords(self):
        config = FusionConfig(max_chord_size=4)
        chords = [[
            make_note(0.0, 0.5, 60, amp=0.8),
            make_note(0.0, 0.5, 64, amp=0.7),
        ]]
        result = _limit_chord_sizes(chords, config)
        assert len(result[0]) == 2

    def test_empty_chords(self):
        config = FusionConfig(max_chord_size=4)
        result = _limit_chord_sizes([], config)
        assert result == []


class TestEndToEndFiltering:
    """Integration test: full pipeline with filtering."""

    def test_filtering_reduces_note_count(self):
        """Verify that the filters reduce note count without crashing."""
        config = FusionConfig()
        # Simulate a noisy input with overtones and ghosts
        notes = [
            # Real chord at t=0
            make_note(0.0, 0.8, 60, amp=0.8),
            make_note(0.0, 0.8, 64, amp=0.7),
            make_note(0.0, 0.8, 67, amp=0.6),
            # Ghost note
            make_note(0.1, 0.7, 45, amp=0.2),
            # Sustain re-detection
            make_note(0.4, 0.9, 60, amp=0.3),
            # Real note at t=1.0
            make_note(1.0, 1.5, 62, amp=0.8),
        ]
        tab_notes = fuse_audio_only(notes, capo_fret=0, config=config)
        # Should have fewer notes than input (ghost and sustain removed)
        assert len(tab_notes) < len(notes)
        assert len(tab_notes) >= 3  # at least the real notes
