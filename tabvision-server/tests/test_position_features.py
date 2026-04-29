"""Smoke tests for the position-selection feature emitter.

Step 1 of the learned-fusion plan: every user-visible single-note position
selection in fuse_audio_only / fuse_audio_video must append a PositionDecision
to FusionConfig._feature_events when emit_position_features=True, and emit
nothing when the flag is off.
"""
from app.audio_pipeline import DetectedNote
from app.fusion_engine import (
    FusionConfig,
    PositionDecision,
    fuse_audio_only,
)


def _make_note(t: float, midi: int, amp: float = 0.7, conf: float = 0.85) -> DetectedNote:
    return DetectedNote(
        start_time=t, end_time=t + 0.2, midi_note=midi,
        confidence=conf, amplitude=amp, pitch_bend=0.0,
    )


class TestEmissionGating:
    def test_flag_off_no_emission(self):
        config = FusionConfig()
        config.enable_prefiltering = False  # keep notes deterministic for the test
        # Notes default: emit_position_features=False, _feature_events=None.
        notes = [_make_note(1.0, 64), _make_note(1.5, 67), _make_note(2.0, 60)]
        fuse_audio_only(notes, capo_fret=0, config=config)
        assert config._feature_events is None

    def test_flag_on_records_one_event_per_single_note(self):
        config = FusionConfig()
        config.enable_prefiltering = False
        config.emit_position_features = True
        config._feature_events = []

        notes = [_make_note(1.0, 64), _make_note(1.5, 67), _make_note(2.0, 60)]
        fuse_audio_only(notes, capo_fret=0, config=config)

        # Three single-note events expected; chord case isn't instrumented yet.
        assert len(config._feature_events) == 3
        for evt in config._feature_events:
            assert isinstance(evt, PositionDecision)
            assert evt.is_chord is False
            assert evt.chord_size == 1
            assert evt.num_candidates == len(evt.candidates)
            assert evt.candidates  # at least one candidate
            picked = [c for c in evt.candidates if c['is_heuristic_pick']]
            assert len(picked) == 1


class TestEventSchema:
    def test_candidate_features_present(self):
        config = FusionConfig()
        config.enable_prefiltering = False
        config.emit_position_features = True
        config._feature_events = []

        # One note that has multiple candidates at standard tuning.
        # MIDI 64 (E4) maps to: e-string fret 0, B-string fret 5,
        # G-string fret 9, D-string fret 14, A-string fret 19, E-string fret 24.
        fuse_audio_only([_make_note(1.0, 64)], capo_fret=0, config=config)

        assert len(config._feature_events) == 1
        evt = config._feature_events[0]
        assert evt.midi_note == 64
        assert evt.onset_time == 1.0
        assert evt.num_candidates >= 2  # multiple positions for E4
        for c in evt.candidates:
            assert 'cand_string' in c
            assert 'cand_fret' in c
            assert 'heuristic_score' in c
            assert isinstance(c['heuristic_score'], float)
            assert 'is_heuristic_pick' in c
            # No previous note -> dist_prev_* are None.
            assert c['dist_prev_fret'] is None
            assert c['dist_prev_string'] is None

    def test_seconds_since_prev_advances(self):
        config = FusionConfig()
        config.enable_prefiltering = False
        config.emit_position_features = True
        config._feature_events = []

        fuse_audio_only(
            [_make_note(1.0, 64), _make_note(1.5, 67), _make_note(2.4, 60)],
            capo_fret=0, config=config,
        )

        events = config._feature_events
        assert events[0].seconds_since_prev is None
        # Second event sees ~0.5s gap from first, third sees ~0.9s from second.
        assert events[1].seconds_since_prev is not None
        assert abs(events[1].seconds_since_prev - 0.5) < 0.01
        assert events[2].seconds_since_prev is not None
        assert abs(events[2].seconds_since_prev - 0.9) < 0.01

    def test_selected_matches_heuristic_pick_when_no_video(self):
        config = FusionConfig()
        config.enable_prefiltering = False
        config.emit_position_features = True
        config._feature_events = []

        fuse_audio_only([_make_note(1.0, 64)], capo_fret=0, config=config)

        evt = config._feature_events[0]
        picked = next(c for c in evt.candidates if c['is_heuristic_pick'])
        assert evt.selected_string == picked['cand_string']
        assert evt.selected_fret == picked['cand_fret']
