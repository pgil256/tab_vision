from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from tabvision.fusion import chord, fuse
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.playability import MAX_HAND_SPAN
from tabvision.fusion.segment_decoder import SegmentDecoderConfig, partition_clusters
from tabvision.fusion.viterbi import (
    assignment_decoder_context,
    decode_segment_v1_with_analysis,
)
from tabvision.types import AudioEvent, GuitarConfig


def _event(pitch: int, onset: float, duration: float = 0.2) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + duration,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=0.8,
    )


def _cluster_data(events: list[AudioEvent], cfg: GuitarConfig | None = None):
    cfg = cfg or GuitarConfig()
    return [
        (cluster, states)
        for cluster in chord.cluster_events(events)
        if (states := chord.enumerate_chord_states(cluster, cfg))
    ]


def test_partition_boundaries_are_cluster_safe_and_obey_rest_note_and_time_caps() -> None:
    simultaneous = [_event(60, 0.0), _event(64, 0.0)]
    events = simultaneous + [_event(62, 0.2), _event(65, 1.1), _event(67, 5.2)]
    config = SegmentDecoderConfig(max_segment_notes=2, max_segment_s=4.0)
    segments = partition_clusters(_cluster_data(events), config)

    assert [(item.start_cluster, item.end_cluster, item.note_count) for item in segments] == [
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
    ]
    assert segments[0].start_onset_s == segments[0].end_onset_s == 0.0


def test_segment_decoder_preserves_pitch_and_chord_constraints() -> None:
    cfg = GuitarConfig()
    events = [
        _event(60, 0.0),
        _event(64, 0.0),
        _event(67, 0.0),
        _event(62, 0.5),
        _event(65, 1.0),
        _event(69, 1.5),
    ]
    result = decode_segment_v1_with_analysis(events, cfg=cfg, k_paths=3)

    assert len(result.paths) == 3
    for path in result.paths:
        assert [item.pitch_midi for item in path.events] == [item.pitch_midi for item in events]
        for item in path.events:
            assert cfg.tuning_midi[item.string_idx] + item.fret == item.pitch_midi
        for cluster in chord.cluster_events(list(path.events)):
            strings = [item.string_idx for item in cluster]
            pressed = [item.fret for item in cluster if item.fret > 0]
            assert len(strings) == len(set(strings))
            assert not pressed or max(pressed) - min(pressed) <= MAX_HAND_SPAN


def test_open_string_short_clip_and_unplayable_event_are_handled() -> None:
    with assignment_decoder_context("segment-v1"):
        open_result = fuse([_event(64, 0.0)], [])
        empty_result = fuse([_event(20, 0.0)], [])

    assert len(open_result) == 1
    assert (open_result[0].string_idx, open_result[0].fret) == (5, 0)
    assert empty_result == []


def test_k_best_paths_rankings_and_confidence_are_deterministic() -> None:
    events = [_event(60, 0.0), _event(62, 0.4), _event(64, 0.8), _event(65, 1.2)]
    first = decode_segment_v1_with_analysis(events, k_paths=3)
    second = decode_segment_v1_with_analysis(events, k_paths=3)

    assert first == second
    assert [path.cost for path in first.paths] == sorted(path.cost for path in first.paths)
    assert first.paths[0].score_delta_from_best == pytest.approx(0.0)
    assert all(0.0 <= event.confidence <= 1.0 for event in first.paths[0].events)
    assert all(row[0].cost_delta_from_best == pytest.approx(0.0) for row in first.candidate_ranks)


def test_baseline_branch_is_bit_identical_when_segment_decoder_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [_event(60, 0.0), _event(64, 0.0), _event(62, 0.5), _event(67, 1.0)]
    monkeypatch.setenv("TABVISION_ASSIGNMENT_DECODER", "baseline")
    env_selected = fuse(events, [])
    with assignment_decoder_context("baseline"):
        explicitly_selected = fuse(events, [])

    assert [asdict(item) for item in explicitly_selected] == [asdict(item) for item in env_selected]


def test_request_local_decoder_contexts_do_not_leak_between_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabvision.fusion.viterbi as viterbi

    events = [_event(60, 0.0), _event(62, 0.5)]
    original = viterbi.decode_segment_clusters
    segment_calls: list[str] = []

    def recording_decode(*args, **kwargs):
        segment_calls.append("segment-v1")
        return original(*args, **kwargs)

    monkeypatch.setattr(viterbi, "decode_segment_clusters", recording_decode)

    def run(decoder: str) -> tuple[str, tuple[tuple[int, int], ...]]:
        with assignment_decoder_context(decoder):
            output = fuse(events, [])
        return decoder, tuple((item.string_idx, item.fret) for item in output)

    settings = ["baseline", "segment-v1"] * 8
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, settings))

    assert len(segment_calls) == settings.count("segment-v1")
    assert [decoder for decoder, _output in results] == settings
    assert all(output for _decoder, output in results)


def test_nonzero_repeat_weight_is_rejected_until_repeat_ablation_passes() -> None:
    with pytest.raises(ValueError, match="repeat consistency is disabled"):
        SegmentDecoderConfig(repeat_weight=0.1)


def test_candidate_type_remains_pitch_equivalent() -> None:
    cfg = GuitarConfig()
    candidate = Candidate(string_idx=3, fret=5)
    assert cfg.tuning_midi[candidate.string_idx] + candidate.fret == 60
