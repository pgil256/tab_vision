"""Gold-tab ingest: tab parsing, pitch alignment, frame corpus, prior labels.

Local-only feature under the widened SPEC §1.5 carve-out (2026-08-02):
(user video + user gold tab) → onset-stamped ground-truth labels → JPEG
frame corpus for future local video-analysis training, plus the same
labels for the personal position prior.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tabvision.fusion.personal_prior import read_personal_labels
from tabvision.personal.alignment import align_gold_notes
from tabvision.personal.gold_tab import GoldNote, load_gold_tab
from tabvision.personal.video_corpus import (
    ingest_frames,
    matches_to_personal_labels,
)
from tabvision.types import AudioEvent, GuitarConfig

# Standard tuning: E2=40 A2=45 D3=50 G3=55 B3=59 E4=64.

requires_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="opencv not installed (frame extraction needs cv2.imwrite)",
)


def _tab_file(tmp_path: Path, notes: list[dict]) -> Path:
    path = tmp_path / "take.tab.json"
    path.write_text(json.dumps({"notes": notes}), encoding="utf-8")
    return path


def _event(pitch: int, onset: float, confidence: float = 0.9) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.4,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=confidence,
    )


class TestGoldTab:
    def test_tab_convention_string_6_is_low_e(self, tmp_path: Path) -> None:
        notes = load_gold_tab(_tab_file(tmp_path, [{"string": 6, "fret": 3}]))

        assert notes == [GoldNote(string_idx=0, fret=3, pitch_midi=43)]

    def test_declared_pitch_is_cross_checked(self, tmp_path: Path) -> None:
        good = _tab_file(tmp_path, [{"string": 1, "fret": 0, "pitch_midi": 64}])
        assert load_gold_tab(good)[0].pitch_midi == 64

        bad = _tab_file(tmp_path, [{"string": 2, "fret": 0, "pitch_midi": 64}])
        with pytest.raises(ValueError, match="check the string number"):
            load_gold_tab(bad)

    def test_structural_problems_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'notes' list"):
            path = tmp_path / "bad.json"
            path.write_text("[]", encoding="utf-8")
            load_gold_tab(path)
        with pytest.raises(ValueError, match="no notes"):
            load_gold_tab(_tab_file(tmp_path, []))
        with pytest.raises(ValueError, match="string must be 1..6"):
            load_gold_tab(_tab_file(tmp_path, [{"string": 0, "fret": 1}]))

    def test_capo_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="capo 0"):
            load_gold_tab(_tab_file(tmp_path, [{"string": 6, "fret": 0}]), GuitarConfig(capo=2))


def _gold(pitches_as_low_e_frets: list[int]) -> list[GoldNote]:
    return [
        GoldNote(string_idx=0, fret=fret, pitch_midi=40 + fret) for fret in pitches_as_low_e_frets
    ]


class TestAlignment:
    def test_clean_take_aligns_every_note_with_onsets(self) -> None:
        gold = _gold([3, 5, 7])
        events = [_event(43, 1.0), _event(45, 2.0), _event(47, 3.0)]

        result = align_gold_notes(events, gold)

        assert result.matched_fraction == 1.0
        assert [match.onset_s for match in result.matches] == [1.0, 2.0, 3.0]
        assert [match.note.fret for match in result.matches] == [3, 5, 7]

    def test_extra_detections_are_absorbed_as_gaps(self) -> None:
        gold = _gold([3, 5])
        events = [_event(43, 1.0), _event(60, 1.5), _event(45, 2.0)]

        result = align_gold_notes(events, gold)

        assert result.matched_fraction == 1.0
        assert [match.onset_s for match in result.matches] == [1.0, 2.0]

    def test_missed_note_still_aligns_the_rest(self) -> None:
        gold = _gold([3, 5, 7])
        events = [_event(43, 1.0), _event(47, 3.0)]  # the 45 was never detected

        result = align_gold_notes(events, gold)

        assert len(result.matches) == 2
        assert [match.note.fret for match in result.matches] == [3, 7]

    def test_wrong_pitch_is_never_matched(self) -> None:
        gold = _gold([3])
        events = [_event(44, 1.0)]  # semitone off — mistake or misdetection

        result = align_gold_notes(events, gold)

        assert result.matches == ()
        assert result.matched_fraction == 0.0

    def test_empty_inputs_are_empty_not_an_error(self) -> None:
        assert align_gold_notes([], _gold([3])).matched_fraction == 0.0
        assert align_gold_notes([_event(43, 1.0)], []).matches == ()


def _frames(timestamps: list[float], *, value_step: int = 10) -> list[tuple[float, np.ndarray]]:
    return [
        (timestamp, np.full((8, 8, 3), (index + 1) * value_step, dtype=np.uint8))
        for index, timestamp in enumerate(timestamps)
    ]


@requires_cv2
class TestVideoCorpus:
    def test_frames_are_extracted_labelled_and_indexed(self, tmp_path: Path) -> None:
        gold = _gold([3])
        events = [_event(43, 1.0)]
        matches = align_gold_notes(events, gold).matches
        frames = _frames([0.90, 0.98, 1.04, 1.12, 1.20, 1.28])

        summary = ingest_frames(
            frames,
            matches,
            tmp_path / "session",
            source_media="take.mp4",
        )

        # Offsets 0.04/0.12/0.20 after the 1.0 onset all have frames in range.
        assert summary.frames_written == 3
        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "rows.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 3
        for row in rows:
            assert row["string_idx"] == 0
            assert row["fret"] == 3
            assert row["pitch_midi"] == 43
            assert (tmp_path / "session" / row["frame"]).is_file()
        assert [row["frame_timestamp_s"] for row in rows] == [1.04, 1.12, 1.20]
        meta = json.loads((tmp_path / "session" / "meta.json").read_text())
        assert meta["frames_written"] == 3
        assert meta["media"] == "take.mp4"

    def test_instants_without_nearby_frames_are_dropped(self, tmp_path: Path) -> None:
        matches = align_gold_notes([_event(43, 1.0)], _gold([3])).matches
        # Only one frame, near the first offset; the 1.12 / 1.20 instants
        # have nothing within the 0.05 tolerance and must not be invented.
        summary = ingest_frames(
            _frames([1.05]),
            matches,
            tmp_path / "session",
            source_media="take.mp4",
        )

        assert summary.frames_written == 1

    def test_nearest_frame_wins_for_each_instant(self, tmp_path: Path) -> None:
        matches = align_gold_notes([_event(43, 1.0)], _gold([3])).matches

        ingest_frames(
            _frames([1.02, 1.05]),
            matches,
            tmp_path / "session",
            source_media="take.mp4",
            frame_offsets_s=(0.04,),
        )

        rows = [
            json.loads(line)
            for line in (tmp_path / "session" / "rows.jsonl").read_text().splitlines()
        ]
        assert [row["frame_timestamp_s"] for row in rows] == [1.05]


class TestPriorBridge:
    def test_matches_become_gold_tab_labels_and_round_trip(self, tmp_path: Path) -> None:
        from tabvision.fusion.personal_prior import append_personal_labels

        matches = align_gold_notes([_event(43, 1.0, confidence=0.8)], _gold([3])).matches
        labels = matches_to_personal_labels(matches)

        assert len(labels) == 1
        assert labels[0].source == "gold-tab"
        assert labels[0].confidence == 0.8

        store = tmp_path / "labels.jsonl"
        append_personal_labels(store, labels, source_media="take.mp4")
        assert read_personal_labels(store) == labels


class TestIngestScript:
    def test_missing_inputs_error_before_any_heavy_work(self, tmp_path: Path) -> None:
        from scripts.train.ingest_gold_session import main

        with pytest.raises(SystemExit):
            main([str(tmp_path / "absent.mp4"), str(tmp_path / "absent.json")])

    def test_session_id_is_deterministic_per_content(self, tmp_path: Path) -> None:
        from scripts.train.ingest_gold_session import _session_id

        video = tmp_path / "take.mp4"
        video.write_bytes(b"video-bytes")
        tab = _tab_file(tmp_path, [{"string": 6, "fret": 3}])

        first = _session_id(video, tab)
        assert first == _session_id(video, tab)
        video.write_bytes(b"different-bytes")
        assert first != _session_id(video, tab)
