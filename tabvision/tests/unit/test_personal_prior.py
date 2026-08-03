"""Personal position-prior labeller: harvest, store, builder, policy, CLI.

The SPEC §1.5 carve-out (2026-08-02) admits exactly one use of the user's
own recordings: opt-in harvest of (pitch, string, fret) labels — audio
pitch joined against FretCam's locked position windows — into a *local*
personal position prior. Track C priced the ceiling at +0.0305
(`docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.md`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabvision.cli import _build_parser, main
from tabvision.errors import ConfigurationError
from tabvision.fusion.inference_policy import resolve_inference_policy
from tabvision.fusion.personal_prior import (
    PersonalLabel,
    append_personal_labels,
    build_personal_prior_payload,
    harvest_position_labels,
    read_personal_labels,
    write_personal_prior_artifact,
)
from tabvision.fusion.position_prior import load_pitch_position_prior
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig
from tabvision.video.position import PositionWindowObservation

# Standard tuning: E2=40 A2=45 D3=50 G3=55 B3=59 E4=64.


def _event(pitch: int, onset: float = 1.0, confidence: float = 0.9) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.5,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=confidence,
    )


def _window(
    timestamp: float = 1.0,
    *,
    position: int = 5,
    frets: tuple[int, ...] = (5, 6, 7, 8),
    confidence: float = 0.9,
    state: str = "locked",
) -> PositionWindowObservation:
    return PositionWindowObservation(
        timestamp_s=timestamp,
        position=position,
        window_frets=frets,
        confidence=confidence,
        state=state,  # type: ignore[arg-type]
    )


class TestHarvest:
    def test_unambiguous_candidate_is_harvested(self) -> None:
        # Pitch 46 at position 5: only string 0 fret 6 is consistent.
        labels = harvest_position_labels([_event(46)], [_window()])

        assert labels == [
            PersonalLabel(pitch_midi=46, string_idx=0, fret=6, onset_s=1.0, confidence=0.9)
        ]

    def test_ambiguous_pitch_abstains(self) -> None:
        # Pitch 50 at position 5: string 1 fret 5 (in window) AND the open
        # D string are both consistent — the harvest must not guess.
        assert harvest_position_labels([_event(50)], [_window()]) == []

    def test_open_string_is_inferred_when_nothing_else_fits(self) -> None:
        # Pitch 40 has exactly one playable candidate anywhere: open low E.
        labels = harvest_position_labels([_event(40)], [_window()])

        assert labels == [
            PersonalLabel(pitch_midi=40, string_idx=0, fret=0, onset_s=1.0, confidence=0.9)
        ]

    def test_confidence_is_the_weaker_of_the_two_channels(self) -> None:
        labels = harvest_position_labels([_event(46, confidence=0.95)], [_window(confidence=0.6)])

        assert labels[0].confidence == 0.6

    def test_gap_state_and_confidence_gates(self) -> None:
        event = _event(46)

        assert harvest_position_labels([event], [_window(timestamp=1.5)]) == []
        assert harvest_position_labels([event], [_window(state="holding")]) == []
        assert harvest_position_labels([event], [_window(confidence=0.3)]) == []
        assert harvest_position_labels([_event(46, confidence=0.3)], [_window()]) == []

    def test_nearest_window_wins(self) -> None:
        # Two locked windows straddle the onset; the closer one (position 1,
        # where pitch 46 has a *different* unique candidate) must be used.
        windows = [
            _window(timestamp=0.9, position=1, frets=(1, 2, 3, 4)),
            _window(timestamp=1.2, position=5),
        ]

        labels = harvest_position_labels([_event(46)], windows)

        assert labels == [
            PersonalLabel(pitch_midi=46, string_idx=1, fret=1, onset_s=1.0, confidence=0.9)
        ]

    def test_capo_sessions_are_refused(self) -> None:
        with pytest.raises(ValueError, match="capo 0"):
            harvest_position_labels([_event(46)], [_window()], GuitarConfig(capo=2))

    def test_no_observations_is_empty_not_an_error(self) -> None:
        assert harvest_position_labels([_event(46)], []) == []


class TestStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = tmp_path / "labels.jsonl"
        first = harvest_position_labels([_event(46)], [_window()])
        second = harvest_position_labels([_event(40, onset=2.0)], [_window(timestamp=2.0)])

        append_personal_labels(store, first, source_media="a.mov")
        append_personal_labels(store, second, source_media="b.mov")

        assert read_personal_labels(store) == first + second
        rows = [json.loads(line) for line in store.read_text().splitlines()]
        assert {row["media"] for row in rows} == {"a.mov", "b.mov"}
        assert all(row["schema_version"] == 1 for row in rows)

    def test_malformed_row_raises(self, tmp_path: Path) -> None:
        store = tmp_path / "labels.jsonl"
        store.write_text('{"schema_version": 1, "pitch_midi": "not-a-pitch"}\n')

        with pytest.raises(ValueError, match="invalid label fields"):
            read_personal_labels(store)

    def test_wrong_schema_raises(self, tmp_path: Path) -> None:
        store = tmp_path / "labels.jsonl"
        store.write_text('{"schema_version": 99}\n')

        with pytest.raises(ValueError, match="unsupported label schema"):
            read_personal_labels(store)


def _labels(pitch: int, string_idx: int, fret: int, count: int) -> list[PersonalLabel]:
    return [
        PersonalLabel(
            pitch_midi=pitch,
            string_idx=string_idx,
            fret=fret,
            onset_s=float(index),
            confidence=0.9,
        )
        for index in range(count)
    ]


class TestBuilder:
    def test_pure_personal_artifact_loads_and_prefers_the_labelled_position(
        self, tmp_path: Path
    ) -> None:
        payload = build_personal_prior_payload(
            _labels(46, 0, 6, 6) + _labels(45, 1, 0, 2),  # 45 is below the floor
            merge_population=None,
        )
        artifact = tmp_path / "personal.json"
        write_personal_prior_artifact(artifact, payload)

        assert payload["personalized_pitches"] == [46]
        assert payload["counts"] == [[46, 0, 6, 6]]
        prior = load_pitch_position_prior(str(artifact))
        matrix = prior.matrix_for_pitch(46)
        assert matrix is not None
        assert float(matrix[0, 6]) == max(float(value) for value in matrix.flat)

    def test_merged_artifact_switches_per_pitch(self) -> None:
        payload = build_personal_prior_payload(_labels(46, 0, 6, 6))

        assert payload["population_base"] == "guitarset-v1"
        counts = payload["counts"]
        assert isinstance(counts, list)
        pitch_46_rows = [row for row in counts if row[0] == 46]
        # Personalized pitch: personal counts only, population rows dropped.
        assert pitch_46_rows == [[46, 0, 6, 6]]
        # Every other pitch keeps the population's counts.
        assert any(row[0] != 46 for row in counts)

    def test_out_of_range_label_raises(self) -> None:
        bad = [PersonalLabel(pitch_midi=46, string_idx=9, fret=6, onset_s=0.0, confidence=0.9)]

        with pytest.raises(ValueError, match="out of range"):
            build_personal_prior_payload(bad, merge_population=None)


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "personal.json"
    write_personal_prior_artifact(
        artifact,
        build_personal_prior_payload(_labels(46, 0, 6, 6), merge_population=None),
    )
    return artifact


class TestPolicy:
    def test_personal_path_resolves_with_hash_identity_and_no_sequence_pairing(
        self, tmp_path: Path
    ) -> None:
        artifact = _write_artifact(tmp_path)

        policy = resolve_inference_policy(
            requested_position_prior=str(artifact),
            requested_sequence_prior="auto",
            requested_string_evidence="none",
            cfg=GuitarConfig(),
            session=SessionConfig(),
            audio_backend_name="highres",
        )

        assert policy.resolved_position_prior == str(artifact)
        assert policy.resolved_sequence_prior == "none"
        assert policy.artifacts[0].name == "personal:personal.json"
        assert len(policy.artifacts[0].sha256) == 64
        assert "no validated sequence pairing" in policy.resolution_reason
        # The pipeline loads exactly what the policy resolved.
        assert load_pitch_position_prior(policy.resolved_position_prior) is not None

    def test_missing_personal_file_fails_at_policy_time(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            resolve_inference_policy(
                requested_position_prior=str(tmp_path / "absent.json"),
                requested_sequence_prior="none",
                requested_string_evidence="none",
                cfg=GuitarConfig(),
                session=SessionConfig(),
                audio_backend_name="highres",
            )

    def test_wrong_schema_fails_at_policy_time(self, tmp_path: Path) -> None:
        artifact = tmp_path / "bad.json"
        artifact.write_text('{"schema_version": 2}', encoding="utf-8")

        with pytest.raises(ConfigurationError, match="unsupported schema"):
            resolve_inference_policy(
                requested_position_prior=str(artifact),
                requested_sequence_prior="none",
                requested_string_evidence="none",
                cfg=GuitarConfig(),
                session=SessionConfig(),
                audio_backend_name="highres",
            )


class TestCli:
    def test_position_prior_accepts_json_path_and_rejects_junk(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(["transcribe", "in.mov", "--position-prior", "mine.json"])
        assert args.position_prior == "mine.json"

        with pytest.raises(SystemExit):
            parser.parse_args(["transcribe", "in.mov", "--position-prior", "bogus"])

    def test_harvest_flag_requires_fretcam(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        input_path = tmp_path / "input.mov"
        input_path.write_bytes(b"pipeline is injected")
        monkeypatch.setattr(
            "tabvision.pipeline.run_pipeline_with_artifacts",
            lambda *a, **k: pytest.fail("must be rejected before the pipeline runs"),
        )

        rc = main(
            [
                "transcribe",
                str(input_path),
                "--harvest-personal-labels",
                str(tmp_path / "labels.jsonl"),
                "--no-preflight",
            ]
        )

        assert rc == 2
        assert "requires --video-backend fretcam" in capsys.readouterr().err

    def test_harvest_flag_refuses_capo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        input_path = tmp_path / "input.mov"
        input_path.write_bytes(b"pipeline is injected")
        monkeypatch.setattr(
            "tabvision.pipeline.run_pipeline_with_artifacts",
            lambda *a, **k: pytest.fail("must be rejected before the pipeline runs"),
        )

        rc = main(
            [
                "transcribe",
                str(input_path),
                "--video-backend",
                "fretcam",
                "--capo",
                "2",
                "--harvest-personal-labels",
                str(tmp_path / "labels.jsonl"),
                "--no-preflight",
            ]
        )

        assert rc == 2
        assert "capo 0" in capsys.readouterr().err

    def test_harvest_appends_labels_to_the_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tabvision.fusion.inference_policy import ResolvedInferencePolicy
        from tabvision.pipeline import PipelineArtifacts

        input_path = tmp_path / "input.mov"
        input_path.write_bytes(b"pipeline is injected")
        store = tmp_path / "labels.jsonl"
        artifacts = PipelineArtifacts(
            tab_events=(),
            audio_events=(_event(46),),
            policy=ResolvedInferencePolicy(
                requested_position_prior="none",
                resolved_position_prior="none",
                requested_sequence_prior="none",
                resolved_sequence_prior="none",
                requested_string_evidence="none",
                resolved_string_evidence="none",
                requested_assignment_decoder="auto",
                resolved_assignment_decoder="baseline",
                assignment_decoder_reason="test",
                artifacts=(),
                resolution_reason="test",
            ),
            resolved_video_backend="fretcam",
            position_observation_count=1,
            position_observations=(_window(),),
        )
        monkeypatch.setattr(
            "tabvision.pipeline.run_pipeline_with_artifacts",
            lambda *a, **k: artifacts,
        )

        rc = main(
            [
                "transcribe",
                str(input_path),
                "--output",
                str(tmp_path / "out.tab"),
                "--video-backend",
                "fretcam",
                "--harvest-personal-labels",
                str(store),
                "--no-preflight",
            ]
        )

        assert rc == 0
        assert read_personal_labels(store) == [
            PersonalLabel(pitch_midi=46, string_idx=0, fret=6, onset_s=1.0, confidence=0.9)
        ]
        assert "harvested 1 personal labels" in capsys.readouterr().err


class TestBuilderScript:
    def test_end_to_end_store_to_artifact(self, tmp_path: Path) -> None:
        from scripts.train.build_personal_prior import main as build_main

        store = tmp_path / "labels.jsonl"
        append_personal_labels(
            store,
            _labels(46, 0, 6, 6),
            source_media="session.mov",
        )
        artifact = tmp_path / "personal.json"

        rc = build_main([str(store), "-o", str(artifact), "--merge-population", "none"])

        assert rc == 0
        prior = load_pitch_position_prior(str(artifact))
        assert prior.matrix_for_pitch(46) is not None

    def test_output_must_be_json(self, tmp_path: Path) -> None:
        from scripts.train.build_personal_prior import main as build_main

        with pytest.raises(SystemExit):
            build_main([str(tmp_path / "labels.jsonl"), "-o", str(tmp_path / "prior.bin")])
