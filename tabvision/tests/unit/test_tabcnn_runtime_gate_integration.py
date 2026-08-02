from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.eval import tabcnn_complementarity_experiment as experiment
from tabvision.eval.tabcnn_posterior import FramePosteriors
from tabvision.types import AudioEvent, GuitarConfig


def _clip(tmp_path: Path, index: int = 0) -> experiment.ExperimentClip:
    audio_path = tmp_path / f"clip-{index}.wav"
    annotation_path = tmp_path / f"clip-{index}.jams"
    event_cache_path = tmp_path / f"clip-{index}.json"
    audio_path.write_bytes(b"wav")
    annotation_path.write_text("{}", encoding="utf-8")
    event_cache_path.write_text("[]", encoding="utf-8")
    return experiment.ExperimentClip(
        clip_id=f"fixture/{index}",
        corpus="guitarset-dev",
        source="fixture",
        split="dev",
        tier="solo",
        player="00",
        mode="solo",
        audio_path=audio_path,
        annotation_path=annotation_path,
        annotation_format="fixture",
        event_cache_path=event_cache_path,
        event_cache_strategy="fixture",
    )


def _model(tmp_path: Path) -> experiment.ModelSpec:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model")
    return experiment.ModelSpec(
        name="fake",
        checkpoint=checkpoint,
        expected_sha256=hashlib.sha256(b"model").hexdigest(),
        family="fixture",
        guitarset_overlap=False,
        frontend_normalization="synthtab",
        artifact_id="fixture",
        source_revision="fixture",
        license_id="fixture",
        license_posture="fixture",
        evaluation_allowed=True,
        shipping_redistribution_allowed=False,
    )


def _paths(tmp_path: Path) -> experiment.RuntimePaths:
    return experiment.RuntimePaths(
        data_root=tmp_path,
        model_root=tmp_path,
        cache_root=tmp_path / "cache",
        guitarset_root=tmp_path / "guitarset",
        gaps_root=tmp_path / "gaps",
        egset12_root=tmp_path / "egset12",
        guitar_techs_root=tmp_path / "guitar-techs",
        q6_dev_cache=tmp_path / "q6-dev",
        q6_player05_cache=tmp_path / "q6-player05",
        q6_gaps_cache=tmp_path / "q6-gaps",
        legacy_guitarset_cache=tmp_path / "legacy",
    )


def test_large_regression_gate_uses_only_tier_player_groups_with_ten_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations: list[tuple[experiment.ExperimentClip, Any]] = []
    for index in range(10):
        evaluations.append((_clip(tmp_path, index), object()))
    for index in range(10, 19):
        evaluations.append(
            (
                replace(
                    _clip(tmp_path, index),
                    tier="small",
                    player="01",
                    corpus="gaps",
                    source="other",
                ),
                object(),
            )
        )

    def aggregate(population: list[Any], **_kwargs: Any) -> Any:
        return SimpleNamespace(
            clips=len(population),
            paired_delta=SimpleNamespace(statistic=-0.01),
        )

    monkeypatch.setattr(experiment, "aggregate_clip_evaluations", aggregate)
    monkeypatch.setattr(
        experiment,
        "_aggregate_dict",
        lambda summary: {"clips": summary.clips},
    )

    results, checks = experiment._large_tier_player_regression_groups(evaluations)

    assert results == {"player:00": {"clips": 10}, "tier:solo": {"clips": 10}}
    assert checks == {"player:00": True, "tier:solo": True}
    assert all(not name.startswith(("corpus:", "source:")) for name in results)


def test_cold_latency_includes_model_frontend_mapping_and_positive_decode_delta() -> None:
    timing_rows = [
        {
            "duration_s": 30.0,
            "timing_seconds": {
                "resample": 1.0,
                "cqt": 2.0,
                "inference": 3.0,
                "mapping": 4.0,
                "decode": {"current": 5.0, "current_plus_tabcnn": 7.0},
            },
        },
        {
            "duration_s": 30.0,
            "timing_seconds": {
                "resample": 1.0,
                "cqt": 1.0,
                "inference": 1.0,
                "mapping": 1.0,
                "decode": {"current": 5.0, "current_plus_tabcnn": 3.0},
            },
        },
    ]

    summary = experiment._cold_latency_summary(
        timing_rows,
        model_load_seconds=2.5,
    )

    assert summary == {
        "evaluated_duration_seconds": 60.0,
        "model_load_seconds": 2.5,
        "warm_added_60s_seconds": 16.0,
        "cold_added_60s_seconds": 18.5,
        "current_decode_only_60s_seconds": 10.0,
    }


def test_attach_candidate_evidence_records_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = AudioEvent(0.0, 0.2, 64, 0.8, 0.9)
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        experiment,
        "attach_tabcnn_priors",
        lambda events, *_args, **_kwargs: list(events),
    )

    attached, elapsed = experiment._attach_candidate_evidence(
        [event],
        [None],
        cfg=GuitarConfig(),
    )

    assert attached == [event]
    assert elapsed == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("clock_values", "evidence_seconds", "expected_incremental"),
    [
        ((1.0, 1.4, 2.0, 2.1), 0.2, 0.0),
        ((1.0, 1.1, 2.0, 2.5), 0.2, 0.6),
    ],
)
def test_guitarset_candidate_prep_is_incremental_over_current(
    monkeypatch: pytest.MonkeyPatch,
    clock_values: tuple[float, float, float, float],
    evidence_seconds: float,
    expected_incremental: float,
) -> None:
    event = AudioEvent(0.0, 0.2, 64, 0.8, 0.9)
    clock = iter(clock_values)
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        experiment,
        "apply_pitch_position_prior",
        lambda events, *_args, **_kwargs: list(events),
    )

    current, candidate, timing = experiment._apply_guitarset_position_pair(
        [event],
        [event],
        object(),
        cfg=GuitarConfig(),
        candidate_evidence_seconds=evidence_seconds,
    )

    assert current == [event]
    assert candidate == [event]
    assert timing["candidate_incremental_prep"] == pytest.approx(expected_incremental)


def test_cache_summary_requires_one_model_and_records_fresh_process_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path)
    model = _model(tmp_path)
    paths = _paths(tmp_path)
    with pytest.raises(RuntimeError, match="one fresh process per model"):
        experiment.cache_all_posteriors(
            [clip],
            [model, model],
            paths,
            chunk_size=256,
        )

    cache_path = tmp_path / "posterior.npz"
    cache_path.write_bytes(b"posterior")
    digest = "a" * 64
    cached = experiment.CachedPosterior(
        frames=FramePosteriors(
            probabilities=np.full((1, 6, 21), 1.0 / 21.0, dtype=np.float32),
            times_s=np.asarray([0.0]),
        ),
        metadata={
            "cache_key": "cache-key",
            "posterior_sha256": digest,
            "duration_s": 3.0,
            "timing_seconds": {
                "audio_load": 0.01,
                "resample": 0.1,
                "cqt": 0.2,
                "inference": 0.3,
            },
        },
        path=cache_path,
    )
    clock = iter((1.0, 1.25))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        experiment,
        "evaluation_code_revision",
        lambda: {"git_revision": "git", "evaluation_sha256": "code"},
    )
    monkeypatch.setattr(experiment, "runtime_manifest", lambda: {"runtime": "fixture"})
    monkeypatch.setattr(experiment, "load_model_backend", lambda _spec: object())
    monkeypatch.setattr(
        experiment,
        "ensure_posterior_cache",
        lambda *_args, **_kwargs: (cached, False),
    )
    monkeypatch.setattr(
        experiment,
        "verify_posterior_determinism",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(experiment, "peak_rss_bytes", lambda: 123_456)

    summary = experiment.cache_all_posteriors(
        [clip],
        [model],
        paths,
        chunk_size=256,
    )

    assert summary["fresh_process_per_model"] is True
    assert summary["model_load_seconds"] == pytest.approx(0.25)
    assert summary["peak_rss_bytes"] == 123_456
    assert summary["posteriors"][0]["cache_key"] == "cache-key"
    assert summary["posteriors"][0]["posterior_sha256"] == digest


def test_cache_summary_reuses_verified_marker_for_resumed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path)
    model = _model(tmp_path)
    cache_path = tmp_path / "resumed-posterior.npz"
    cache_path.write_bytes(b"posterior")
    cached = experiment.CachedPosterior(
        frames=FramePosteriors(
            probabilities=np.full((1, 6, 21), 1.0 / 21.0, dtype=np.float32),
            times_s=np.asarray([0.0]),
        ),
        metadata={
            "cache_key": "resumed-key",
            "posterior_sha256": "b" * 64,
            "duration_s": 3.0,
            "timing_seconds": {
                "audio_load": 0.01,
                "resample": 0.1,
                "cqt": 0.2,
                "inference": 0.3,
            },
        },
        path=cache_path,
    )
    marker = {"cache_key": "resumed-key", "verified": True}
    clock = iter((1.0, 1.2))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        experiment,
        "evaluation_code_revision",
        lambda: {"git_revision": "git", "evaluation_sha256": "code"},
    )
    monkeypatch.setattr(experiment, "runtime_manifest", lambda: {"runtime": "fixture"})
    monkeypatch.setattr(experiment, "load_model_backend", lambda _spec: object())
    monkeypatch.setattr(
        experiment,
        "ensure_posterior_cache",
        lambda *_args, **_kwargs: (cached, True),
    )
    monkeypatch.setattr(
        experiment,
        "posterior_determinism_status",
        lambda _cached: marker,
    )
    monkeypatch.setattr(
        experiment,
        "verify_posterior_determinism",
        lambda *_args, **_kwargs: pytest.fail("verified markers must be reused"),
    )
    monkeypatch.setattr(experiment, "peak_rss_bytes", lambda: 123_456)

    summary = experiment.cache_all_posteriors(
        [clip],
        [model],
        _paths(tmp_path),
        chunk_size=256,
    )

    assert summary["posteriors"][0]["determinism"] == marker
    assert summary["posteriors"][0]["determinism_reused"] is True
