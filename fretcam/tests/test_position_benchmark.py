from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import fretcam.position_benchmark as position_benchmark
from fretcam.position_benchmark import (
    BenchmarkManifest,
    BenchmarkSequence,
    FramePrediction,
    PositionLabel,
    _apply_annotation_transform,
    default_manifest_path,
    load_manifest,
    score_predictions,
    validate_manifest,
)


def _label(
    start: float,
    end: float,
    *,
    state: str = "stable",
    position: int | None = 5,
    occlusion_rect: tuple[float, float, float, float] | None = None,
) -> PositionLabel:
    return PositionLabel(
        start_s=start,
        end_s=end,
        state=state,  # type: ignore[arg-type]
        position=position,
        technique="note",
        visibility="full_neck",
        lighting="bright",
        verification="frame_reviewed",
        notes="test",
        occlusion_rect=occlusion_rect,
    )


def _sequence(
    sequence_id: str,
    source: str,
    split: str,
    labels: tuple[PositionLabel, ...],
) -> BenchmarkSequence:
    return BenchmarkSequence(
        sequence_id=sequence_id,
        source=source,
        split=split,  # type: ignore[arg-type]
        start_s=min(label.start_s for label in labels),
        end_s=max(label.end_s for label in labels),
        labels=labels,
    )


def _manifest(*sequences: BenchmarkSequence) -> BenchmarkManifest:
    return BenchmarkManifest(
        version=1,
        name="test",
        corpus="GAPS",
        corpus_license="CC-BY-NC-SA-4.0",
        public_only=True,
        sample_fps=10.0,
        annotation_policy="test",
        sequences=tuple(sequences),
    )


def _prediction(
    sequence: BenchmarkSequence,
    timestamp: float,
    *,
    position: int | None,
    state: str = "locked",
    valid: bool = True,
) -> FramePrediction:
    return FramePrediction(
        sequence_id=sequence.sequence_id,
        source=sequence.source,
        split=sequence.split,
        timestamp_s=timestamp,
        state=state,
        position=position,
        confidence=0.8 if position is not None else 0.0,
        observation_valid=valid,
    )


def test_checked_in_manifest_is_public_source_disjoint_and_covers_target_events() -> (
    None
):
    manifest = load_manifest(default_manifest_path())

    assert {sequence.split for sequence in manifest.sequences} == {"dev", "test"}
    for split in ("dev", "test"):
        labels = [
            label
            for sequence in manifest.sequences
            if sequence.split == split
            for label in sequence.labels
        ]
        assert {
            label.position for label in labels if label.position is not None
        }.issuperset({1, 3, 5, 7, 9})
        assert {label.state for label in labels}.issuperset(
            {"stable", "shifting", "dropout", "invalid"}
        )
    assert any(
        label.technique == "barre"
        for sequence in manifest.sequences
        if sequence.split == "test"
        for label in sequence.labels
    )


def test_manifest_rejects_source_leakage_and_overlapping_labels() -> None:
    dev = _sequence("dev", "same", "dev", (_label(0.0, 1.0),))
    test = _sequence("test", "same", "test", (_label(0.0, 1.0),))
    with pytest.raises(ValueError, match="crosses dev/test"):
        validate_manifest(_manifest(dev, test))

    overlap = _sequence(
        "overlap",
        "dev-source",
        "dev",
        (_label(0.0, 1.0), _label(0.5, 1.5)),
    )
    other = _sequence("other", "test-source", "test", (_label(0.0, 1.0),))
    with pytest.raises(ValueError, match="overlapping"):
        validate_manifest(_manifest(overlap, other))


def test_scorer_reports_precision_coverage_false_locks_and_negative_controls() -> None:
    dev = _sequence("dev", "dev-source", "dev", (_label(0.0, 0.4),))
    invalid = _sequence(
        "invalid",
        "dev-source",
        "dev",
        (_label(1.0, 1.2, state="invalid", position=None),),
    )
    test = _sequence("test", "test-source", "test", (_label(0.0, 0.1),))
    manifest = _manifest(dev, invalid, test)
    predictions = [
        _prediction(dev, 0.0, position=5),
        _prediction(dev, 0.1, position=6),
        _prediction(dev, 0.2, position=None, state="lost", valid=False),
        _prediction(dev, 0.3, position=5, state="holding", valid=False),
        _prediction(invalid, 1.0, position=2),
        _prediction(invalid, 1.1, position=None, state="lost", valid=False),
        _prediction(test, 0.0, position=5),
    ]

    metrics = score_predictions(manifest, predictions)
    dev_metrics = metrics["splits"]["dev"]
    assert dev_metrics["displayed_position_precision"]["value"] == pytest.approx(2 / 4)
    assert dev_metrics["coverage"]["value"] == pytest.approx(3 / 4)
    assert dev_metrics["false_lock_rate"]["value"] == pytest.approx(1 / 4)
    assert dev_metrics["valid_observation_rate"]["value"] == pytest.approx(1 / 2)
    assert metrics["negative_control_display_rate"]["value"] == pytest.approx(1 / 2)


def test_scorer_measures_shift_latency_and_relock_after_valid_return() -> None:
    shift = _sequence(
        "shift",
        "dev-source",
        "dev",
        (
            _label(0.0, 0.2, position=2),
            _label(0.2, 0.4, state="shifting", position=None),
            _label(0.4, 0.8, position=9),
        ),
    )
    dropout = _sequence(
        "dropout",
        "dev-source",
        "dev",
        (
            _label(1.0, 1.2, position=5),
            _label(
                1.2,
                1.4,
                state="dropout",
                position=5,
                occlusion_rect=(0.4, 0.0, 0.9, 0.8),
            ),
            _label(1.4, 1.8, position=5),
        ),
    )
    test = _sequence("test", "test-source", "test", (_label(0.0, 0.1),))
    manifest = _manifest(shift, dropout, test)
    predictions = [
        _prediction(shift, 0.0, position=2),
        _prediction(shift, 0.1, position=2),
        _prediction(shift, 0.2, position=2, state="holding", valid=False),
        _prediction(shift, 0.3, position=None, state="shifting"),
        _prediction(shift, 0.4, position=None, state="shifting"),
        _prediction(shift, 0.5, position=None, state="acquiring"),
        _prediction(shift, 0.6, position=9),
        _prediction(shift, 0.7, position=9),
        _prediction(dropout, 1.0, position=5),
        _prediction(dropout, 1.1, position=5),
        _prediction(dropout, 1.2, position=5, state="holding", valid=False),
        _prediction(dropout, 1.3, position=None, state="lost", valid=False),
        _prediction(dropout, 1.4, position=5, state="holding", valid=False),
        _prediction(dropout, 1.5, position=5, valid=True),
        _prediction(dropout, 1.6, position=5, valid=True),
        _prediction(dropout, 1.7, position=5, valid=True),
        _prediction(test, 0.0, position=5),
    ]

    metrics = score_predictions(manifest, predictions)
    assert metrics["shift_latency"]["median_s"] == pytest.approx(0.2)
    assert metrics["dropout_recovery"]["median_s"] == pytest.approx(0.1)
    assert metrics["shift_latency"]["origin_not_locked_events"] == 0
    assert metrics["dropout_recovery"]["censored_events"] == 0
    assert metrics["state_display_rates"]["shifting"]["value"] == pytest.approx(1 / 2)


def test_position_labels_require_semantically_valid_positions() -> None:
    dev = _sequence("dev", "dev-source", "dev", (_label(0.0, 1.0),))
    test = _sequence("test", "test-source", "test", (_label(0.0, 1.0),))

    bad_stable = replace(dev, labels=(_label(0.0, 1.0, position=None),))
    with pytest.raises(ValueError, match="stable labels require"):
        validate_manifest(_manifest(bad_stable, test))

    bad_shift = replace(
        dev,
        labels=(_label(0.0, 1.0, state="shifting", position=4),),
    )
    with pytest.raises(ValueError, match="shifting and invalid"):
        validate_manifest(_manifest(bad_shift, test))


def test_manifest_rejects_cold_transitions_invalid_enums_and_unbounded_dropout() -> (
    None
):
    test = _sequence("test", "test-source", "test", (_label(0.0, 1.0),))
    cold_shift = _sequence(
        "cold",
        "dev-source",
        "dev",
        (
            _label(0.0, 0.2, state="shifting", position=None),
            _label(0.2, 1.0, position=9),
        ),
    )
    with pytest.raises(ValueError, match="between stable"):
        validate_manifest(_manifest(cold_shift, test))

    bad_state = replace(
        _label(0.0, 1.0, state="invalid", position=None),
        state="mystery",  # type: ignore[arg-type]
    )
    dev = _sequence("dev", "dev-source", "dev", (bad_state,))
    with pytest.raises(ValueError, match="unsupported label state"):
        validate_manifest(_manifest(dev, test))

    dropout_without_rect = _sequence(
        "dropout",
        "dev-source",
        "dev",
        (
            _label(0.0, 0.2),
            _label(0.2, 0.4, state="dropout", position=5),
            _label(0.4, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="occlusion_rect"):
        validate_manifest(_manifest(dropout_without_rect, test))


def test_scorer_rejects_missing_and_duplicate_prediction_samples() -> None:
    dev = _sequence("dev", "dev-source", "dev", (_label(0.0, 0.2),))
    test = _sequence("test", "test-source", "test", (_label(0.0, 0.1),))
    manifest = _manifest(dev, test)
    complete = [
        _prediction(dev, 0.0, position=5),
        _prediction(dev, 0.1, position=5),
    ]

    with pytest.raises(ValueError, match="grid mismatch"):
        score_predictions(manifest, complete[:1])
    with pytest.raises(ValueError, match="duplicate prediction"):
        score_predictions(manifest, [*complete, complete[-1]])

    second_dev = _sequence(
        "second-dev",
        "second-dev-source",
        "dev",
        (_label(0.0, 0.1),),
    )
    with pytest.raises(ValueError, match="grid mismatch"):
        score_predictions(_manifest(dev, second_dev, test), complete)


def test_unrun_split_does_not_create_censored_transition_events() -> None:
    dev = _sequence(
        "dev",
        "dev-source",
        "dev",
        (
            _label(0.0, 0.2, position=2),
            _label(0.2, 0.4, state="shifting", position=None),
            _label(0.4, 0.6, position=7),
        ),
    )
    test = _sequence("test", "test-source", "test", (_label(0.0, 0.2),))
    manifest = _manifest(dev, test)
    metrics = score_predictions(
        manifest,
        [
            _prediction(test, 0.0, position=5),
            _prediction(test, 0.1, position=5),
        ],
    )

    assert metrics["shift_latency"]["events"] == 0
    assert metrics["shift_latency"]["censored_events"] == 0
    assert metrics["shift_latency"]["origin_not_locked_events"] == 0


def test_transition_requires_a_fresh_valid_origin_lock() -> None:
    dev = _sequence(
        "dev",
        "dev-source",
        "dev",
        (
            _label(0.0, 0.2, position=2),
            _label(0.2, 0.4, state="shifting", position=None),
            _label(0.4, 0.6, position=7),
        ),
    )
    test = _sequence("test", "test-source", "test", (_label(0.0, 0.2),))
    metrics = score_predictions(
        _manifest(dev, test),
        [
            _prediction(dev, 0.0, position=2),
            _prediction(dev, 0.1, position=2, state="holding", valid=False),
            _prediction(dev, 0.2, position=None, state="shifting"),
            _prediction(dev, 0.3, position=None, state="shifting"),
            _prediction(dev, 0.4, position=7),
            _prediction(dev, 0.5, position=7),
        ],
    )

    assert metrics["shift_latency"]["events"] == 0
    assert metrics["shift_latency"]["origin_not_locked_events"] == 1


def test_dropout_transform_masks_only_the_annotated_public_frame_region() -> None:
    frame = np.full((10, 20, 3), 255, dtype=np.uint8)
    label = _label(
        0.0,
        1.0,
        state="dropout",
        position=None,
        occlusion_rect=(0.25, 0.2, 0.75, 0.8),
    )

    masked = _apply_annotation_transform(frame, label)

    assert np.all(masked[2:8, 5:15] == 0)
    assert np.all(masked[:2] == 255)
    assert np.all(frame == 255)


def test_cli_defaults_to_dev_without_opening_the_heldout_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(*_args: object, **kwargs: object) -> list[FramePrediction]:
        observed["splits"] = kwargs["splits"]
        return []

    monkeypatch.setattr(position_benchmark, "run_inference", fake_run)
    monkeypatch.setattr(
        position_benchmark,
        "score_predictions",
        lambda *_args: {"stub": True},
    )

    position_benchmark.main(["--manifest", str(default_manifest_path())])

    assert observed["splits"] == {"dev"}
