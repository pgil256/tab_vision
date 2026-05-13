"""Integration smoke tests for the composite-eval harness (Phase 0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabvision.eval.composite import (
    Predictor,
    run_composite_eval,
)
from tabvision.types import SessionConfig, TabEvent

# Standard tuning open pitches for derived MIDI.
_OPEN_PITCH = (40, 45, 50, 55, 59, 64)


def _write_jams(
    path: Path,
    notes: list[tuple[float, float, int, int]],
) -> None:
    """Write a minimal GuitarSet-style JAMS at ``path``.

    Each ``notes`` tuple is ``(onset_s, duration_s, string_idx, fret)``.
    """
    by_string: dict[int, list[dict[str, float]]] = {}
    for onset, duration, string_idx, fret in notes:
        midi = _OPEN_PITCH[string_idx] + fret
        by_string.setdefault(string_idx, []).append(
            {"time": float(onset), "duration": float(duration), "value": float(midi)}
        )
    payload = {
        "annotations": [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": str(string_idx)},
                "data": data,
            }
            for string_idx, data in sorted(by_string.items())
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tab_event(onset: float, duration: float, string_idx: int, fret: int) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=duration,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=_OPEN_PITCH[string_idx] + fret,
        confidence=1.0,
    )


def _write_manifest(
    manifest_path: Path,
    entries: list[dict[str, str]],
) -> None:
    """Build a TOML manifest from a list of clip-dict entries."""
    lines: list[str] = []
    for entry in entries:
        lines.append("[[clips]]")
        for key, value in entry.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def _make_predictor(gold_by_path: dict[str, list[TabEvent]]) -> Predictor:
    """Return a predictor that echoes gold for each known path."""

    def predict(media_path: Path, session: SessionConfig) -> list[TabEvent]:
        del session
        key = str(media_path)
        if key not in gold_by_path:
            raise KeyError(f"unknown media path in test: {key}")
        return list(gold_by_path[key])

    return predict


def _shifted_predictor(gold_by_path: dict[str, list[TabEvent]]) -> Predictor:
    """Return a predictor that shifts every event to a different string with the same pitch."""

    def predict(media_path: Path, session: SessionConfig) -> list[TabEvent]:
        del session
        gold = gold_by_path[str(media_path)]
        out: list[TabEvent] = []
        for event in gold:
            for candidate_string in range(6):
                if candidate_string == event.string_idx:
                    continue
                fret = event.pitch_midi - _OPEN_PITCH[candidate_string]
                if 0 <= fret <= 24:
                    out.append(
                        TabEvent(
                            onset_s=event.onset_s,
                            duration_s=event.duration_s,
                            string_idx=candidate_string,
                            fret=fret,
                            pitch_midi=event.pitch_midi,
                            confidence=event.confidence,
                        )
                    )
                    break
        return out

    return predict


def _build_two_tier_manifest(tmp_path: Path) -> tuple[Path, dict[str, list[TabEvent]]]:
    """Two clips in clean_acoustic_strummed + one in clean_acoustic_single_line.

    Returns (manifest_path, gold_by_media_path).
    """
    # Mid-range pitches so the shifted_predictor in tests below can find a
    # legal alternate string (low pitches like low-E fret 3 can only live on
    # string 0; shifting them yields no prediction).
    clips = [
        (
            "guitarset-strum-01",
            "clean_acoustic_strummed",
            [(0.0, 0.5, 0, 7), (0.0, 0.5, 1, 7), (0.0, 0.5, 2, 7)],
        ),
        (
            "guitarset-strum-02",
            "clean_acoustic_strummed",
            [(1.0, 0.4, 3, 5), (1.5, 0.4, 4, 5)],
        ),
        (
            "guitarset-single-01",
            "clean_acoustic_single_line",
            [(0.0, 0.2, 2, 5), (0.5, 0.2, 2, 7), (1.0, 0.2, 2, 9)],
        ),
    ]

    gold_by_path: dict[str, list[TabEvent]] = {}
    entries: list[dict[str, str]] = []
    for clip_id, tier, notes in clips:
        jams_path = tmp_path / f"{clip_id}.jams"
        media_path = tmp_path / f"{clip_id}.wav"
        media_path.write_bytes(b"")  # zero-byte placeholder; predictor doesn't read it
        _write_jams(jams_path, notes)
        gold_by_path[str(media_path)] = [
            _tab_event(o, d, s, f) for (o, d, s, f) in notes
        ]
        entries.append(
            {
                "id": clip_id,
                "tier": tier,
                "source": "GuitarSet",
                "split": "validation",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                "annotation_format": "guitarset_jams",
            }
        )

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(manifest_path, entries)
    return manifest_path, gold_by_path


def test_perfect_predictor_yields_pass_on_both_tiers(tmp_path: Path) -> None:
    manifest_path, gold_by_path = _build_two_tier_manifest(tmp_path)
    predictor = _make_predictor(gold_by_path)

    report = run_composite_eval(
        manifest_path,
        predictor=predictor,
        bootstrap_n=500,
        bootstrap_seed=42,
    )

    assert set(report.tiers) == {
        "clean_acoustic_strummed",
        "clean_acoustic_single_line",
    }
    for tier, tier_report in report.tiers.items():
        assert tier_report.tab_f1.statistic == pytest.approx(1.0), (
            f"tier {tier} should be perfect with echo predictor"
        )
        assert tier_report.onset_f1.statistic == pytest.approx(1.0)
        assert tier_report.pitch_f1.statistic == pytest.approx(1.0)


def test_acceptance_helper_classifies_pass_gap_fail(tmp_path: Path) -> None:
    manifest_path, gold_by_path = _build_two_tier_manifest(tmp_path)
    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor(gold_by_path),
        bootstrap_n=500,
    )

    targets = {
        "clean_acoustic_strummed": 0.90,
        "clean_acoustic_single_line": 0.85,
        "clean_electric": 0.87,  # not in manifest
    }
    statuses = report.tab_f1_acceptance(targets)
    assert statuses["clean_acoustic_strummed"] == "pass"
    assert statuses["clean_acoustic_single_line"] == "pass"
    assert statuses["clean_electric"] == "missing"


def test_shifted_predictor_populates_wrong_position_bucket(tmp_path: Path) -> None:
    """Every prediction same-pitch different-string → fills wrong_position_same_pitch."""
    manifest_path, gold_by_path = _build_two_tier_manifest(tmp_path)
    predictor = _shifted_predictor(gold_by_path)

    report = run_composite_eval(
        manifest_path,
        predictor=predictor,
        bootstrap_n=500,
    )

    strum = report.tiers["clean_acoustic_strummed"].errors
    # All predictions are pitch-correct but position-wrong: zero correct,
    # all in the wrong_position bucket.
    assert strum.correct == 0
    assert strum.wrong_position_same_pitch > 0
    assert strum.pitch_off == 0
    assert strum.missed_onset == 0


def test_train_clips_skipped_by_default(tmp_path: Path) -> None:
    """A train-split clip should not appear in per_clip results."""
    jams_path = tmp_path / "train.jams"
    media_path = tmp_path / "train.wav"
    media_path.write_bytes(b"")
    _write_jams(jams_path, [(0.0, 0.2, 0, 0)])

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "train-01",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "train",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                "annotation_format": "guitarset_jams",
            }
        ],
    )

    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor({}),
        bootstrap_n=100,
    )

    assert report.per_clip == []
    assert report.tiers == {}


def test_explicit_train_split_includes_train_clips(tmp_path: Path) -> None:
    jams_path = tmp_path / "train.jams"
    media_path = tmp_path / "train.wav"
    media_path.write_bytes(b"")
    notes = [(0.0, 0.2, 0, 0)]
    _write_jams(jams_path, notes)

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "train-01",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "train",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                "annotation_format": "guitarset_jams",
            }
        ],
    )

    gold = {str(media_path): [_tab_event(o, d, s, f) for (o, d, s, f) in notes]}
    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor(gold),
        splits=("train",),
        bootstrap_n=100,
    )

    assert len(report.per_clip) == 1
    assert report.per_clip[0].clip_id == "train-01"


def test_rejects_manifest_with_fail_issues(tmp_path: Path) -> None:
    """Missing required field (annotation_format) should block the eval."""
    jams_path = tmp_path / "clip.jams"
    media_path = tmp_path / "clip.wav"
    media_path.write_bytes(b"")
    _write_jams(jams_path, [(0.0, 0.2, 0, 0)])

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "clip-no-format",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "validation",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                # annotation_format intentionally omitted
            }
        ],
    )

    with pytest.raises(ValueError, match="fail-severity"):
        run_composite_eval(
            manifest_path,
            predictor=_make_predictor({}),
            bootstrap_n=100,
        )


def test_unknown_parser_format_raises(tmp_path: Path) -> None:
    """A manifest referencing an unregistered parser should raise KeyError at dispatch."""
    jams_path = tmp_path / "clip.jams"
    media_path = tmp_path / "clip.wav"
    media_path.write_bytes(b"")
    _write_jams(jams_path, [(0.0, 0.2, 0, 0)])

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "weird",
                "tier": "clean_acoustic_single_line",
                "source": "Unknown",
                "split": "validation",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                "annotation_format": "non_existent_format",
            }
        ],
    )

    with pytest.raises(KeyError, match="non_existent_format"):
        run_composite_eval(
            manifest_path,
            predictor=_make_predictor({}),
            bootstrap_n=100,
        )


def test_data_root_substitution_uses_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """$TABVISION_DATA_ROOT in paths is expanded via env var when no override."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    jams_path = data_root / "clip.jams"
    media_path = data_root / "clip.wav"
    media_path.write_bytes(b"")
    _write_jams(jams_path, [(0.0, 0.2, 0, 0)])

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "with-root",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "validation",
                "media_path": "$TABVISION_DATA_ROOT/clip.wav",
                "annotation_path": "$TABVISION_DATA_ROOT/clip.jams",
                "annotation_format": "guitarset_jams",
            }
        ],
    )

    monkeypatch.setenv("TABVISION_DATA_ROOT", str(data_root))
    gold = {str(media_path): [_tab_event(0.0, 0.2, 0, 0)]}

    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor(gold),
        bootstrap_n=100,
    )

    assert len(report.per_clip) == 1


def test_data_root_substitution_uses_function_arg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``annotation_root`` arg overrides the env var."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    jams_path = real_root / "clip.jams"
    media_path = real_root / "clip.wav"
    media_path.write_bytes(b"")
    _write_jams(jams_path, [(0.0, 0.2, 0, 0)])

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "rooted",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "validation",
                "media_path": "$TABVISION_DATA_ROOT/clip.wav",
                "annotation_path": "$TABVISION_DATA_ROOT/clip.jams",
                "annotation_format": "guitarset_jams",
            }
        ],
    )

    monkeypatch.setenv("TABVISION_DATA_ROOT", "/nonexistent")
    gold = {str(media_path): [_tab_event(0.0, 0.2, 0, 0)]}

    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor(gold),
        media_root=str(real_root),
        annotation_root=str(real_root),
        bootstrap_n=100,
    )

    assert len(report.per_clip) == 1


def test_per_clip_metrics_include_error_decomposition(tmp_path: Path) -> None:
    """Each ClipEvalResult should carry the 7-bucket decomposition."""
    manifest_path, gold_by_path = _build_two_tier_manifest(tmp_path)
    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor(gold_by_path),
        bootstrap_n=100,
    )

    for clip_result in report.per_clip:
        # Echo predictor → all gold notes should be correct
        assert clip_result.errors.correct == clip_result.n_gold
        assert clip_result.errors.total_loss == 0


def test_clip_with_no_gold_or_predictions(tmp_path: Path) -> None:
    """Empty-gold clip should not break aggregation; F1 is 0 by convention."""
    jams_path = tmp_path / "empty.jams"
    jams_path.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    media_path = tmp_path / "empty.wav"
    media_path.write_bytes(b"")

    manifest_path = tmp_path / "composite.toml"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "empty-clip",
                "tier": "clean_acoustic_single_line",
                "source": "GuitarSet",
                "split": "validation",
                "media_path": str(media_path),
                "annotation_path": str(jams_path),
                "annotation_format": "guitarset_jams",
            }
        ],
    )

    report = run_composite_eval(
        manifest_path,
        predictor=_make_predictor({str(media_path): []}),
        bootstrap_n=100,
    )

    assert len(report.per_clip) == 1
    assert report.per_clip[0].tab.f1 == 0.0
