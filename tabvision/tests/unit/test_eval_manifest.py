"""Unit tests for eval manifest validation and debt summaries."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from tabvision.eval.manifest import REQUIRED_TIERS, validate_manifest


def test_missing_manifest_is_optional_for_v1_release(tmp_path: Path) -> None:
    missing = tmp_path / "manifest.toml"

    result = validate_manifest(missing)

    assert result.passed
    assert result.clip_count == 0
    assert result.missing_tiers == list(REQUIRED_TIERS)
    assert any(item.code == "MANIFEST_MISSING" and item.severity == "warn" for item in result.items)
    assert all(item.severity != "fail" for item in result.items)


def test_manifest_reports_missing_required_clip_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "clip-a"
tier = "clean_acoustic_single_line"
source = "GuitarSet"
split = "validation"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    assert not result.passed
    assert result.clip_count == 1
    assert {item.code for item in result.items if item.clip_id == "clip-a"} >= {
        "MISSING_MEDIA_PATH",
        "MISSING_ANNOTATION_PATH",
    }


def test_manifest_validation_is_json_serializable_and_sorted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "b"
tier = "distorted_electric"
source = "EGDB"
split = "test"
media_path = "$TABVISION_DATA_ROOT/egdb/b.wav"
annotation_path = "$TABVISION_DATA_ROOT/egdb/b.gp5"
annotation_format = "egdb_gp"

[[clips]]
id = "a"
tier = "clean_acoustic_strummed"
source = "GuitarSet"
split = "validation"
media_path = "$TABVISION_DATA_ROOT/guitarset/a.wav"
annotation_path = "$TABVISION_DATA_ROOT/guitarset/a.jams"
annotation_format = "guitarset_jams"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first = validate_manifest(manifest).to_json_bytes()
    second = validate_manifest(manifest).to_json_bytes()

    assert first == second
    payload = json.loads(first)
    assert payload["clip_ids"] == ["a", "b"]
    assert payload["present_tiers"] == ["clean_acoustic_strummed", "distorted_electric"]
    assert payload["passed"] is True
    assert tomllib.loads(manifest.read_text(encoding="utf-8"))["clips"][0]["id"] == "b"


def test_annotation_format_is_required(tmp_path: Path) -> None:
    """Phase 0: every clip must declare its parser dispatch key."""
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "missing-format"
tier = "clean_acoustic_strummed"
source = "GuitarSet"
split = "validation"
media_path = "$TABVISION_DATA_ROOT/guitarset/a.wav"
annotation_path = "$TABVISION_DATA_ROOT/guitarset/a.jams"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    assert not result.passed
    assert any(
        item.code == "MISSING_ANNOTATION_FORMAT" and item.severity == "fail"
        for item in result.items
    )


def test_synthetic_source_blocked_in_test_split(tmp_path: Path) -> None:
    """Cross-contamination guard: synthetic-source clip in test split is rejected."""
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "synth-in-test"
tier = "clean_electric"
source = "synthtab/electric"
split = "test"
media_path = "$TABVISION_DATA_ROOT/synthtab/x.wav"
annotation_path = "$TABVISION_DATA_ROOT/synthtab/x.json"
annotation_format = "synthtab_json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    assert not result.passed
    failures = [
        item
        for item in result.items
        if item.code == "SYNTHETIC_IN_EVAL_SPLIT" and item.severity == "fail"
    ]
    assert len(failures) == 1
    assert failures[0].clip_id == "synth-in-test"


def test_synthetic_source_blocked_in_validation_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "synth-in-validation"
tier = "clean_electric"
source = "DadaGP/render-001"
split = "validation"
media_path = "$TABVISION_DATA_ROOT/dadagp/x.wav"
annotation_path = "$TABVISION_DATA_ROOT/dadagp/x.json"
annotation_format = "dadagp_json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    failures = [
        item
        for item in result.items
        if item.code == "SYNTHETIC_IN_EVAL_SPLIT" and item.severity == "fail"
    ]
    assert len(failures) == 1
    assert failures[0].clip_id == "synth-in-validation"


def test_synthetic_source_allowed_in_train_split(tmp_path: Path) -> None:
    """Synthetic data is permitted as training material (per design plan §4.2)."""
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "synth-in-train"
tier = "clean_electric"
source = "synthtab/electric"
split = "train"
media_path = "$TABVISION_DATA_ROOT/synthtab/x.wav"
annotation_path = "$TABVISION_DATA_ROOT/synthtab/x.json"
annotation_format = "synthtab_json"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    assert not any(item.code == "SYNTHETIC_IN_EVAL_SPLIT" for item in result.items)


def test_guitartechs_highres_smoke_manifest_is_valid() -> None:
    manifest = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "eval"
        / "guitartechs_highres_smoke.toml"
    )

    result = validate_manifest(manifest)

    assert result.passed
    assert result.clip_count == 1
    assert result.present_tiers == ["clean_electric"]
    assert result.clip_ids == ["guitar-techs/P1_chords/midi/midi_Set1_aug"]
